"""Live transcript translation via the local Ollama LLM.

Each spoken utterance (one ``audio_transcriptions`` row) is translated into the
configured target language and pushed to the UI in real time. To keep quality
high despite translating one short utterance at a time, the translator feeds the
LLM a **sliding window of the preceding utterances** as context (for pronoun /
terminology consistency) while instructing it to translate only the current
line. The context is per-device so a meeting's two audio streams (mic vs.
loopback) don't bleed into each other.

This module is intentionally self-contained and fail-soft: if Ollama is
unreachable or the target language already matches the source, it returns
``None`` and the caller simply leaves the transcript untranslated. It never
raises into the audio loop.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

from ..logger import get

logger = get("audio.translator")

# LLM helpers are bound lazily to avoid a circular import (engine.llm pulls in
# engine.__init__ → daemon → audio). They are real module attributes so tests
# can monkeypatch ``deskmate.audio.translator.chat_ollama``. ``_ensure_llm``
# fills any that are still unbound on first use.
chat_ollama = None
strip_thinking = None


def _ensure_llm() -> None:
    global chat_ollama, strip_thinking
    if chat_ollama is None or strip_thinking is None:
        from ..engine import llm  # noqa: PLC0415

        if chat_ollama is None:
            chat_ollama = llm.chat_ollama
        if strip_thinking is None:
            strip_thinking = llm.strip_thinking

# Human-readable names for the prompt so the model gets an unambiguous target
# (qwen handles ISO codes too, but names are more reliable across languages).
_LANG_NAMES = {
    "zh": "Simplified Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
}


def _lang_name(code: str | None) -> str:
    if not code:
        return "the source language"
    return _LANG_NAMES.get(code.lower(), code)


class TranscriptTranslator:
    """Translate transcript utterances with a per-device rolling context window.

    Thread-safe: the audio/translation worker calls :meth:`translate` from a
    background thread; the small context buffers are guarded by a lock.
    """

    def __init__(
        self,
        *,
        target_lang: str = "zh",
        model: str = "",
        skip_if_same: bool = True,
        context_window: int = 2,
    ) -> None:
        from ..engine.llm import resolve_ollama_settings  # noqa: PLC0415

        self.target_lang = (target_lang or "zh").lower()
        self.skip_if_same = skip_if_same
        self.context_window = max(0, context_window)
        base, default_model, timeout = resolve_ollama_settings()
        self._base = base
        self._model = model or default_model
        self._timeout = timeout
        self._lock = threading.Lock()
        # device -> recent source-text utterances (for context).
        self._context: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.context_window or 1)
        )

    def _build_messages(self, text: str, source_lang: str | None, context: list[str]) -> list[dict]:
        target = _lang_name(self.target_lang)
        sys = (
            f"You are a professional simultaneous interpreter. Translate the user's "
            f"line into {target}. Output ONLY the translation — no quotes, no "
            f"explanations, no notes, no source text. Preserve names, numbers and "
            f"technical terms. If the line is already in {target}, repeat it unchanged."
        )
        parts: list[str] = []
        if context:
            joined = "\n".join(context)
            parts.append(
                "Conversation so far (context only — do NOT translate or repeat "
                f"this):\n{joined}\n"
            )
        parts.append(f"Translate this line into {target}:\n{text}")
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": "\n".join(parts)},
        ]

    def translate(self, text: str, *, source_lang: str | None, device: str = "") -> str | None:
        """Translate one utterance. Returns the translation, or ``None`` to skip.

        ``None`` means "leave untranslated": empty input, already-target language
        (when ``skip_if_same``), or an LLM/transport failure. The current text is
        always appended to the device context window afterwards so subsequent
        lines have it as context even when this line itself was skipped.
        """
        clean = (text or "").strip()
        if not clean:
            return None

        if self.skip_if_same and source_lang and source_lang.lower() == self.target_lang:
            self._remember(device, clean)
            return None

        context = self._snapshot_context(device)
        _ensure_llm()
        try:
            message = chat_ollama(
                self._build_messages(clean, source_lang, context),
                base=self._base,
                model=self._model,
                num_predict=512,
                temperature=0.2,
                timeout=self._timeout,
            )
            translation = strip_thinking(message.get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("translation failed (%s); leaving untranslated", exc)
            self._remember(device, clean)
            return None

        self._remember(device, clean)
        if not translation:
            return None
        return translation

    def _snapshot_context(self, device: str) -> list[str]:
        if self.context_window == 0:
            return []
        with self._lock:
            return list(self._context[device])

    def _remember(self, device: str, text: str) -> None:
        if self.context_window == 0:
            return
        with self._lock:
            self._context[device].append(text)

    def reset_context(self, device: str | None = None) -> None:
        """Clear context for one device (or all). Call when a meeting ends."""
        with self._lock:
            if device is None:
                self._context.clear()
            else:
                self._context.pop(device, None)
