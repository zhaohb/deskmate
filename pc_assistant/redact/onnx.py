"""ONNX-backed PII detector with CPU and DirectML execution providers.

The detector is **optional**: we load it lazily and gracefully degrade to
the regex path if onnxruntime or the model file are missing. Pass
`model_path` (and an optional `tokenizer_path`) via
`RedactConfig.onnx_model_path` / `onnx_tokenizer_path`.

This module deliberately does not pick a tokenizer — the caller's model
might be sentencepiece, wordpiece, or pure-char. Provide a `tokenize()`
callable that maps `str → list[int]` if you bring an exotic model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..logger import get

logger = get("redact.onnx")


class OnnxRedactor:
    """Loads an ONNX model and redacts spans tagged as PII.

    Inputs: a string, plus an `entity_threshold` (probability above which a
    token is treated as PII).

    Output: redacted string. Spans are replaced with `[REDACTED]`.
    """

    def __init__(
        self,
        model_path: Path | str | None,
        *,
        tokenizer_path: Path | str | None = None,
        providers: list[str] | None = None,
        tokenize: Callable[[str], list[int]] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.tokenize_fn = tokenize
        # Default execution provider list: try DirectML first on Windows,
        # then CPU. Override with `providers=[...]` if needed.
        self.providers = providers or ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.labels = labels or ["O", "PII"]
        self._session = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        return self._session is not None or self._try_load()

    def _try_load(self) -> bool:
        if self.model_path is None or not self.model_path.exists():
            return False
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            logger.warning("onnxruntime missing (%s) — ONNX redact disabled", exc)
            return False
        try:
            # filter providers to the ones actually available
            available = ort.get_available_providers()
            providers = [p for p in self.providers if p in available] or ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            logger.info("ONNX redactor loaded (%s, providers=%s)", self.model_path.name, providers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONNX session init failed: %s", exc)
            return False

        if self.tokenizer_path and self.tokenize_fn is None:
            try:
                from tokenizers import Tokenizer  # type: ignore[import-not-found]
                self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("tokenizer load failed (%s) — ONNX redact disabled", exc)
                self._session = None
                return False
        return True

    def redact(self, text: str, *, threshold: float = 0.5) -> str:
        """Return `text` with detected PII tokens replaced by `[REDACTED]`."""
        if not text or not self.available:
            return text
        try:
            tokens, offsets = self._encode(text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("tokenize failed (%s) — passing through", exc)
            return text
        if not tokens:
            return text

        try:
            import numpy as np  # type: ignore[import-untyped]
            input_ids = np.asarray([tokens], dtype="int64")
            attention_mask = np.ones_like(input_ids, dtype="int64")
            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            logits = outputs[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONNX inference failed: %s — pass-through", exc)
            return text

        # Simple greedy argmax decode; pick spans whose label-1 probability
        # exceeds `threshold` after softmax.
        try:
            import numpy as np
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp / exp.sum(axis=-1, keepdims=True)
            pii_prob = probs[0, :, 1] if probs.shape[-1] > 1 else probs[0, :, 0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("softmax failed: %s", exc)
            return text

        spans = self._collect_spans(pii_prob, offsets, threshold=threshold)
        return self._splice(text, spans)

    # ─── internals ─────────────────────────────────────────────────────────
    def _encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        if self.tokenize_fn is not None:
            ids = self.tokenize_fn(text)
            return list(ids), [(0, 0)] * len(ids)
        if self._tokenizer is None:
            return [], []
        enc = self._tokenizer.encode(text)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    @staticmethod
    def _collect_spans(probs, offsets, *, threshold: float) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        for prob, off in zip(probs, offsets):
            if off == (0, 0):
                continue
            if float(prob) >= threshold:
                if cur is None:
                    cur = (int(off[0]), int(off[1]))
                else:
                    cur = (cur[0], int(off[1]))
            else:
                if cur is not None:
                    spans.append(cur)
                    cur = None
        if cur is not None:
            spans.append(cur)
        return spans

    @staticmethod
    def _splice(text: str, spans: list[tuple[int, int]]) -> str:
        if not spans:
            return text
        out = []
        cursor = 0
        for s, e in sorted(spans):
            out.append(text[cursor:s])
            out.append("[REDACTED]")
            cursor = e
        out.append(text[cursor:])
        return "".join(out)
