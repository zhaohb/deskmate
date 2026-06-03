"""Runtime progress for first-run model downloads vs. cached loads.

DeskMate pulls a few local models on first use (faster-whisper, Silero VAD,
speaker embedders). The first run downloads them (which can take minutes and
otherwise looks like a hang); later runs just load the cached copy. This module
makes that visible:

    with loading("Whisper (small)", cached=whisper_cached("small")):
        model = WhisperModel("small", ...)

It prints a one-line status to stderr (so it shows up even when file logging is
on) and mirrors it to the logger. ``cached=True`` says "found locally, loading";
``cached=False`` says "downloading, first run". Ollama models are intentionally
out of scope here.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .logger import get

logger = get("model_status")


def _emit(message: str) -> None:
    """Print a status line to stderr so users see it during long startups."""
    print(message, file=sys.stderr, flush=True)


@contextmanager
def loading(name: str, *, cached: bool, detail: str = "") -> Iterator[None]:
    """Report the load of a model and whether it had to be downloaded first.

    Wrap the actual model-construction call. On success prints the elapsed
    time; on failure prints how long it ran before erroring and re-raises.
    """
    suffix = f" ({detail})" if detail else ""
    if cached:
        _emit(f"[model] {name}: found locally — loading{suffix}…")
        logger.info("loading cached model: %s%s", name, suffix)
    else:
        _emit(
            f"[model] {name}: not cached — downloading now"
            f" (first run only, may take several minutes){suffix}…"
        )
        logger.info("downloading model: %s%s", name, suffix)

    start = time.perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        _emit(f"[model] {name}: failed after {elapsed:.1f}s — {exc}")
        logger.warning("model %s failed after %.1fs: %s", name, elapsed, exc)
        raise
    elapsed = time.perf_counter() - start
    verb = "downloaded and loaded" if not cached else "loaded"
    _emit(f"[model] {name}: {verb} in {elapsed:.1f}s — ready.")
    logger.info("model %s ready in %.1fs (cached=%s)", name, elapsed, cached)


def hf_cached(repo_id: str, filenames: tuple[str, ...] = ("config.json",)) -> bool:
    """True if any of ``filenames`` for ``repo_id`` is in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        # Can't tell — assume not cached so the user still gets a heads-up.
        return False
    for filename in filenames:
        try:
            result = try_to_load_from_cache(repo_id, filename)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(result, str):
            return True
    return False


def whisper_cached(model_size: str) -> bool:
    """True if a faster-whisper model is already present locally.

    Handles three forms of ``model_size``: a local directory path, a full HF
    repo id (``owner/name``), or a bare size like ``small`` / ``large-v3``.
    """
    if not model_size:
        return False
    if Path(model_size).expanduser().exists():
        return True
    if "/" in model_size:
        candidates = [model_size]
    else:
        candidates = [
            f"Systran/faster-whisper-{model_size}",
            f"guillaumekln/faster-whisper-{model_size}",
        ]
    return any(hf_cached(repo, ("model.bin", "config.json")) for repo in candidates)
