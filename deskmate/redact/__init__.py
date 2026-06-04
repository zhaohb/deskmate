"""PII redaction pipeline.

Two layers:

1. **Fast regex path** (`apply_text` / `maybe_redact`) — synchronous, runs
   inline on every captured text. Implementation in `core.pii`.

2. **ONNX detector path** (`OnnxRedactor` in `.onnx`) — optional, loads an
   ONNX token classifier on CPU or DirectML. Used by the **reconciler** to
   re-scan already-stored text and patch `redacted_text` / `redacted_text_json`
   columns. The reconciliation worker runs on a background thread and walks
   pending frames in age order.

Models are NOT bundled (they are external assets); the redactor accepts a
path via `RedactConfig.onnx_model_path`. Without a model, the regex path
remains active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.pii import remove_pii
from .onnx import OnnxRedactor
from .reconciler import RedactReconciler

if TYPE_CHECKING:
    from ..config import Config


def apply_text(text: str, rules: list[str] | None = None) -> str:
    return remove_pii(text, rules=rules)


def maybe_redact(text: str, cfg: Config) -> str:
    """Convenience: respects `cfg.redact.enabled`. Synchronous regex pass."""
    if not text or not cfg.redact.enabled:
        return text
    return apply_text(text, cfg.redact.rules)


__all__ = ["OnnxRedactor", "RedactReconciler", "apply_text", "maybe_redact"]
