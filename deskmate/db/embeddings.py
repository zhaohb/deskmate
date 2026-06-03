"""Local text-embedding model for semantic search.

Semantic search needs a way to turn text into vectors so that paraphrases land
near each other in vector space (something a keyword/FTS index can never do).
This module wraps a small, CPU-friendly ONNX embedding model via ``fastembed``
so there is no heavy ``torch`` dependency and no cloud call — embeddings are
computed locally, consistent with the rest of DeskMate.

Everything here degrades gracefully: if ``fastembed`` (or its model download)
is unavailable, :func:`get_embedder` returns ``None`` and callers fall back to
pure keyword search.
"""

from __future__ import annotations

import struct
import threading
from typing import Iterable, Sequence

from ..logger import get
from ..model_status import hf_cached, loading

logger = get("embeddings")

# float32, little-endian — the on-disk layout for embedding BLOBs.
_F32 = "<f"


def vector_to_blob(vector: Sequence[float]) -> bytes:
    """Pack a float vector into a little-endian float32 BLOB."""
    return struct.pack(f"<{len(vector)}f", *vector)


def blob_to_vector(blob: bytes) -> list[float]:
    """Unpack a float32 BLOB back into a Python list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


class EmbeddingModel:
    """Thin wrapper around a ``fastembed`` text embedding model.

    Vectors are L2-normalized, so a plain dot product equals cosine similarity.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> bool:
        """Lazily construct the underlying model. Returns False on failure."""
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            try:
                from fastembed import TextEmbedding  # noqa: PLC0415
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fastembed not available (%s); semantic search disabled. "
                    "Install with: pip install 'deskmate[semantic]'",
                    exc,
                )
                return False
            try:
                # fastembed caches under huggingface_hub; the ONNX model file is
                # named model.onnx / model_optimized.onnx depending on the repo.
                cached = hf_cached(
                    self.model_name,
                    ("onnx/model.onnx", "model.onnx", "model_optimized.onnx"),
                )
                with loading(f"Embeddings ({self.model_name})", cached=cached):
                    self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load embedding model %s: %s", self.model_name, exc)
                self._model = None
                return False
        return True

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Embed a batch of texts. Returns ``None`` if the model is unavailable."""
        if not texts:
            return []
        if not self._ensure():
            return None
        try:
            vectors = [list(map(float, v)) for v in self._model.embed(list(texts))]
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding failed: %s", exc)
            return None
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    def embed_one(self, text: str) -> list[float] | None:
        """Embed a single text. Returns ``None`` if unavailable."""
        result = self.embed([text])
        if not result:
            return None
        return result[0]


# Process-wide singleton, keyed by model name so a config change rebuilds it.
_embedder: EmbeddingModel | None = None
_embedder_lock = threading.Lock()


def get_embedder(model_name: str) -> EmbeddingModel | None:
    """Return a shared :class:`EmbeddingModel`, or ``None`` if it can't load."""
    global _embedder
    with _embedder_lock:
        if _embedder is None or _embedder.model_name != model_name:
            _embedder = EmbeddingModel(model_name)
        embedder = _embedder
    return embedder if embedder._ensure() else None


def iter_batches(items: Iterable, size: int):
    """Yield successive lists of at most ``size`` items."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
