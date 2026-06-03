"""Speaker identification / diarization.

We expose a simple two-step API:

```
sid = SpeakerIdentifier()
emb = sid.embed(wav_path, start_s, end_s)
speaker_id = sid.match_or_create(db, emb, threshold=0.75)
```

To avoid forcing a 200MB+ download and the pyannote license click-through
during a clean install, this module:

1. Tries `pyannote.audio` if installed (the user opts in via the
   `[speaker]` extra).
2. Tries `speechbrain` if pyannote is unavailable.
3. Falls back to a deterministic 16-dim spectral-feature pseudo-embedding
   so the pipeline still produces *something* and the speakers table
   schema stays exercised. Quality will be poor — clearly logged.

The DB-side embedding aggregation (centroid average / cosine match) is
identical regardless of which backend produced the embedding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..logger import get

logger = get("audio.speaker")


class SpeakerIdentifier:
    def __init__(self) -> None:
        self._backend = "uninitialized"
        self._embedder = None

    def _ensure(self) -> None:
        if self._backend != "uninitialized":
            return
        from ..model_status import hf_cached, loading  # noqa: PLC0415

        try:  # pyannote first (preferred embedder)
            from pyannote.audio import Inference  # type: ignore[import-not-found]
            with loading(
                "Speaker embedder (pyannote/embedding)",
                cached=hf_cached("pyannote/embedding", ("pytorch_model.bin", "config.yaml")),
            ):
                self._embedder = Inference("pyannote/embedding", device="cpu")
            self._backend = "pyannote"
            logger.info("speaker embedder: pyannote")
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import-not-found]
            with loading(
                "Speaker embedder (speechbrain ECAPA)",
                cached=Path("./.cache/spkrec-ecapa").exists()
                or hf_cached("speechbrain/spkrec-ecapa-voxceleb", ("embedding_model.ckpt", "hyperparams.yaml")),
            ):
                self._embedder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="./.cache/spkrec-ecapa",
                    run_opts={"device": "cpu"},
                )
            self._backend = "speechbrain"
            logger.info("speaker embedder: speechbrain (ECAPA)")
            return
        except Exception:  # noqa: BLE001
            pass
        self._backend = "spectral"
        logger.warning("speaker embedder: spectral fallback (low quality)")

    @property
    def backend(self) -> str:
        return self._backend if self._backend != "uninitialized" else "(lazy)"

    def embed(self, wav_path: Path, start_s: float = 0.0, end_s: float | None = None) -> list[float]:
        self._ensure()
        try:
            import soundfile as sf  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except Exception:  # noqa: BLE001
            logger.warning("soundfile/numpy missing; speaker disabled")
            return []
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if end_s is None:
            end_s = len(wav) / sr
        s = max(0, int(start_s * sr))
        e = min(len(wav), int(end_s * sr))
        clip = wav[s:e]
        if clip.size == 0:
            return []

        if self._backend == "pyannote" and self._embedder is not None:
            try:
                import torch  # type: ignore[import-not-found]
                t = torch.from_numpy(clip).unsqueeze(0)
                emb = self._embedder({"waveform": t, "sample_rate": sr}).data  # type: ignore[attr-defined]
                v = emb.flatten().tolist()
                return _normalize(v)
            except Exception as exc:  # noqa: BLE001
                logger.warning("pyannote embed failed: %s", exc)

        if self._backend == "speechbrain" and self._embedder is not None:
            try:
                import torch  # type: ignore[import-not-found]
                t = torch.from_numpy(clip).unsqueeze(0)
                emb = self._embedder.encode_batch(t).squeeze().tolist()
                return _normalize(emb)
            except Exception as exc:  # noqa: BLE001
                logger.warning("speechbrain embed failed: %s", exc)

        # spectral fallback: 16 Mel-ish bands averaged across time
        try:
            import numpy as np  # type: ignore[import-untyped]
            n = 16
            chunks = np.array_split(clip, n)
            energies = [float((c ** 2).mean()) ** 0.5 for c in chunks]
            return _normalize(energies)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spectral embed failed: %s", exc)
            return []

    def match_or_create(self, db, emb: list[float], *, threshold: float = 0.75) -> int | None:
        """Find best matching speaker (cosine ≥ threshold) else create one."""
        if not emb:
            return None
        rows = db._conn.execute(  # noqa: SLF001
            "SELECT id, centroid_json, sample_count FROM speakers WHERE centroid_json IS NOT NULL"
        ).fetchall()

        best_id = None
        best_sim = -1.0
        for row in rows:
            try:
                centroid = json.loads(row["centroid_json"])
            except (TypeError, ValueError):
                continue
            sim = _cosine(emb, centroid)
            if sim > best_sim:
                best_sim, best_id = sim, int(row["id"])

        if best_id is not None and best_sim >= threshold:
            self._update_centroid(db, best_id, emb)
            return best_id

        cur = db._conn.execute(  # noqa: SLF001
            "INSERT INTO speakers(name, centroid_json, sample_count) VALUES (?, ?, ?)",
            ("", json.dumps(emb), 1),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _update_centroid(db, speaker_id: int, emb: list[float]) -> None:
        row = db._conn.execute(  # noqa: SLF001
            "SELECT centroid_json, sample_count FROM speakers WHERE id = ?", (speaker_id,)
        ).fetchone()
        if not row:
            return
        try:
            centroid = json.loads(row["centroid_json"])
        except (TypeError, ValueError):
            centroid = emb
        n = int(row["sample_count"] or 0)
        new = [(c * n + e) / (n + 1) for c, e in zip(centroid, emb)]
        db._conn.execute(  # noqa: SLF001
            """UPDATE speakers
                  SET centroid_json = ?, sample_count = ?, updated_at = datetime('now')
                WHERE id = ?""",
            (json.dumps(new), n + 1, speaker_id),
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]
