"""Subprocess entry point for one LoRA training run.

Training is launched as a SEPARATE process (not a thread in the long-lived UI
server) so that when it finishes the process exits and the OS reclaims ALL of
its GPU / iGPU memory — no manual ``empty_cache`` can match that. The parent
(``POST /training/lora``) mines the SFT pairs, writes a job file, spawns this
module, awaits it, then reads the result file.

Protocol (both files are JSON/UTF-8):
    python -m deskmate.learning.training._worker <job_file> <result_file>

  job_file   = {"model_name": str, "config": {<LoRATrainingConfig kwargs>},
                "pairs": [{"input": ..., "output": ...}, ...]}
  result_file = the training summary dict on success, or
                {"status": "error", "error": str} on failure.

Exit code 0 = result file holds a summary; non-zero = it holds an error (the
parent also treats a missing/blank result file as an error).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _run(job_path: str, result_path: str) -> int:
    result_file = Path(result_path)

    def _write(obj: dict[str, Any]) -> None:
        result_file.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    try:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        pairs = job.get("pairs") or []
        config_kwargs = job.get("config") or {}
        model_name = job.get("model_name")

        # Import the heavy ML stack only here, inside the child, so the parent
        # UI process never loads torch/unsloth just to spawn a run.
        from .lora import LoRATrainer, LoRATrainingConfig  # noqa: PLC0415

        lcfg = LoRATrainingConfig(**config_kwargs)
        trainer = LoRATrainer(lcfg, model_name=model_name)
        summary = trainer.train(pairs)
        _write(summary)
        return 0
    except Exception as exc:  # noqa: BLE001 - report any failure to the parent
        traceback.print_exc()
        try:
            _write({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001 - last resort, parent handles blank file
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m deskmate.learning.training._worker "
              "<job_file> <result_file>", file=sys.stderr)
        return 2
    return _run(args[0], args[1])


if __name__ == "__main__":
    raise SystemExit(main())
