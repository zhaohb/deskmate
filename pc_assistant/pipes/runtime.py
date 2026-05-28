"""Local pipe runtime.

This implements the pipe lifecycle for pc_assistant: durable execution rows,
per-run output directories, permission context, timeouts and Python/JavaScript
subprocess execution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .. import paths
from ..capture import paired_capture
from ..config import Config
from ..db import DatabaseManager
from ..logger import get
from .loader import Pipe

logger = get("pipes.runtime")


class PipeRuntime:
    def __init__(
        self,
        db: DatabaseManager,
        cfg: Config | None = None,
        *,
        timeout_seconds: int = 300,
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.timeout_seconds = timeout_seconds

    def run(self, pipe: Pipe, *, trigger: str = "manual") -> int:
        started = time.time()
        execution_id = self.db.insert_pipe_execution(
            pipe_name=pipe.frontmatter.name,
            status="running",
            output="",
        )
        output_dir = paths.pipes_dir() / pipe.frontmatter.name / "output" / str(execution_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            context_path = self._write_context(pipe, execution_id, output_dir, trigger=trigger)
            result = self._run_body(pipe, execution_id, output_dir, context_path)
            status = "success" if result.returncode == 0 else "failed"
            output = json.dumps({
                "runtime": pipe.frontmatter.runtime,
                "trigger": trigger,
                "exit_code": result.returncode,
                "duration_ms": int((time.time() - started) * 1000),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_dir": str(output_dir),
            }, ensure_ascii=False)
            self.db.finish_pipe_execution(
                execution_id,
                status=status,
                output=output,
                session_path=str(output_dir),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pipe %s failed: %s", pipe.frontmatter.name, exc)
            self.db.finish_pipe_execution(
                execution_id,
                status="failed",
                output=json.dumps({
                    "error": str(exc),
                    "trigger": trigger,
                    "duration_ms": int((time.time() - started) * 1000),
                    "output_dir": str(output_dir),
                }, ensure_ascii=False),
                session_path=str(output_dir),
            )
        return execution_id

    def _write_context(self, pipe: Pipe, execution_id: int, output_dir: Path, *, trigger: str) -> Path:
        perms = pipe.frontmatter.permissions
        context = {
            "pipe_name": pipe.frontmatter.name,
            "execution_id": execution_id,
            "trigger": trigger,
            "permissions": asdict(perms),
            "output_dir": str(output_dir),
            "db_path": str(self.db.path) if perms.read_db else None,
            "api_base": self._api_base(),
        }
        if perms.trigger_capture and self.cfg is not None:
            frame_ids = paired_capture(self.cfg, self.db, trigger=f"pipe/{pipe.frontmatter.name}")
            context["triggered_frame_ids"] = frame_ids
        context_path = output_dir / "context.json"
        context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        return context_path

    def _run_body(
        self,
        pipe: Pipe,
        execution_id: int,
        output_dir: Path,
        context_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        runtime = (pipe.frontmatter.runtime or "none").lower()
        if runtime == "none":
            return subprocess.CompletedProcess(args=["none"], returncode=0, stdout="(no-op pipe)", stderr="")
        if runtime == "python":
            script = output_dir / "pipe.py"
            script.write_text(pipe.body, encoding="utf-8")
            return self._run_subprocess([sys.executable, str(script)], pipe, execution_id, output_dir, context_path)
        if runtime in {"js", "javascript", "node"}:
            node = shutil.which("node")
            if not node:
                raise RuntimeError("node runtime not found on PATH")
            script = output_dir / "pipe.mjs"
            script.write_text(pipe.body, encoding="utf-8")
            return self._run_subprocess([node, str(script)], pipe, execution_id, output_dir, context_path)
        raise RuntimeError(f"unsupported pipe runtime: {pipe.frontmatter.runtime}")

    def _run_subprocess(
        self,
        cmd: list[str],
        pipe: Pipe,
        execution_id: int,
        output_dir: Path,
        context_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            **dict(os.environ),
            "PC_ASSISTANT_PIPE_NAME": pipe.frontmatter.name,
            "PC_ASSISTANT_PIPE_EXECUTION_ID": str(execution_id),
            "PC_ASSISTANT_PIPE_CONTEXT": str(context_path),
            "PC_ASSISTANT_OUTPUT_DIR": str(output_dir),
            "PC_ASSISTANT_API": self._api_base(),
        }
        if pipe.frontmatter.permissions.read_db:
            env["PC_ASSISTANT_DB"] = str(self.db.path)
        return subprocess.run(  # noqa: S603
            cmd,
            cwd=str(output_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )

    def _api_base(self) -> str:
        if self.cfg is None:
            return "http://127.0.0.1:3030"
        return f"http://{self.cfg.server.host}:{self.cfg.server.port}"
