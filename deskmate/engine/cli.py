"""Command-line entry point for DeskMate."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import typer
import uvicorn

from .. import paths
from ..config import load as load_config
from ..console import echo
from ..db import DatabaseManager
from .api import create_app
from .daemon import Daemon

app = typer.Typer(add_completion=False, help="DeskMate — local desktop activity recorder")


@app.command()
def record() -> None:
    """Start the recording daemon (capture + a11y + audio + retention).

    Does NOT start the HTTP API. Use `serve` for that, or run both:
        deskmate record & deskmate serve
    """
    Daemon().run_forever()


@app.command()
def serve(host: str | None = None, port: int | None = None, run_daemon: bool = True) -> None:
    """Start the HTTP API. By default also runs the recording daemon."""
    cfg = load_config()
    db = DatabaseManager()
    d: Daemon | None = None
    if run_daemon:
        d = Daemon(cfg=cfg, db=db)
        d.start()
        if cfg.audio.enabled and d.transcriber and not d.transcriber.available and d.transcriber.user_hint:
            echo(f"Audio transcription: {d.transcriber.user_hint}", err=True)
    try:
        uvicorn.run(
            create_app(cfg=cfg, db=db, daemon=d),
            host=host or cfg.server.host,
            port=port or cfg.server.port,
            log_level="info",
        )
    finally:
        if d:
            d.stop()


@app.command()
def ui(host: str | None = None, port: int | None = None, run_daemon: bool = True) -> None:
    """Start the HTTP API and open the browser UI at /ui."""
    cfg = load_config()
    target_host = host or cfg.server.host
    target_port = port or cfg.server.port
    url = f"http://{target_host}:{target_port}/ui"

    def _open_browser() -> None:
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, name="open-ui-browser", daemon=True).start()
    echo(f"opening {url}")
    serve(host=target_host, port=target_port, run_daemon=run_daemon)


@app.command()
def search(query: str, limit: int = 10, app_name: str | None = None, semantic: bool = False) -> None:
    """Query the HTTP API. The daemon must be running (`deskmate serve`)."""
    cfg = load_config()
    url = f"http://{cfg.server.host}:{cfg.server.port}/search"
    r = httpx.get(
        url,
        params={"q": query, "limit": limit, "app_name": app_name, "semantic": semantic},
    )
    r.raise_for_status()
    echo(json.dumps(r.json(), indent=2, ensure_ascii=False))


@app.command()
def index(
    batch_size: int | None = None,
    max_rows: int | None = None,
) -> None:
    """Build the semantic (vector) search index over existing content.

    Embeds OCR/transcript/UI text that hasn't been indexed yet. Safe to re-run;
    it only processes new rows. Requires the optional `[semantic]` extra.
    """
    cfg = load_config()
    db = DatabaseManager()
    model = cfg.search.embedding_model
    pending = db.semantic_pending_count(model_name=model, min_chars=cfg.search.min_chars)
    if pending == 0:
        echo("semantic index is up to date — nothing to do")
        return
    echo(f"indexing {pending} item(s) with {model}…")

    def _progress(content_type: str, done: int) -> None:
        echo(f"  {content_type}: {done} embedded", err=True)

    indexed = db.build_semantic_index(
        model_name=model,
        batch_size=batch_size or cfg.search.index_batch,
        min_chars=cfg.search.min_chars,
        max_rows=max_rows,
        progress=_progress,
    )
    if indexed == 0:
        echo(
            "no rows indexed — is the semantic extra installed? "
            "pip install 'deskmate[semantic]'",
            err=True,
        )
    else:
        echo(f"done — embedded {indexed} item(s)")



@app.command()
def capture_once() -> None:
    """One-shot paired capture for debugging."""
    cfg = load_config()
    db = DatabaseManager()
    from ..capture import paired_capture  # noqa: PLC0415

    ids = paired_capture(cfg, db, trigger="manual")
    echo(json.dumps({"frame_ids": ids}, ensure_ascii=False))


@app.command()
def health() -> None:
    cfg = load_config()
    r = httpx.get(f"http://{cfg.server.host}:{cfg.server.port}/health")
    echo(r.text)


@app.command()
def mcp() -> None:
    """Run the MCP stdio server (talks to a running deskmate HTTP API)."""
    from ..mcp.server import run_stdio  # noqa: PLC0415

    run_stdio()


@app.command("train-lora")
def train_lora(
    model: str | None = None,
    output_dir: str | None = None,
    sources: str | None = None,
    epochs: int | None = None,
    max_pairs: int | None = None,
    dry_run: bool = False,
    export: str | None = None,
) -> None:
    """Fine-tune a local model with LoRA on DeskMate-derived SFT pairs.

    Mines (input, output) pairs from useful habit suggestions, successful pipe
    runs and the unified timeline, then trains LoRA adapters. Use ``--dry-run``
    to preview the mined data without training. Requires `pip install
    'deskmate[training]'` to actually train.
    """
    from ..learning.training import (  # noqa: PLC0415
        DeskMateTrainingDataMiner,
        LoRATrainer,
        LoRATrainingConfig,
        missing_training_deps,
    )

    cfg = load_config().training
    src = (
        [s.strip() for s in sources.split(",") if s.strip()]
        if sources
        else list(cfg.sources)
    )

    miner = DeskMateTrainingDataMiner(min_feedback=cfg.min_feedback, min_chars=cfg.min_chars)
    try:
        breakdown = miner.source_breakdown(sources=src, limit_per_source=cfg.limit_per_source)
        pairs = miner.extract_sft_pairs(
            sources=src,
            limit_per_source=cfg.limit_per_source,
            max_pairs=max_pairs or cfg.max_pairs,
        )
    finally:
        miner.close()

    echo(f"mined {len(pairs)} SFT pair(s) from {src} (per-source: {breakdown})")

    if export:
        out = Path(export)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        echo(f"exported {len(pairs)} pair(s) to {out}")
        return

    if dry_run:
        echo(json.dumps(pairs[:5], ensure_ascii=False, indent=2))
        return
    if not pairs:
        echo("no training data — nothing to do")
        return
    missing = missing_training_deps()
    if missing:
        echo(
            f"training deps not installed (missing {', '.join(missing)}) — "
            "run: pip install 'deskmate[training]'",
            err=True,
        )
        raise typer.Exit(code=1)

    out_dir = output_dir or cfg.output_dir or str(paths.root() / "checkpoints" / "lora")
    lcfg = LoRATrainingConfig(
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        num_epochs=epochs or cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        max_seq_length=cfg.max_seq_length,
        use_4bit=cfg.use_4bit,
        output_dir=out_dir,
    )
    trainer = LoRATrainer(lcfg, model_name=model or cfg.model_name)
    echo(f"training {cfg.model_name} → {out_dir} …")
    summary = trainer.train(pairs)
    echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
