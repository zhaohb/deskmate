"""Command-line entry point for DeskMate."""

from __future__ import annotations

import json
import threading
import time
import webbrowser

import httpx
import typer
import uvicorn

from ..config import load as load_config
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
            typer.echo(f"Audio transcription: {d.transcriber.user_hint}", err=True)
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
    typer.echo(f"opening {url}")
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
    typer.echo(json.dumps(r.json(), indent=2, ensure_ascii=False))


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
        typer.echo("semantic index is up to date — nothing to do")
        return
    typer.echo(f"indexing {pending} item(s) with {model}…")

    def _progress(content_type: str, done: int) -> None:
        typer.echo(f"  {content_type}: {done} embedded", err=True)

    indexed = db.build_semantic_index(
        model_name=model,
        batch_size=batch_size or cfg.search.index_batch,
        min_chars=cfg.search.min_chars,
        max_rows=max_rows,
        progress=_progress,
    )
    if indexed == 0:
        typer.echo(
            "no rows indexed — is the semantic extra installed? "
            "pip install 'deskmate[semantic]'",
            err=True,
        )
    else:
        typer.echo(f"done — embedded {indexed} item(s)")



@app.command()
def capture_once() -> None:
    """One-shot paired capture for debugging."""
    cfg = load_config()
    db = DatabaseManager()
    from ..capture import paired_capture  # noqa: PLC0415

    ids = paired_capture(cfg, db, trigger="manual")
    typer.echo(json.dumps({"frame_ids": ids}, ensure_ascii=False))


@app.command()
def health() -> None:
    cfg = load_config()
    r = httpx.get(f"http://{cfg.server.host}:{cfg.server.port}/health")
    typer.echo(r.text)


@app.command()
def mcp() -> None:
    """Run the MCP stdio server (talks to a running deskmate HTTP API)."""
    from ..mcp.server import run_stdio  # noqa: PLC0415

    run_stdio()


if __name__ == "__main__":
    app()
