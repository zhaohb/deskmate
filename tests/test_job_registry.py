"""In-flight long jobs are reported by the server, so a reload can restore them.

The UI used to track "a recap is generating" only in browser memory, so a
reload — or a second tab — showed nothing while the work was still running.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from deskmate.engine import jobs

APP_JS = Path(__file__).parents[1] / "deskmate" / "ui" / "static" / "app.js"
API_PY = Path(__file__).parents[1] / "deskmate" / "engine" / "api.py"


@pytest.fixture(autouse=True)
def _clean_registry():
    jobs.clear()
    yield
    jobs.clear()


def test_running_job_is_reported_until_it_finishes() -> None:
    assert jobs.running() == []
    with jobs.track("lrn-recap:3", label="openvino", meta={"session_id": 3}):
        rows = jobs.running()
        assert [r["key"] for r in rows] == ["lrn-recap:3"]
        assert rows[0]["label"] == "openvino"
        assert rows[0]["meta"] == {"session_id": 3}
        assert rows[0]["running_ms"] >= 0
    assert jobs.running() == []


def test_a_failed_job_does_not_stay_running_forever() -> None:
    """A crash used to be the difference between a stuck hint and a correct one."""
    with pytest.raises(RuntimeError), jobs.track("lrn-recap:3"):
        raise RuntimeError("model died")
    assert jobs.running() == []


def test_a_cancelled_request_releases_the_job() -> None:
    """The client walking away is exactly the reload case."""
    import asyncio

    async def scenario() -> None:
        started = asyncio.Event()

        async def work() -> None:
            with jobs.track("mtg-summary:7"):
                started.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(work())
        await started.wait()
        assert jobs.is_running("mtg-summary:7")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert jobs.running() == []

    asyncio.run(scenario())


def test_repeated_start_keeps_the_original_start_time() -> None:
    jobs.start("training")
    first = jobs.running()[0]["running_ms"]
    jobs.start("training")
    assert len(jobs.running()) == 1
    assert jobs.running()[0]["running_ms"] >= first


def test_jobs_endpoint_serves_the_registry() -> None:
    from fastapi.testclient import TestClient

    from deskmate.engine.api import create_app

    client = TestClient(create_app())
    assert client.get("/jobs").json() == {"data": []}

    with jobs.track("app-run:user-learning", label="user-learning"):
        body = client.get("/jobs").json()
    assert [r["key"] for r in body["data"]] == ["app-run:user-learning"]


def test_every_long_endpoint_registers_a_job() -> None:
    """A long endpoint that skips the registry is invisible after a reload."""
    source = API_PY.read_text(encoding="utf-8")
    for key in (
        'jobs.track(\n            f"lrn-recap:{session_id}"',
        'jobs.track(f"lrn-video:{session_id}"',
        'jobs.track(f"lrn-journey:{session_id}"',
        'jobs.track(\n            f"mtg-summary:{meeting_id}"',
        'jobs.track(f"mtg-video:{meeting_id}"',
        'jobs.track(f"app-run:{app_name}"',
        'jobs.track("training"',
    ):
        assert key in source, key


def test_client_and_server_agree_on_job_keys() -> None:
    """The client rebuilds a restored job's UI from its key, so the two key
    vocabularies have to match exactly."""
    api_src = API_PY.read_text(encoding="utf-8")
    js_src = APP_JS.read_text(encoding="utf-8")

    server_kinds = set(re.findall(r'jobs\.track\(\s*f?"([a-z-]+)[:"]', api_src))
    assert server_kinds == {
        "lrn-recap", "lrn-video", "lrn-journey",
        "mtg-summary", "mtg-video", "app-run", "training",
    }

    spec_block = js_src[js_src.index("function uiJobSpec"):js_src.index("async function syncUiJobs")]
    client_kinds = set(re.findall(r'case "([a-z-]+)"', spec_block))
    assert server_kinds <= client_kinds, server_kinds - client_kinds


def test_client_restores_and_polls_server_reported_jobs() -> None:
    js_src = APP_JS.read_text(encoding="utf-8")

    assert "function syncUiJobs" in js_src
    assert 'api("/jobs")' in js_src
    assert "function uiJobSpec" in js_src
    assert "function showRestoredJobResult" in js_src
    # Restored jobs have no pending fetch to resolve them, so they are polled.
    assert "setInterval(() => syncUiJobs(), 2500)" in js_src
    # And the poll stops once nothing is left, rather than running forever.
    assert "clearInterval(jobSync.timer)" in js_src
