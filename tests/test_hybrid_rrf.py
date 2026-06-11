"""Recency/confidence-weighted RRF fusion in hybrid_search (5.3)."""

from __future__ import annotations

from deskmate.db.search_engine import SearchEngine, SearchResult, SearchResultKind


def _ocr(frame_id: int, ts: str) -> SearchResult:
    return SearchResult(
        kind=SearchResultKind.OCR,
        timestamp=ts,
        payload={"frame_id": frame_id, "text": f"frame {frame_id}", "timestamp": ts},
    )


def _engine() -> SearchEngine:
    # hybrid_search only touches self.search / self.semantic_search, which we
    # monkeypatch — so a bare instance with no real connection is fine.
    return SearchEngine(conn=None, lock=None)


def test_recency_decay_breaks_rank_ties(monkeypatch) -> None:
    """Two equal-rank hits: the more recent one must win after decay."""
    eng = _engine()
    old = _ocr(1, "2026-06-01T09:00:00+08:00")
    new = _ocr(2, "2026-06-10T09:00:00+08:00")
    # Same FTS rank order [old, new]; no semantic leg signal beyond presence.
    monkeypatch.setattr(eng, "search", lambda *a, **k: [old, new])
    monkeypatch.setattr(eng, "semantic_search", lambda *a, **k: [(old, 0.5), (new, 0.5)])
    out = eng.hybrid_search("q", "ocr", model_name="m", limit=2)
    assert [r.payload["frame_id"] for r in out] == [2, 1]


def test_semantic_confidence_scales_contribution(monkeypatch) -> None:
    """A high-similarity semantic hit outranks a low-similarity one at same age."""
    eng = _engine()
    ts = "2026-06-10T09:00:00+08:00"
    a = _ocr(1, ts)
    b = _ocr(2, ts)
    # FTS ranks them equally adjacent; semantic gives b far higher confidence.
    monkeypatch.setattr(eng, "search", lambda *a_, **k: [a, b])
    monkeypatch.setattr(eng, "semantic_search", lambda *a_, **k: [(a, 0.05), (b, 0.95)])
    out = eng.hybrid_search("q", "ocr", model_name="m", limit=2)
    assert out[0].payload["frame_id"] == 2


def test_falls_back_to_fts_without_semantic(monkeypatch) -> None:
    eng = _engine()
    a = _ocr(1, "2026-06-10T09:00:00+08:00")
    monkeypatch.setattr(eng, "search", lambda *a_, **k: [a])
    monkeypatch.setattr(eng, "semantic_search", lambda *a_, **k: [])
    out = eng.hybrid_search("q", "ocr", model_name="m", limit=5)
    assert [r.payload["frame_id"] for r in out] == [1]


def test_no_recency_decay_when_halflife_zero(monkeypatch) -> None:
    """halflife<=0 disables decay → pure rank/confidence RRF (old behavior)."""
    eng = _engine()
    old = _ocr(1, "2020-01-01T00:00:00+08:00")
    new = _ocr(2, "2026-06-10T09:00:00+08:00")
    monkeypatch.setattr(eng, "search", lambda *a, **k: [old, new])
    monkeypatch.setattr(eng, "semantic_search", lambda *a, **k: [(old, 0.5), (new, 0.5)])
    out = eng.hybrid_search("q", "ocr", model_name="m", limit=2, recency_halflife_hours=0)
    # Without decay, FTS rank order (old first) is preserved.
    assert out[0].payload["frame_id"] == 1
