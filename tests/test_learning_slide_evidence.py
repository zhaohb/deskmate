"""Slide harvest, ASR canonicalization, and concept-preserving audio budgets.

Fixtures cover two unrelated lectures so the helpers cannot secretly depend on
one subject's vocabulary. The first is a browser capture with the title buried
after window chrome; the second is a meeting-player capture of a biology slide.
"""

from __future__ import annotations

from deskmate.apps.learning_evidence import (
    budget_paragraphs,
    canonicalize_against_slides,
    harvest_slide_evidence,
)

# Browser player: chrome + URLs first, slide title thousands of characters in.
_BROWSER_OCR = """
2026.2版 OpenVINO™ 的新功能_哔哩哔哩_bilibili - Google Chrome
Minimize
Maximize
Restore
Close
Back
Forward
Reload
Home
Bookmarks
https://example.com/video/abc
首页
番剧
直播
收藏
历史
投稿
""" + ("推荐视频占位 " * 80) + """
OpenVINO intel DEVCON Workshop Series 2026 LoRA Adaptors in Video Generation Pipeline CAKEIFY
pythonlora_text2video.py ./ltx_video_ov/FP32
ltx_pixel_pytorch_lora_weights.safetensors 1.0
弹幕礼仪
清晰度
倍速
"""

# Meeting player: different chrome, different subject, same failure shape.
_MEETING_OCR = """
Weekly seminar - Zoom Meeting
Mute
Start Video
Participants
Share
Chat
Leave
""" + ("waiting for host\n" * 8) + """
Mitosis vs Meiosis in Cell Division
prophase_checkpoints.md
"""


def test_harvest_recovers_title_buried_in_browser_chrome() -> None:
    slides = harvest_slide_evidence([_BROWSER_OCR])
    blob = " ".join(slides.prompt_lines())
    assert "LoRA Adaptors in Video Generation Pipeline" in blob
    assert "lora_text2video.py" in blob.lower()
    assert "lora_weights.safetensors" in blob.lower()
    assert "Minimize" not in blob
    assert "https://example.com" not in blob


def test_harvest_recovers_title_buried_in_meeting_chrome() -> None:
    slides = harvest_slide_evidence([_MEETING_OCR])
    blob = " ".join(slides.prompt_lines())
    assert "Mitosis vs Meiosis in Cell Division" in blob
    assert "prophase_checkpoints.md" in blob
    assert "Mute" not in blob
    assert "waiting for host" not in blob.lower()


def test_harvest_keeps_title_when_a_filename_shares_the_line() -> None:
    """Real captures glue the slide title and a script path into one OCR dump."""
    blob = (
        "Ask Google OpenVINO intel. DEVCON Workshop Series 2026 "
        "LoRA Adaptors in Video Generation Pipeline CAKEIFY "
        "python lora_text2video.py ./ltx_video_ov/FP32"
    )
    slides = harvest_slide_evidence([blob])
    joined = " ".join(slides.prompt_lines())
    assert "LoRA" in joined
    assert "lora_text2video.py" in joined.lower()
    assert "LoRA" in canonicalize_against_slides(["Laura Adapter"], slides)[0]


def test_harvest_prompt_lines_put_titles_before_body() -> None:
    slides = harvest_slide_evidence([_BROWSER_OCR])
    lines = slides.prompt_lines()
    titles = [i for i, ln in enumerate(lines) if ln.startswith("课件标题:")]
    body = [i for i, ln in enumerate(lines) if ln.startswith("课件OCR:")]
    assert titles, "slide titles must be labeled so the model can cite them"
    if body:
        assert max(titles) < min(body)


def test_an_activity_feed_is_not_courseware() -> None:
    """Capture tools list "app.exe · timestamp" rows; the recap read them as slides.

    DeskMate's own timeline was on screen beside the lecture, so its rows became
    "课件标题" and their identifiers became mandatory recap topics.
    """
    blobs = [
        "转录 三事件 chrome.exe·8/25/2026,2:59:04PM DeskMate -\n"
        "学习 chrome.exe·8/25/2026,2:58:01PM 待办 Details bilibili-\n"
        "Gradient Descent Converges Linearly\n",
        "采集 训练 chrome.exe·8/25/2026,2:57:29 PM FRAME 模型服务 bilibili-\n"
        "Gradient Descent Converges Linearly\n",
    ]
    slides = harvest_slide_evidence(blobs)
    titles = [ln for ln in slides.prompt_lines() if ln.startswith("课件标题:")]
    blob = " ".join(titles)
    assert "Gradient Descent Converges Linearly" in blob
    assert "chrome.exe" not in blob
    assert "模型服务" not in blob


def test_repeated_widgets_drop_but_a_stable_slide_title_stays() -> None:
    """Chrome is high-DF *and* widget-shaped. A title on every frame is not."""
    widgets = "Mute\nShare\nChat\nLeave\n"
    blobs = [
        widgets + "Binary Search Trees\n",
        widgets + "Binary Search Trees\ninorder walk\n",
        widgets + "Binary Search Trees\n",
        widgets + "AVL rotations on disk\n",
        widgets + "Binary Search Trees\n",
    ]
    slides = harvest_slide_evidence(blobs)
    blob = " ".join(slides.prompt_lines())
    assert "Binary Search Trees" in blob
    assert "Mute" not in blob
    assert "Share" not in blob


def test_canonicalize_rewrites_asr_toward_ocr_spellings() -> None:
    video = harvest_slide_evidence([_BROWSER_OCR])
    fixed_video = canonicalize_against_slides(
        ["15:04:21: 還支持了Laura Adapter 也就是Laura適配器的一個插入"],
        video,
    )
    assert "LoRA" in fixed_video[0]
    assert "Laura" not in fixed_video[0]

    bio = harvest_slide_evidence([_MEETING_OCR])
    fixed_bio = canonicalize_against_slides(
        ["the two paths are mitosis versus meosis in animals"],
        bio,
    )
    assert "Mitosis versus Meiosis" in fixed_bio[0]
    assert "meosis" not in fixed_bio[0]


def test_budget_keeps_a_rare_aside_when_even_sampling_would_skip_it() -> None:
    paragraphs = [(f"{i:02d}: filler about hardware platforms and deployment", 1) for i in range(40)]
    paragraphs[17] = ("17: aside on Merkle trees in the log structure", 1)
    kept = budget_paragraphs(
        paragraphs, max_lines=8, max_chars=800, anchor_terms=["Merkle"],
    )
    texts = [p[0] for p in kept]
    assert any("Merkle" in t for t in texts)
    assert texts[0].startswith("00:")
    assert texts[-1].startswith("39:")


def test_budget_does_not_pin_the_lecture_theme() -> None:
    """A name on every line is the theme; spanning it is enough."""
    paragraphs = [
        (f"{i:02d}: ThemeWord GPU Pipeline 部署和硬件加速讲解", 1) for i in range(40)
    ]
    paragraphs[17] = ("17: aside on Merkle trees in the log structure", 1)
    kept = budget_paragraphs(
        paragraphs,
        max_lines=8,
        max_chars=800,
        anchor_terms=["ThemeWord", "GPU", "Pipeline", "Merkle"],
    )
    texts = [p[0] for p in kept]
    assert len(kept) <= 8
    assert any("Merkle" in t for t in texts)
    assert texts[0].startswith("00:")
    assert texts[-1].startswith("39:")


def test_collect_ocr_from_db_keeps_buried_slide_title(tmp_path, monkeypatch) -> None:
    from deskmate.apps.agent import _collect_courseware_ocr_lines
    from deskmate.db.manager import DatabaseManager

    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db = DatabaseManager(tmp_path / "data.db")
    ts = "2026-08-25T15:04:23+08:00"
    cur = db._conn.execute(  # noqa: SLF001
        "INSERT INTO frames (offset_index, timestamp, app_name) VALUES (0, ?, ?)",
        (ts, "chrome.exe"),
    )
    fid = int(cur.lastrowid)
    db._conn.execute(  # noqa: SLF001
        "INSERT INTO ocr_text (frame_id, text, text_length) VALUES (?, ?, ?)",
        (fid, _BROWSER_OCR, len(_BROWSER_OCR)),
    )
    lines = _collect_courseware_ocr_lines(ts, ts, [{"kind": "study_other", "apps": ["chrome.exe"]}])
    blob = "\n".join(lines)
    assert "LoRA Adaptors in Video Generation Pipeline" in blob
    assert "Minimize" not in blob


def test_collect_audio_rewrites_near_miss_and_keeps_the_aside(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timedelta

    from deskmate.apps.agent import _collect_learning_audio_bits
    from deskmate.apps.learning_evidence import harvest_slide_evidence
    from deskmate.db.manager import DatabaseManager

    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db = DatabaseManager(tmp_path / "data.db")
    start = datetime(2026, 8, 25, 15, 0, 0).astimezone()
    for i in range(80):
        ts = (start + timedelta(seconds=i * 4)).replace(microsecond=0).isoformat()
        text = "還支持了Laura Adapter 也就是适配器插入" if i == 40 else f"第{i}段讲解部署和硬件加速。"
        db._conn.execute(  # noqa: SLF001
            """INSERT INTO audio_transcriptions (timestamp, transcription, device, text_length)
               VALUES (?,?,?,?)""",
            (ts, text, "loopback", len(text)),
        )
    slides = harvest_slide_evidence([_BROWSER_OCR])
    lo = start.replace(microsecond=0).isoformat()
    hi = (start + timedelta(seconds=79 * 4)).replace(microsecond=0).isoformat()
    bits, stats = _collect_learning_audio_bits(lo, hi, {}, slides=slides)
    joined = "\n".join(bits)
    assert "LoRA" in joined
    assert "Laura" not in joined
    assert stats["included"] <= stats["total"]


def test_enrichment_prompt_requires_extracted_concepts_in_the_recap() -> None:
    from deskmate.learning_memory.extract import ConceptHit, ExtractionResult
    from deskmate.learning_memory.pipeline import format_enrichment_prompt

    result = ExtractionResult(concepts=[], lecture_items=[], topic_summary="")
    block = format_enrichment_prompt(
        result,
        due_reviews=[],
        topics=[],
        edges=[],
        events=[],
        concepts=[ConceptHit(name="Merkle tree", topic="storage")],
    )
    assert "MUST COVER" in block
    assert "Merkle tree" in block
    assert "shorter substitute graph" in block
