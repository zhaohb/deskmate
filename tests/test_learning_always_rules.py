"""Always-learning whitelist: the escape hatch for the detection heuristics.

Video sites are only *candidates* for a learning session — they must clear a
lecture-content score, or every entertainment video would become a study log.
That gate keys on school-flavoured wording (课程 / 讲义 / tutorial / lecture),
so a real technical talk can score 0 and be silently rejected. The whitelist
lets the user name sources they trust; these tests pin both halves of that deal:
a whitelisted source opens a session, and a non-whitelisted one still does not.

The motivating real case is `BV16FKy6kEVk` — "2026.2版 OpenVINO™ 的新功能" from
the OpenVINO 中文社区 channel: 23 minutes of genuine technical teaching whose
title contains no coursework vocabulary at all.
"""

from __future__ import annotations

from deskmate.apps.learning_slice import (
    classify_learning_signal,
    detect_problem_text,
    lecture_content_score,
    match_always_learning,
    normalize_always_rules,
)

# The real video under test.
BILI_URL = (
    "https://www.bilibili.com/video/BV16FKy6kEVk/"
    "?spm_id_from=333.337.search-card.all.click"
)
BILI_TITLE = "2026.2版 OpenVINO™ 的新功能_哔哩哔哩_bilibili"


# ── the gap the whitelist exists to close ────────────────────────────────────

def test_real_technical_talk_scores_zero_on_the_lecture_gate() -> None:
    """A genuine technical talk has no coursework wording — hence no score."""
    assert lecture_content_score(title=BILI_TITLE, pathq="/video/BV16FKy6kEVk/") == 0.0


def test_technical_talk_is_rejected_without_a_whitelist() -> None:
    kind, conf, _ = classify_learning_signal(
        app_name="chrome.exe", window_name=BILI_TITLE, browser_url=BILI_URL,
    )
    assert kind is None
    assert conf == 0.0


# ── whitelist admits it, by each supported match target ──────────────────────

def test_channel_name_from_on_screen_text_opens_a_session() -> None:
    """bilibili window titles omit the UP主, so the name is matched in OCR text."""
    kind, conf, reason = classify_learning_signal(
        app_name="chrome.exe",
        window_name=BILI_TITLE,
        browser_url=BILI_URL,
        text="OpenVINO中文社区  已关注  23:18  2026.2版 OpenVINO 的新功能",
        always_rules=["OpenVINO中文社区"],
    )
    assert kind == "courseware_view"
    assert conf >= 0.75          # must clear the default start_confidence
    assert "always-learning rule" in reason


def test_whitelist_matches_a_bare_domain() -> None:
    kind, _, reason = classify_learning_signal(
        app_name="chrome.exe",
        window_name="Install OpenVINO",
        browser_url="https://docs.openvino.ai/2026/get-started.html",
        always_rules=["docs.openvino.ai"],
    )
    assert kind == "courseware_view"
    assert "docs.openvino.ai" in reason


def test_whitelist_matches_a_creator_space_url() -> None:
    kind, _, _ = classify_learning_signal(
        app_name="chrome.exe",
        window_name="某个视频_哔哩哔哩_bilibili",
        browser_url="https://space.bilibili.com/123456/video",
        always_rules=["space.bilibili.com/123456"],
    )
    assert kind == "courseware_view"


def test_whitelisted_search_url_is_classified_as_material_query() -> None:
    """A whitelist hit still picks the more specific kind when there is a query."""
    kind, _, _ = classify_learning_signal(
        app_name="chrome.exe",
        window_name="搜索",
        browser_url="https://docs.openvino.ai/search.html?q=npu+dynamic+shape",
        always_rules=["docs.openvino.ai"],
    )
    assert kind == "material_query"


def test_whitelist_is_case_insensitive() -> None:
    kind, _, _ = classify_learning_signal(
        app_name="chrome.exe",
        window_name="Docs",
        browser_url="https://DOCS.OpenVINO.ai/2026/",
        always_rules=["docs.openvino.AI"],
    )
    assert kind == "courseware_view"


# ── the whitelist must not become a blanket "everything is learning" ─────────

def test_unrelated_page_is_untouched_by_an_active_whitelist() -> None:
    """A configured rule must not leak onto pages that do not match it."""
    kind, conf, _ = classify_learning_signal(
        app_name="chrome.exe",
        window_name="双十一狂欢购物节_淘宝",
        browser_url="https://www.taobao.com/list?cat=shoes",
        text="加入购物车 立即购买 领券",
        always_rules=["OpenVINO中文社区", "docs.openvino.ai"],
    )
    assert kind is None
    assert conf == 0.0


def test_entertainment_video_still_rejected_with_whitelist_configured() -> None:
    kind, _, _ = classify_learning_signal(
        app_name="chrome.exe",
        window_name="【鬼畜】经典名场面合集_哔哩哔哩_bilibili",
        browser_url="https://www.bilibili.com/video/BV1xx411c7XX/",
        always_rules=["docs.openvino.ai"],
    )
    assert kind is None


def test_on_screen_match_is_bounded_to_the_head_of_long_text() -> None:
    """A rule buried deep in an unrelated page's OCR dump must not trigger."""
    buried = ("无关内容 " * 4000) + "OpenVINO中文社区"
    assert match_always_learning(
        ["OpenVINO中文社区"], app_name="chrome.exe",
        window_name="某购物页", browser_url="https://example.com/x", text=buried,
    ) == ""


def test_error_on_screen_does_not_override_a_whitelisted_source() -> None:
    """An error is an event WITHIN the session, not the session's kind.

    This test previously asserted the opposite — that `problem` outranked the
    whitelist — because the error check ran first and returned a kind. That made
    one stray word ("失败" in a comment, a stack trace in release notes) relabel
    a whole sitting and outrank every real learning signal. The error is still
    detected, but through `detect_problem_text`, and recorded against whatever
    session is open.
    """
    text = "Traceback (most recent call last): RuntimeError: model compile failed"
    kind, _, reason = classify_learning_signal(
        app_name="chrome.exe",
        window_name="OpenVINO Docs",
        browser_url="https://docs.openvino.ai/2026/",
        text=text,
        always_rules=["docs.openvino.ai"],
    )
    assert kind == "courseware_view"
    assert "always-learning rule" in reason
    # …and the error is still visible to the caller, just on its own channel.
    line, _marker = detect_problem_text(text, "OpenVINO Docs")
    assert line


# ── normalization guards ─────────────────────────────────────────────────────

def test_short_rules_are_refused() -> None:
    """A 1-2 char rule would match nearly any page, so it is dropped."""
    assert normalize_always_rules(["a", "ab", "abc", "  "]) == ("abc",)


def test_rules_are_deduped_trimmed_and_lowercased() -> None:
    assert normalize_always_rules(
        ["  Docs.OpenVINO.ai ", "docs.openvino.ai", "OTHER"],
    ) == ("docs.openvino.ai", "other")


def test_empty_whitelist_changes_nothing() -> None:
    for rules in ([], (), None, ""):
        assert match_always_learning(
            rules or (), app_name="chrome.exe",
            window_name=BILI_TITLE, browser_url=BILI_URL,
        ) == ""
