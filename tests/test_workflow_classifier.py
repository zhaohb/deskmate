"""Workflow classifier: app-priority matching + browser negative override."""

from __future__ import annotations

from deskmate.workflow.classifier import classify_frame


def test_app_name_match_wins() -> None:
    assert classify_frame("Cursor.exe", "main.py — project") == "coding"
    assert classify_frame("OUTLOOK.EXE", "Inbox") == "email"
    assert classify_frame("Teams.exe", "Chat") == "communication"


def test_browser_title_does_not_misclassify_as_coding() -> None:
    # A YouTube tutorial whose title contains "Code" must stay browsing, not
    # be reclassified as coding off the title substring.
    assert classify_frame("chrome.exe", "How to Code in Python - YouTube") == "browsing"


def test_browser_title_does_not_misclassify_as_meeting() -> None:
    # A Slack web tab whose title contains "meet" must stay browsing.
    assert classify_frame("msedge.exe", "let's meet later (#general) | Slack") == "browsing"


def test_browser_app_is_browsing_even_with_coding_title() -> None:
    assert classify_frame("firefox.exe", "Visual Studio Code docs") == "browsing"


def test_title_fallback_when_app_unknown() -> None:
    # No app-name signal → fall back to title (non-browser app).
    assert classify_frame("unknown.exe", "Zoom Meeting") == "communication"


def test_unmatched_is_other() -> None:
    assert classify_frame("randomapp.exe", "some window") == "other"
