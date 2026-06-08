"""Tests for editor-title → document_path extraction."""

from __future__ import annotations

import pytest

from deskmate.a11y.document import extract_document_path, is_editor_app


@pytest.mark.parametrize(
    "app,title,expected",
    [
        ("Code.exe", "config.toml - hongbo - Visual Studio Code", "config.toml"),
        ("Code.exe", "● config.toml - hongbo - Visual Studio Code", "config.toml"),
        (
            "Code.exe",
            "04_doorbell_roundtrip.sh - dong [SSH: 10.239.136.255] - Visual Studio Code",
            "04_doorbell_roundtrip.sh",
        ),
        ("notepad.exe", "*note.txt - 记事本", "note.txt"),
        ("pycharm64.exe", "main.py – myproj", "main.py"),
        ("Cursor.exe", "ask.py - deskmate - Cursor", "ask.py"),
    ],
)
def test_extracts_filename_for_editors(app, title, expected) -> None:
    assert extract_document_path(app, title) == expected


@pytest.mark.parametrize(
    "app,title",
    [
        ("Code.exe", "Welcome - Visual Studio Code"),  # not a file
        ("chrome.exe", "收件箱 (383) - me@gmail.com - Gmail - Google Chrome"),  # browser
        ("explorer.exe", "Downloads"),  # file manager, no doc
        ("Code.exe", ""),  # empty title
        ("", "config.toml - x - Visual Studio Code"),  # unknown app
    ],
)
def test_returns_none_when_no_clear_document(app, title) -> None:
    assert extract_document_path(app, title) is None


def test_is_editor_app() -> None:
    assert is_editor_app("Code.exe")
    assert is_editor_app("Cursor.exe")
    assert is_editor_app("pycharm64.exe")
    assert not is_editor_app("chrome.exe")
    assert not is_editor_app("explorer.exe")
