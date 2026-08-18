"""Visual-change probe: the diff maths and the baseline state machine.

A fullscreen video keeps app+title constant, so idle capture deduped it away and
playback never reached the timeline. These lock in the two properties that fix
guarantees: a first frame only seeds a baseline (no spurious capture), and a
real pixel change past the threshold fires while a still screen stays quiet.
"""

from __future__ import annotations

from deskmate.capture.visual_change import VisualChangeProbe, diff_ratio


def test_diff_ratio_is_zero_for_identical_thumbnails() -> None:
    assert diff_ratio(bytes([10] * 256), bytes([10] * 256)) == 0.0


def test_diff_ratio_is_one_for_full_swing() -> None:
    assert diff_ratio(bytes([0] * 8), bytes([255] * 8)) == 1.0


def test_diff_ratio_guards_empty_and_mismatched_lengths() -> None:
    assert diff_ratio(b"", b"abc") == 0.0
    assert diff_ratio(bytes([1, 2, 3]), bytes([1, 2])) == 0.0


def test_first_signature_only_seeds_baseline() -> None:
    probe = VisualChangeProbe(threshold=0.06)
    # No previous frame yet — entering a screen must not fire a capture.
    assert probe.evaluate(bytes([100] * 16)) is False


def test_change_past_threshold_fires_and_stillness_stays_quiet() -> None:
    probe = VisualChangeProbe(threshold=0.06)
    probe.evaluate(bytes([100] * 16))            # baseline
    assert probe.evaluate(bytes([100] * 16)) is False   # unchanged → quiet
    assert probe.evaluate(bytes([200] * 16)) is True    # big change → fire
    assert probe.evaluate(bytes([200] * 16)) is False   # settled again


def test_change_below_threshold_does_not_fire() -> None:
    probe = VisualChangeProbe(threshold=0.5)
    probe.evaluate(bytes([100] * 16))
    # ~0.06 mean change, well under a 0.5 threshold.
    assert probe.evaluate(bytes([115] * 16)) is False


def test_visual_ready_respects_the_live_enable_toggle() -> None:
    """The UI switch flips cfg.capture.capture_on_visual_change at runtime."""
    from deskmate.capture.event_driven_capture import EventDrivenCapture
    from deskmate.config import Config

    cfg = Config()
    cfg.capture.visual_change_probe_ms = 0  # never rate-limited by interval
    state = EventDrivenCapture(cfg)

    cfg.capture.capture_on_visual_change = True
    assert state.visual_ready(1_000_000.0) is True

    cfg.capture.capture_on_visual_change = False
    assert state.visual_ready(2_000_000.0) is False

