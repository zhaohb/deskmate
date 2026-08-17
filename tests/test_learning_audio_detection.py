"""Audio-driven study detection: hearing a class while doing something else.

Screen text and speech used to be concatenated into one blob before reaching
the classifier, so a lecture's transcript could only ever tint the verdict for
whatever window was in front. Listening to a class while working was therefore
filed under the foreground app — in the case that prompted this, an OpenVINO
talk playing behind an editor was recorded as `code_edit`, and the concepts
extracted from it were fragments of the editor's own UI.

Covers the six changes that fixed it:

1. audio arrives as its own channel, never merged into on-screen text
2. `lecture_audio_score` scores speech on its own terms
3. classification precedence puts lecture audio above foreground activity
4. `problem` is an event inside a session, not a kind of session
5. a session's kind may be promoted but never demoted
6. concepts come from whichever source carried the signal

Thresholds are calibrated against a real 35-minute recorded talk (630 rows,
9327 chars: 我们 ×116, 大家 ×112, technical terms 4.2/100 chars).
"""

from __future__ import annotations

import pytest

from deskmate.apps.learning_slice import (
    classify_learning_signal,
    detect_problem_text,
    lecture_audio_score,
)
from deskmate.learning_memory.detector import (
    KIND_RANK,
    LearningSessionDetector,
    _better_kind,
)

IDE_APP = "Code.exe"
IDE_TITLE = "hongbo - Visual Studio Code"

# Shape mirrors the real transcript: audience address, structural moves,
# domain nouns, sustained length.
LECTURE = (
    "OpenVINO都支持在这些不同硬件平台上面 可以进行方便的快速的模型部署 "
    "另外 不管大家平时习惯使用的操作系统是 Windows Linux还是macOS "
    "那说到OpenVINO的最新的这个发布的版本 针对现在大家对生成式AI模型持续 "
    "首先我们来看这次新版本在推理性能上的优化 大家可以看到 内存占用明显下降 "
    "接下来我们再看第二个部分 也就是说 模型量化之后的精度损失 "
    "比如我们常用的这个生成的Pipeline 我们也增加了对应的API支持 "
    "总结一下 这次版本主要在推理加速和显存优化两方面有改进 "
)


def _classify(audio: str = "", *, text: str = "def foo(): pass"):
    """Classify with an IDE in the foreground — the contested case."""
    return classify_learning_signal(
        app_name=IDE_APP, window_name=IDE_TITLE, browser_url="",
        text=text, audio_text=audio,
    )


# ── 2. lecture_audio_score ───────────────────────────────────────────────────

def test_real_lecture_speech_scores_high() -> None:
    assert lecture_audio_score(LECTURE) >= 0.55


def test_silence_scores_zero() -> None:
    assert lecture_audio_score("") == 0.0


def test_brief_speech_is_not_a_lecture() -> None:
    """A passing remark must not open a study session."""
    assert lecture_audio_score("好的 我知道了 等下发你") == 0.0


def test_casual_conversation_scores_low() -> None:
    chatter = "嗯 好的 那个 我等下发给你 行 就这样 拜拜 你到了跟我说一声 好嘞 那先这样 " * 8
    assert lecture_audio_score(chatter) < 0.55


def test_sung_music_scores_low() -> None:
    """Music with vocals: dense transcript, but no audience and no structure."""
    singing = "啦啦啦 夜色温柔 星光洒落 心事轻轻飘过 时间慢慢走 " * 12
    assert lecture_audio_score(singing) < 0.55


def test_technical_shoptalk_is_not_a_lecture() -> None:
    """The hard negative: dense domain vocabulary, but nobody is teaching.

    Technical terms alone must never carry a verdict — a colleague debugging
    with you sounds exactly like this.
    """
    shoptalk = "这个 bug 我看一下 你把 log 发我 嗯 是这个 model 的 API 变了 你改一下参数 好 " * 8
    assert lecture_audio_score(shoptalk) < 0.55


def test_formal_written_register_also_counts() -> None:
    """A scripted or read-aloud talk uses connectives the observed speaker didn't."""
    formal = (
        "本节课我们讨论神经网络的梯度传播 首先给出定义 反向传播定义为 "
        "换句话说 我们需要计算损失函数对每个参数的偏导数 "
        "举例来说 对于一个三层的模型 具体来说 由此可见 综上所述 "
        "值得注意的是 该算法的复杂度与张量维度相关 因此在实际训练中需要优化 "
    ) * 2
    assert lecture_audio_score(formal) >= 0.55


def test_english_lecture_counts() -> None:
    english = (
        "So first of all let's take a look at how the model handles inference. "
        "As you can see everyone the runtime latency drops significantly here. "
        "In other words we are trading a little accuracy for throughput. "
        "For example our quantized tensor pipeline runs on the GPU kernel directly. "
        "To summarize we covered the algorithm and the compiler optimizations. "
    ) * 2
    assert lecture_audio_score(english) >= 0.55


def test_longer_lookback_needs_more_speech() -> None:
    """The same text over a wider window is weaker evidence of continuity."""
    tight = lecture_audio_score(LECTURE, lookback_seconds=90.0)
    wide = lecture_audio_score(LECTURE, lookback_seconds=600.0)
    assert wide <= tight


# ── 1 + 3. audio channel and precedence ──────────────────────────────────────

def test_lecture_audio_beats_foreground_ide() -> None:
    """The headline case: a class playing behind an editor is a class."""
    kind, conf, reason = _classify(LECTURE)
    assert kind == "courseware_view"
    assert conf >= 0.75
    assert "lecture audio" in reason


def test_ide_alone_is_still_code_edit() -> None:
    """No lecture playing → unchanged behaviour."""
    kind, _, _ = _classify("")
    assert kind == "code_edit"


def test_audio_does_not_leak_into_on_screen_scoring() -> None:
    """Audio passed as `text` (the old merge) must not be how this works."""
    as_audio = classify_learning_signal(
        app_name=IDE_APP, window_name=IDE_TITLE, text="", audio_text=LECTURE,
    )
    assert as_audio[0] == "courseware_view"


def test_foreground_courseware_outranks_lecture_audio() -> None:
    """Looking at material is stronger evidence than hearing it.

    `pytorch.org` is a recognised courseware host, so the URL branch must settle
    this before the audio branch is ever consulted.
    """
    kind, _, reason = classify_learning_signal(
        app_name="chrome.exe", window_name="Docs",
        browser_url="https://pytorch.org/docs/stable/index.html",
        audio_text=LECTURE,
    )
    assert kind == "courseware_view"
    assert "lecture audio" not in reason
    assert "host" in reason


def test_unrecognised_docs_host_falls_through_to_audio() -> None:
    """A docs site not on the host list is no better than an unrelated window.

    Guards the precedence itself: the audio branch is the fallback, so it must
    still fire when the foreground offers nothing the classifier recognises.
    """
    kind, _, reason = classify_learning_signal(
        app_name="chrome.exe", window_name="Some Vendor Docs",
        browser_url="https://docs.some-vendor.example/guide",
        audio_text=LECTURE,
    )
    assert kind == "courseware_view"
    assert "lecture audio" in reason


def test_whitelist_outranks_everything() -> None:
    kind, _, reason = classify_learning_signal(
        app_name=IDE_APP, window_name=IDE_TITLE,
        browser_url="https://example.edu/talk", audio_text=LECTURE,
        always_rules=["example.edu"],
    )
    assert kind == "courseware_view"
    assert "always-learning rule" in reason


# ── 4. problem is an event, not a kind ───────────────────────────────────────

def test_problem_is_never_a_session_kind() -> None:
    """One stray error word used to relabel an entire session."""
    kind, _, _ = classify_learning_signal(
        app_name="chrome.exe", window_name="Some page",
        browser_url="https://example.com/",
        text="部署失败 undefined reference to symbol",
    )
    assert kind != "problem"


def test_error_text_is_still_detected_separately() -> None:
    assert detect_problem_text("", "Traceback (most recent call last)")
    assert detect_problem_text("build failed", "")
    assert detect_problem_text("", "一切正常") == ""


def test_error_on_screen_does_not_disturb_the_lecture_verdict() -> None:
    """A page mentioning an error while a class plays is still a class."""
    kind, _, reason = _classify(LECTURE, text="编译失败 error: cannot open file")
    assert kind == "courseware_view"
    assert "lecture audio" in reason


# ── 5. kind may be promoted, never demoted ───────────────────────────────────

def test_kind_ladder_is_ordered_strongest_first() -> None:
    assert KIND_RANK["courseware_view"] > KIND_RANK["material_query"]
    assert KIND_RANK["material_query"] > KIND_RANK["code_edit"]
    assert KIND_RANK["code_edit"] > KIND_RANK["study_other"]


@pytest.mark.parametrize(
    ("current", "candidate", "expected"),
    [
        ("code_edit", "courseware_view", "courseware_view"),   # promote
        ("courseware_view", "code_edit", "courseware_view"),   # never demote
        ("study_other", "material_query", "material_query"),
        (None, "code_edit", "code_edit"),
        ("code_edit", None, "code_edit"),                       # no signal, keep
        ("code_edit", "code_edit", "code_edit"),
    ],
)
def test_kind_promotion(current, candidate, expected) -> None:
    assert _better_kind(current, candidate) == expected


def test_unknown_kind_never_displaces_a_known_one() -> None:
    assert _better_kind("courseware_view", "something_else") == "courseware_view"


# ── 6. concepts come from whichever source carried the signal ────────────────

def test_audio_driven_session_ignores_the_foreground_title() -> None:
    """The bug this fixes: a lecture session described by the editor's own UI.

    Mining the foreground title during an audio-driven session produced concepts
    like "EXPLORER" and stray phrases from whatever was on screen, which then
    entered the knowledge graph and the review queue as if they were taught.
    """
    _, from_audio = LearningSessionDetector._tag(
        IDE_TITLE, "", LECTURE, kind="courseware_view", title_is_relevant=False,
    )
    joined = " ".join(from_audio)
    assert "Visual Studio Code" not in joined
    assert "hongbo" not in joined


def test_screen_driven_session_still_mines_the_title() -> None:
    """Unchanged for on-screen sessions, where the title IS the subject."""
    _, concepts = LearningSessionDetector._tag(
        "深度学习 第3讲 反向传播", "", "梯度下降与链式法则",
        kind="courseware_view", title_is_relevant=True,
    )
    assert concepts, "a descriptive title should still yield concepts"
