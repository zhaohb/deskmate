"""MUST COVER names are patched into the recap when the writer omits them."""

from __future__ import annotations

from deskmate.learning_memory.extract import ConceptHit
from deskmate.learning_memory.pipeline import collect_must_cover, ensure_must_cover
from deskmate.learning_memory.topics_llm import TopicHit

_RECIP = """## 是否在学习
是

## 讲了什么
- **生成式AI API**：视频生成 Pipeline

## 课程重点
- KV 缓存压缩

## 知识图谱
节点：生成式AI API、KV缓存

## 掌握状态
- 待确认：全场
"""


def test_ensure_must_cover_inserts_omitted_short_topic() -> None:
    out = ensure_must_cover(
        _RECIP,
        [{"name": "Laura Adapter", "evidence": "還支持了Laura Adapter", "source": "extracted"}],
    )
    spoke = out.split("## 讲了什么", 1)[1].split("## ", 1)[0]
    graph = out.split("## 知识图谱", 1)[1].split("## ", 1)[0]
    assert "Laura Adapter" in spoke
    assert "Laura Adapter" in graph


def test_ensure_must_cover_skips_section_that_already_names_it() -> None:
    md = _RECIP.replace("视频生成 Pipeline", "视频生成 Pipeline 与 Laura Adapter")
    out = ensure_must_cover(
        md,
        [{"name": "Laura Adapter", "evidence": "還支持了Laura Adapter", "source": "extracted"}],
    )
    spoke = out.split("## 讲了什么", 1)[1].split("\n## ", 1)[0]
    assert spoke.count("Laura Adapter") == 1


def test_takeaways_section_is_never_machine_patched() -> None:
    """课程重点 states why a point matters — a patcher cannot judge that.

    Padding it also duplicated 讲了什么 verbatim, which the prompt forbids.
    """
    out = ensure_must_cover(
        _RECIP,
        [
            {"name": "Laura Adapter", "evidence": "還支持了Laura Adapter", "source": "extracted"},
            {"name": "DevCon", "evidence": "大家好 欢迎来到 DevCon", "source": "transcript"},
        ],
    )
    focus = out.split("## 课程重点", 1)[1].split("\n## ", 1)[0]
    assert focus.strip() == "- KV 缓存压缩"


def test_transcript_scraped_names_are_listed_not_explained() -> None:
    """A raw ASR line is not an explanation; pretending otherwise is the bug.

    Only evidence the extractor wrote as a summary earns a full bullet. Names
    whose only backing is a scraped transcript line get one honest mention.
    """
    out = ensure_must_cover(
        _RECIP,
        [
            {
                "name": "Int4 KV Cache压缩",
                "evidence": "推出Int4 KV Cache压缩技术，体积压缩68%",
                "source": "extracted",
            },
            {
                "name": "LoRA",
                "evidence": "那在这次的2026.2的新版本里面 我们除了可以直接利用 video0 Generation",
                "source": "transcript",
            },
            {
                "name": "GitHub",
                "evidence": "对人物的作用 然后我们要继续 Inference model 运行催礼的这样一个效果",
                "source": "transcript",
            },
        ],
    )
    spoke = out.split("## 讲了什么", 1)[1].split("\n## ", 1)[0]
    assert "推出Int4 KV Cache压缩技术，体积压缩68%" in spoke
    assert "运行催礼" not in spoke
    assert "video0 Generation" not in spoke
    # The two ungrounded names share a single trailing line, in name order.
    mentions = [ln for ln in spoke.splitlines() if "本场还提到" in ln]
    assert len(mentions) == 1
    assert "LoRA" in mentions[0]
    assert "GitHub" in mentions[0]


def test_graph_node_patches_collapse_into_one_line() -> None:
    """Nine consecutive "节点补全：X（相关）" lines were noise, not a graph."""
    out = ensure_must_cover(
        _RECIP,
        [
            {"name": "DevCon", "evidence": "", "source": "transcript"},
            {"name": "GitHub", "evidence": "", "source": "transcript"},
            {"name": "LoRA", "evidence": "", "source": "transcript"},
        ],
    )
    graph = out.split("## 知识图谱", 1)[1].split("\n## ", 1)[0]
    patched = [ln for ln in graph.splitlines() if "节点补全" in ln]
    assert len(patched) == 1
    assert all(name in patched[0] for name in ("DevCon", "GitHub", "LoRA"))


def test_patched_bullets_never_leak_writer_instructions_or_placeholders() -> None:
    """These read as filler, and "短主题，须保留" is a rule aimed at the model."""
    out = ensure_must_cover(
        _RECIP,
        [
            {"name": "LoRA", "evidence": "還支持了Laura Adapter 也就是适配器的插入"},
            {"name": "DevCon", "evidence": ""},
        ],
    )
    assert "短主题，须保留" not in out
    assert "见本场录音/课件" not in out
    assert "（录音/课件）" not in out


def test_context_free_name_is_mentioned_but_never_given_a_fake_definition() -> None:
    out = ensure_must_cover(_RECIP, [{"name": "DevCon", "evidence": ""}])
    spoke = out.split("## 讲了什么", 1)[1].split("\n## ", 1)[0]
    graph = out.split("## 知识图谱", 1)[1].split("\n## ", 1)[0]
    assert "- **DevCon**" not in spoke
    assert "DevCon" in spoke
    assert "DevCon" in graph


def test_collect_must_cover_grounds_each_name_in_a_spoken_sentence() -> None:
    covers = collect_must_cover(
        topics=[TopicHit(name="OpenVINO 2026.2 新特性")],
        concepts=[],
        due_reviews=[],
        audio_texts=[
            "00:12 [spk]: OpenVINO 2026.2的新版本主要是在生成式AI API里面提供了很多新特性的更新",
            "00:20 [spk]: 我们还支持了Laura Adapter 也就是适配器的插入",
        ],
        ocr_texts=["课件标题: LoRA Adaptors in Video Generation Pipeline"],
    )
    by_name = {c["name"]: c["evidence"] for c in covers}
    # A multi-word topic is never said verbatim, so the line is matched on tokens.
    assert "新版本" in by_name["OpenVINO 2026.2 新特性"]
    assert "适配器" in by_name["LoRA"]
    assert all("[spk]" not in ev for ev in by_name.values())


def test_stored_json_evidence_is_unwrapped_into_a_sentence() -> None:
    covers = collect_must_cover(
        topics=[],
        concepts=[],
        due_reviews=[{"name": "OpenVINO", "evidence_json": '["OpenVINO是英特尔的推理加速工具套件"]'}],
        audio_texts=["00:01 [spk]: OpenVINO是英特尔的推理加速工具套件"],
        ocr_texts=[],
    )
    ev = covers[0]["evidence"]
    assert ev == "OpenVINO是英特尔的推理加速工具套件"


_ENUMERATION = (
    "那OpenVINO是由英特尔推出的一款针对AI模型优化的工具套件 "
    "OpenVINO目前可以支持市面上所有主流的深度学习框架训练出来的模型 "
    "像大家现在用的最多的PyTorch模型格式 以及也常用的像ONNX、PaddlePaddle这样的模型格式"
)
_EXPLANATION = (
    "那在这次的2026.2的新版本里面 我们除了可以直接利用视频生成管线之外 "
    "還支持了LoRA Adapter 也就是LoRA適配器的一個插入"
)


def test_names_only_ever_listed_alongside_others_are_not_taught() -> None:
    """A slide's logo wall and an architecture diagram are enumerations.

    "PyTorch / ONNX / PaddlePaddle" is the set of formats OpenVINO reads, not
    three subjects this lecture taught. Every mention sitting in a line that
    rattles off several names at once is the signal.
    """
    covers = collect_must_cover(
        topics=[],
        concepts=[],
        due_reviews=[],
        audio_texts=[_ENUMERATION, _EXPLANATION],
        ocr_texts=[
            "课件标题: PyTorch TensorFlow Keras TensorFlowLite PaddlePaddle",
            "课件标题: LoRA Adaptors in Video Generation Pipeline",
        ],
    )
    names = [c["name"] for c in covers]
    assert "LoRA" in names
    assert "PyTorch" not in names
    assert "PaddlePaddle" not in names
    assert "TensorFlowLite" not in names


def test_one_sentence_does_not_explain_two_different_names() -> None:
    """PyTorch and PaddlePaddle shipped the same line, so the recap said it twice."""
    covers = collect_must_cover(
        topics=[TopicHit(name="LLMPipeline"), TopicHit(name="VLMPipeline")],
        concepts=[],
        due_reviews=[],
        audio_texts=["我们有LLMPipeline来支持文本 也有VLMPipeline来支持多模态"],
        ocr_texts=[],
    )
    evidence = [c["evidence"] for c in covers if c["evidence"]]
    assert len(evidence) == len(set(evidence))


def test_a_name_contained_in_a_longer_one_is_not_covered_twice() -> None:
    """"OpenVINO 2026.2" and "OpenVINO 2026.2 新特性" are one subject."""
    covers = collect_must_cover(
        topics=[TopicHit(name="OpenVINO 2026.2 新特性"), TopicHit(name="OpenVINO 2026.2")],
        concepts=[],
        due_reviews=[],
        audio_texts=["OpenVINO 2026.2的新版本提供了很多新特性的更新"],
        ocr_texts=[],
    )
    assert [c["name"] for c in covers] == ["OpenVINO 2026.2 新特性"]


def test_extracted_summaries_and_scraped_lines_are_told_apart() -> None:
    covers = collect_must_cover(
        topics=[TopicHit(name="Int4 KV Cache压缩", evidence=["推出Int4 KV Cache压缩技术，体积压缩68%"])],
        concepts=[],
        due_reviews=[],
        audio_texts=["推出Int4 KV Cache压缩技术，体积压缩68%", _EXPLANATION],
        ocr_texts=["课件标题: LoRA Adaptors in Video Generation Pipeline"],
    )
    by_name = {c["name"]: c for c in covers}
    assert by_name["Int4 KV Cache压缩"]["source"] == "extracted"
    assert by_name["LoRA"]["source"] == "transcript"


def test_a_name_never_spoken_aloud_was_not_taught() -> None:
    """OCR also sees the browser and the capture tool; the lecturer never says those.

    Chrome and DeskMate are on screen all session, so the extractor hands back
    "bilibili" and "DeskMate" as concepts complete with evidence — which used
    to make them arrive as fully-formed explanations.
    """
    covers = collect_must_cover(
        topics=[],
        concepts=[
            ConceptHit(name="bilibili", topic="ui", evidence=["课件标题: X bilibili.com/video/BV16"]),
            ConceptHit(name="DeskMate", topic="ui", evidence=["课件标题: DeskMate 2026."]),
            ConceptHit(name="KV Cache压缩", topic="ov", evidence=["推出KV Cache压缩技术，体积压缩68%"]),
        ],
        due_reviews=[],
        audio_texts=["我们推出了KV Cache压缩技术 体积压缩68%"],
        ocr_texts=["课件标题: X bilibili.com/video/BV16", "课件标题: DeskMate 2026."],
    )
    assert [c["name"] for c in covers] == ["KV Cache压缩"]


def test_slides_alone_still_yield_names_when_nothing_was_recorded() -> None:
    """Silent study is normal; "never spoken" cannot mean "nothing qualifies"."""
    covers = collect_must_cover(
        topics=[],
        concepts=[ConceptHit(name="KV Cache压缩", topic="ov", evidence=["体积压缩68%"])],
        due_reviews=[],
        audio_texts=[],
        ocr_texts=["课件标题: KV Cache压缩"],
    )
    assert [c["name"] for c in covers] == ["KV Cache压缩"]


def test_short_names_count_as_terms_only_when_they_are_acronyms() -> None:
    """"exe" and "com" match inside "executable"/"compression" and slipped through."""
    covers = collect_must_cover(
        topics=[],
        concepts=[
            ConceptHit(name="exe", topic="ui", evidence=["chrome.exe"]),
            ConceptHit(name="com", topic="ui", evidence=["bilibili.com"]),
            ConceptHit(name="NPU", topic="hw", evidence=["还有我们AIPC独有的NPU"]),
        ],
        due_reviews=[],
        audio_texts=["the compression is executable on 我们AIPC独有的NPU"],
        ocr_texts=[],
    )
    assert [c["name"] for c in covers] == ["NPU"]


def test_words_read_off_the_screen_never_count_as_an_explanation() -> None:
    """"intel" is a logo in every slide header, and the extractor called it a concept.

    The lecturer does say "Intel", so it passes the spoken check — but its only
    evidence is the header, which turned into the bullet
    "intel：OpenVINO intel. DEVCON Workshop Series 2026".
    """
    covers = collect_must_cover(
        topics=[],
        concepts=[
            ConceptHit(
                name="intel",
                topic="ui",
                evidence=["课件标题: OpenVINO intel. DEVCON Workshop Series 2026"],
            ),
        ],
        due_reviews=[],
        audio_texts=["比如Intel的各款的CPU 还有ARM架构的CPU 都可以部署"],
        ocr_texts=["课件标题: OpenVINO intel. DEVCON Workshop Series 2026"],
    )
    assert [c["source"] for c in covers] == ["transcript"]


def test_overlapping_asr_windows_do_not_restate_themselves_in_a_bullet() -> None:
    """Consecutive ASR windows share audio, so one line repeats its own opening."""
    doubled = (
        "我们除了可以直接利用 Video Generation Pipeline去生成视频 "
        "我们除了可以直接利用 Video Generation Pipeline去生成视频之外 还支持了适配器"
    )
    covers = collect_must_cover(
        topics=[TopicHit(name="Video Generation Pipeline", evidence=[doubled])],
        concepts=[],
        due_reviews=[],
        audio_texts=[doubled],
        ocr_texts=[],
    )
    assert covers[0]["evidence"].count("去生成视频") == 1


def test_a_misheard_name_is_respelled_from_the_slides() -> None:
    """The review queue stores what ASR heard; the slide has the real spelling.

    "Laura Adapter" and "LoRA" both reached the recap, so one subject was
    explained under the misheard name and listed again under the right one.
    """
    covers = collect_must_cover(
        topics=[],
        concepts=[],
        due_reviews=[
            {
                "name": "Laura Adapter",
                "evidence_json": '["支持了Laura Adapter 也就是适配器的插入"]',
            }
        ],
        audio_texts=["還支持了Laura Adapter 也就是适配器的插入"],
        ocr_texts=["课件标题: LoRA Adaptors in Video Generation Pipeline"],
    )
    assert [c["name"] for c in covers] == ["LoRA Adapter"]


def test_collect_must_cover_keeps_ocr_name_heard_in_audio() -> None:
    covers = collect_must_cover(
        topics=[TopicHit(name="OpenVINO 2026.2新特性")],
        concepts=[ConceptHit(name="Video Generation Pipeline", topic="ov")],
        due_reviews=[{"name": "Laura Adapter", "evidence_json": "還支持了Laura Adapter"}],
        audio_texts=[
            "今天讲OpenVINO 2026.2新特性 这个新版本更新了很多东西",
            "先看 Video Generation Pipeline 怎么把文字变成视频",
            "還支持了Laura Adapter 也就是适配器插入",
        ],
        ocr_texts=["课件标题: LoRA Adaptors in Video Generation Pipeline"],
    )
    names = [c["name"] for c in covers]
    # The slide spelling wins and absorbs the bare "LoRA" the OCR also yielded.
    assert "LoRA Adapter" in names
    assert "Laura Adapter" not in names
    assert "OpenVINO 2026.2新特性" in names
