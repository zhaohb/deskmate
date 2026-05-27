"""OCR engines for Windows-native OCR and Tesseract.

`winrt`     → uses Windows.Media.Ocr via `winrt` Python bindings.
`tesseract` → uses pytesseract (must have Tesseract installed on PATH).
`off`       → no-op, returns empty text.

Output shape: `(text, words_json, confidence)` where words_json is JSON of
`[{text, left, top, width, height, conf, ...}, ...]`.

**All bounding-box and conf values are stringified** (this is non-obvious;
keeping strings preserves compatibility with consumers that expect JSON text
fields instead of floats.
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from functools import lru_cache
from typing import Any

from PIL import Image

from ..logger import get

logger = get("screen.ocr")


class OcrEngine(str, Enum):
    OFF = "off"
    WINRT = "winrt"
    TESSERACT = "tesseract"


@lru_cache(maxsize=1)
def _winrt_available() -> bool:
    try:
        # WinRT OCR async calls depend on these namespaces together.
        import winrt.windows.foundation  # noqa: F401, PLC0415
        import winrt.windows.foundation.collections  # noqa: F401, PLC0415
        import winrt.windows.globalization  # noqa: F401, PLC0415
        import winrt.windows.graphics.imaging  # noqa: F401, PLC0415
        import winrt.windows.media.ocr  # noqa: F401, PLC0415
        import winrt.windows.storage.streams  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def perform_ocr(
    image: Image.Image,
    engine: OcrEngine,
    languages: list[str] | None = None,
    *,
    tesseract_cmd: str | None = None,
) -> tuple[str, str, float | None]:
    if engine == OcrEngine.OFF:
        return "", "[]", None
    if image.width == 0 or image.height == 0:
        return "", "[]", None
    if engine == OcrEngine.WINRT and _winrt_available():
        return _winrt(image, languages or ["en-US"])
    if engine == OcrEngine.TESSERACT:
        return _tesseract(image, _tesseract_languages(languages), tesseract_cmd=tesseract_cmd)
    if engine == OcrEngine.WINRT:
        logger.info("winrt OCR unavailable, falling back to tesseract")
        return _tesseract(image, _tesseract_languages(languages), tesseract_cmd=tesseract_cmd)
    return "", "[]", None


# ─── tesseract ─────────────────────────────────────────────────────────────
_TESSERACT_LANGUAGE_ALIASES = {
    "en": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    "zh": "chi_sim",
    "zh-cn": "chi_sim",
    "zh-hans": "chi_sim",
    "zh-tw": "chi_tra",
    "zh-hant": "chi_tra",
    "ja": "jpn",
    "ja-jp": "jpn",
    "ko": "kor",
    "ko-kr": "kor",
    "fr": "fra",
    "fr-fr": "fra",
    "de": "deu",
    "de-de": "deu",
    "es": "spa",
    "es-es": "spa",
    "ru": "rus",
    "ru-ru": "rus",
}


def _tesseract_languages(languages: list[str] | None) -> list[str]:
    """Convert BCP-47/WinRT language names into Tesseract language codes."""
    if not languages:
        return ["eng"]
    mapped: list[str] = []
    for lang in languages:
        raw = (lang or "").strip().lower()
        if not raw:
            continue
        if raw in {"eng", "chi_sim", "chi_tra", "jpn", "kor", "fra", "deu", "spa", "rus"}:
            mapped.append(raw)
            continue
        key = raw.replace("_", "-")
        mapped.append(_TESSERACT_LANGUAGE_ALIASES.get(key, key.split("-")[0]))
    return mapped or ["eng"]


def _winrt_languages(languages: list[str] | None) -> list[str]:
    """Normalize config language names into WinRT language tags."""
    if not languages:
        return ["en-US"]
    mapped: list[str] = []
    aliases = {
        "en": "en-US",
        "eng": "en-US",
        "zh": "zh-Hans",
        "zh-cn": "zh-Hans",
        "zh-hans": "zh-Hans",
        "chi_sim": "zh-Hans",
        "chi-sim": "zh-Hans",
        "zh-tw": "zh-Hant",
        "zh-hant": "zh-Hant",
        "chi_tra": "zh-Hant",
        "chi-tra": "zh-Hant",
        "ja": "ja-JP",
        "jpn": "ja-JP",
        "ko": "ko-KR",
        "kor": "ko-KR",
    }
    for lang in languages:
        raw = (lang or "").strip()
        if not raw:
            continue
        mapped.append(aliases.get(raw.lower().replace("_", "-"), raw))
    return mapped or ["en-US"]


def _tesseract(
    image: Image.Image,
    languages: list[str],
    *,
    tesseract_cmd: str | None = None,
) -> tuple[str, str, float | None]:
    try:
        import pytesseract  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("pytesseract not installed: %s", exc)
        return "", "[]", None
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    lang = "+".join(languages)
    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("tesseract failed: %s", exc)
        return "", "[]", None

    w = float(image.width) if image.width > 0 else 1.0
    h = float(image.height) if image.height > 0 else 1.0

    words: list[dict[str, str]] = []
    parts: list[str] = []
    raw_confs: list[float] = []

    levels = data.get("level", [])
    page_nums = data.get("page_num", [])
    block_nums = data.get("block_num", [])
    par_nums = data.get("par_num", [])
    line_nums = data.get("line_num", [])
    word_nums = data.get("word_num", [])

    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        wn = _int_at_seq(word_nums, i, default=0)
        if not txt or wn <= 0:
            continue
        parts.append(txt)
        conf_values = data.get("conf", [])
        conf_raw_str = str(conf_values[i]) if i < len(conf_values) else "-1"
        try:
            conf_raw = float(conf_raw_str)
        except ValueError:
            conf_raw = -1.0
        if conf_raw >= 0:
            raw_confs.append(conf_raw)
        words.append({
            "text": txt,
            "conf": f"{conf_raw:.2f}",
            "left":   str(_float_at(data, "left", i) / w),
            "top":    str(_float_at(data, "top", i) / h),
            "width":  str(_float_at(data, "width", i) / w),
            "height": str(_float_at(data, "height", i) / h),
            "level":     str(_int_at_seq(levels, i)),
            "page_num":  str(_int_at_seq(page_nums, i)),
            "block_num": str(_int_at_seq(block_nums, i)),
            "par_num":   str(_int_at_seq(par_nums, i)),
            "line_num":  str(_int_at_seq(line_nums, i)),
            "word_num":  str(wn),
        })

    text = " ".join(parts)
    confidence = (sum(raw_confs) / len(raw_confs)) if raw_confs else 0.0
    return text, json.dumps(words, ensure_ascii=False), float(confidence)


def _float_at(data: dict[str, Any], key: str, index: int, default: float = 0.0) -> float:
    values = data.get(key, [])
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return default


def _int_at_seq(values: Any, index: int, default: int = 0) -> int:
    try:
        return int(values[index])
    except (IndexError, TypeError, ValueError):
        return default


# ─── WinRT Windows.Media.Ocr ───────────────────────────────────────────────
def _winrt(image: Image.Image, languages: list[str]) -> tuple[str, str, float | None]:
    try:
        return asyncio.run(_winrt_async(image, languages))
    except RuntimeError:
        import threading  # noqa: PLC0415

        out: dict[str, Any] = {}

        def _runner() -> None:
            out["res"] = asyncio.new_event_loop().run_until_complete(_winrt_async(image, languages))

        t = threading.Thread(target=_runner)
        t.start(); t.join()
        return out.get("res", ("", "[]", None))


async def _winrt_async(image: Image.Image, languages: list[str]) -> tuple[str, str, float | None]:
    try:
        # Import foundation explicitly so incomplete winrt installs fail early
        # and we cleanly degrade instead of emitting runtime tracebacks.
        import winrt.windows.foundation  # noqa: F401, PLC0415
        import winrt.windows.foundation.collections  # noqa: F401, PLC0415
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine as WinOcrEngine
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
    except Exception as exc:  # noqa: BLE001
        logger.warning("winrt imports failed: %s", exc)
        return "", "[]", None

    buf = image.convert("RGB")
    from io import BytesIO  # noqa: PLC0415
    bio = BytesIO()
    buf.save(bio, format="PNG")
    raw = bio.getvalue()

    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream.get_output_stream_at(0))
    writer.write_bytes(raw)
    await writer.store_async()
    await writer.flush_async()
    stream.seek(0)

    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()

    engine = None
    for tag in _winrt_languages(languages):
        try:
            engine = WinOcrEngine.try_create_from_language(Language(tag))
        except Exception as exc:  # noqa: BLE001
            logger.debug("winrt language %r unavailable: %s", tag, exc)
            continue
        if engine is not None:
            break
    if engine is None:
        engine = WinOcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        return "", "[]", None
    result = await engine.recognize_async(bitmap)

    w = float(image.width) if image.width > 0 else 1.0
    h = float(image.height) if image.height > 0 else 1.0

    words: list[dict[str, str]] = []
    parts: list[str] = []
    for line in result.lines:
        for word in line.words:
            t = word.text
            if not t:
                continue
            parts.append(t)
            r = word.bounding_rect
            words.append({
                "text": t,
                "left":   str(float(r.x)      / w),
                "top":    str(float(r.y)      / h),
                "width":  str(float(r.width)  / w),
                "height": str(float(r.height) / h),
                "conf":   "1.0",     # Windows OCR has no word-level confidence.
            })

    text = " ".join(parts) or (result.text or "")
    return text, json.dumps(words, ensure_ascii=False), (1.0 if text else None)
