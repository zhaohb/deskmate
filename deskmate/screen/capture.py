"""Multi-monitor screenshot capture via `mss`."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from ..logger import get

logger = get("screen.capture")


@dataclass(frozen=True)
class Monitor:
    id: int
    left: int
    top: int
    width: int
    height: int

    @property
    def name(self) -> str:
        return f"monitor-{self.id}"


def list_monitors() -> list[Monitor]:
    try:
        import mss  # noqa: PLC0415
    except ImportError:
        logger.warning("mss not installed; cannot enumerate monitors")
        return []
    with mss.mss() as sct:
        mons = sct.monitors
    # index 0 is the "all monitors" virtual rect — skip it.
    return [
        Monitor(id=i, left=m["left"], top=m["top"], width=m["width"], height=m["height"])
        for i, m in enumerate(mons[1:], start=1)
    ]


def grab_monitor(monitor: Monitor, *, max_width: int = 0) -> Image.Image | None:
    try:
        import mss  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("mss missing: %s", exc)
        return None
    with mss.mss() as sct:
        raw = sct.grab({"left": monitor.left, "top": monitor.top, "width": monitor.width, "height": monitor.height})
        img = Image.frombytes("RGB", raw.size, raw.rgb)
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
    return img


def encode_jpeg(img: Image.Image, quality: int = 80) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
