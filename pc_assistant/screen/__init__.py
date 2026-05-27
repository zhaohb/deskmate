"""Screenshot + OCR."""

from .capture import Monitor, list_monitors, grab_monitor
from .ocr import OcrEngine, perform_ocr
from .snapshot import SnapshotWriter

__all__ = ["Monitor", "list_monitors", "grab_monitor", "OcrEngine", "perform_ocr", "SnapshotWriter"]
