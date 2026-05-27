"""Meeting detection and transcript grouping."""

from .detector import MeetingDetector, MeetingObservation, detect_meeting

__all__ = ["MeetingDetector", "MeetingObservation", "detect_meeting"]
