"""Workflow classifier.

The default classifier is a simple local heuristic that buckets frames by app:

- IDE / terminal / editor      → "coding"
- browser                      → "browsing"
- email client                 → "email"
- video call / messenger       → "communication"
- everything else              → "other"

If you want to plug in a real classifier, set `WORKFLOW_CLASSIFIER` env
var to a HTTP endpoint that accepts `{"app": "...", "title": "..."}` and
returns `{"workflow": "..."}`.
"""

from .classifier import WorkflowClassifier, classify_frame

__all__ = ["WorkflowClassifier", "classify_frame"]
