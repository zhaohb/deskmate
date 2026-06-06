"""Learning subsystem (additive).

Houses opt-in, self-contained machine-learning utilities that read from the
existing DeskMate database without modifying any capture/producer code. The
first member is :mod:`deskmate.learning.training`, a faithful port of
OpenJarvis's LoRA fine-tuning pipeline adapted to DeskMate's own data sources.
"""

from __future__ import annotations

__all__: list[str] = []
