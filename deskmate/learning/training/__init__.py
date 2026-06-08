"""Training utilities: LoRA fine-tuning + DeskMate SFT data mining.

Built around DeskMate's local SQLite data model. All heavy ML imports
(``torch``, ``transformers``, ``peft``) are guarded so this package imports
cleanly without the optional ``[training]`` extra installed; the trainer only
raises at construction time when ``torch`` is missing.
"""

from __future__ import annotations

from .data import DeskMateTrainingDataMiner
from .lora import (
    HAS_PEFT,
    HAS_TORCH,
    HAS_TRANSFORMERS,
    LoRATrainer,
    LoRATrainingConfig,
    missing_training_deps,
)

__all__ = [
    "DeskMateTrainingDataMiner",
    "HAS_PEFT",
    "HAS_TORCH",
    "HAS_TRANSFORMERS",
    "LoRATrainer",
    "LoRATrainingConfig",
    "missing_training_deps",
]
