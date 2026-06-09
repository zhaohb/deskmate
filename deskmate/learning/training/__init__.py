"""Training utilities: LoRA fine-tuning + DeskMate SFT data mining.

Built around DeskMate's local SQLite data model. The LoRA backend is Unsloth
(``FastLanguageModel``) with a ``transformers`` + ``peft`` fallback; all heavy ML
imports are guarded so this package imports cleanly without the optional
``[training]`` extra installed.
"""

from __future__ import annotations

from .data import DeskMateTrainingDataMiner
from .lora import (
    HAS_PEFT,
    HAS_TORCH,
    HAS_TRANSFORMERS,
    HAS_UNSLOTH,
    LoRATrainer,
    LoRATrainingConfig,
    missing_training_deps,
)

__all__ = [
    "DeskMateTrainingDataMiner",
    "HAS_PEFT",
    "HAS_TORCH",
    "HAS_TRANSFORMERS",
    "HAS_UNSLOTH",
    "LoRATrainer",
    "LoRATrainingConfig",
    "missing_training_deps",
]
