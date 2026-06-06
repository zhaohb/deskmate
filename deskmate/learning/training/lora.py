"""LoRATrainer — fine-tune local models via LoRA/QLoRA from DeskMate SFT pairs.

All ``torch``, ``transformers`` and ``peft`` imports are guarded so the module
can be imported without GPU dependencies. :class:`LoRATrainingConfig` works with
no optional deps; :class:`LoRATrainer` raises ``ImportError`` at construction
time when ``torch`` is unavailable (install with ``pip install
'deskmate[training]'``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...logger import get

logger = get("learning.training.lora")

# Optional imports -----------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    LoraConfig = None  # type: ignore[assignment,misc]
    TaskType = None  # type: ignore[assignment,misc]
    get_peft_model = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _select_device(hint: str | None = None) -> str:
    """Select the best available PyTorch device.

    Priority: explicit *hint* > cuda > mps > cpu.
    """
    if hint is not None:
        return hint
    if not HAS_TORCH or torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA / QLoRA fine-tuning."""

    # LoRA params
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training params
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048

    # QLoRA
    use_4bit: bool = False

    # Output
    output_dir: str = "checkpoints/lora"
    save_every_n_epochs: int = 1

    # Memory
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class LoRATrainer:
    """Fine-tune a local causal LM with LoRA (or QLoRA) adapters.

    Parameters
    ----------
    config:
        LoRA training configuration.
    model_name:
        HuggingFace model identifier or local path.
    device:
        PyTorch device string. ``None`` auto-detects (cuda > mps > cpu).

    Raises
    ------
    ImportError
        If ``torch`` is not installed.
    """

    def __init__(
        self,
        config: LoRATrainingConfig,
        *,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str | None = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for LoRATrainer. "
                "Install with: pip install 'deskmate[training]'"
            )

        self.config = config
        self.model_name = model_name
        self.device = _select_device(device)
        self.tokenizer: Any = None
        self.model: Any = None

    # -- Public API ----------------------------------------------------------

    def prepare_dataset(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert SFT pairs to tokenized examples with prompt-masked labels.

        Each returned dict contains ``input_ids``, ``attention_mask``,
        ``labels`` and ``text``. ``labels`` mask the prompt tokens with -100 so
        the loss is computed ONLY over the assistant's response — the standard
        SFT objective. Sequences are NOT padded here; padding happens per-batch
        in the collator so we don't waste compute padding every row to
        ``max_seq_length``.

        Parameters
        ----------
        pairs:
            List of dicts with at least ``input`` and ``output`` keys, as
            produced by :class:`DeskMateTrainingDataMiner.extract_sft_pairs`.
        """
        self._ensure_tokenizer()

        dataset: list[dict[str, Any]] = []
        max_len = self.config.max_seq_length
        for pair in pairs:
            prompt_text, full_text = self._format_pair_with_prompt(pair)
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[:max_len]
            n_prompt = min(len(prompt_ids), len(full_ids))

            # Mask the prompt so loss is computed only on the response tokens.
            labels = list(full_ids)
            for i in range(n_prompt):
                labels[i] = -100
            # A pair whose response was entirely truncated away teaches nothing.
            if all(t == -100 for t in labels):
                continue

            dataset.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "text": full_text,
                }
            )

        return dataset

    def train(self, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Run LoRA fine-tuning on the given SFT pairs.

        Parameters
        ----------
        pairs:
            List of dicts with at least ``input`` and ``output`` keys.

        Returns
        -------
        dict
            Training summary with keys: ``status``, ``epochs``, ``total_steps``,
            ``avg_loss``, ``adapter_path``, ``training_samples``.
        """
        if not pairs:
            return {"status": "skipped", "reason": "no training data"}

        dataset = self.prepare_dataset(pairs)
        self._load_model()
        self._apply_lora()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = 0
        cumulative_loss = 0.0
        num_loss_steps = 0

        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = self._train_epoch(dataset, optimizer)
            steps_in_epoch = max(
                1, (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
            )
            total_steps += steps_in_epoch
            cumulative_loss += epoch_loss * steps_in_epoch
            num_loss_steps += steps_in_epoch

            logger.info(
                "Epoch %d/%d  loss=%.4f",
                epoch + 1,
                self.config.num_epochs,
                epoch_loss,
            )

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_adapter(epoch + 1)

        avg_loss = cumulative_loss / num_loss_steps if num_loss_steps else 0.0
        adapter_path = str(Path(self.config.output_dir) / "final")
        self._save_adapter_to(adapter_path)

        return {
            "status": "completed",
            "epochs": self.config.num_epochs,
            "total_steps": total_steps,
            "avg_loss": avg_loss,
            "adapter_path": adapter_path,
            "training_samples": len(pairs),
        }

    # -- Internal helpers ----------------------------------------------------

    def _ensure_tokenizer(self) -> None:
        """Lazily load the tokenizer."""
        if self.tokenizer is not None:
            return

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for LoRATrainer. "
                "Install with: pip install 'deskmate[training]'"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self) -> None:
        """Load the base model for fine-tuning."""
        if self.model is not None:
            return

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for LoRATrainer. "
                "Install with: pip install 'deskmate[training]'"
            )

        self._ensure_tokenizer()

        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}

        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed; falling back to bf16 (QLoRA disabled)"
                )

        if self.device in ("cuda", "auto"):
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _apply_lora(self) -> None:
        """Wrap the loaded model with LoRA adapters via peft."""
        if not HAS_PEFT:
            raise ImportError(
                "peft is required for LoRA training. "
                "Install with: pip install 'deskmate[training]'"
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info(
            "LoRA applied: rank=%d, alpha=%d, targets=%s",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.target_modules,
        )

    def _format_pair(self, pair: dict[str, Any]) -> str:
        """Format an SFT pair as a chat-style training string (full text)."""
        return self._format_pair_with_prompt(pair)[1]

    def _format_pair_with_prompt(self, pair: dict[str, Any]) -> tuple[str, str]:
        """Return ``(prompt_text, full_text)`` for an SFT pair.

        ``prompt_text`` is everything up to and including the assistant
        generation prefix (no response); ``full_text`` appends the response.
        The two share an identical prefix so token-length subtraction gives the
        exact boundary used to mask the prompt in the loss.
        """
        user_input = pair.get("input", "")
        assistant_output = pair.get("output", "")

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                user_msg = [{"role": "user", "content": user_input}]
                prompt_text = self.tokenizer.apply_chat_template(
                    user_msg, tokenize=False, add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(
                    user_msg + [{"role": "assistant", "content": assistant_output}],
                    tokenize=False, add_generation_prompt=False,
                )
                # Guard: full must extend prompt; otherwise fall through.
                if full_text.startswith(prompt_text):
                    return prompt_text, full_text
            except Exception as exc:  # noqa: BLE001
                logger.debug("Auto chat template failed, using manual format: %s", exc)

        eos = ""
        if self.tokenizer is not None:
            eos = getattr(self.tokenizer, "eos_token", "") or ""
        prompt_text = f"<|user|>\n{user_input}\n<|assistant|>\n"
        full_text = f"{prompt_text}{assistant_output}{eos}"
        return prompt_text, full_text

    def _train_epoch(
        self,
        dataset: list[dict[str, Any]],
        optimizer: Any,
    ) -> float:
        """Train one epoch over the dataset. Returns average loss."""
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(dataset), self.config.batch_size):
            batch_items = dataset[i : i + self.config.batch_size]
            loss = self._train_step(batch_items, optimizer)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches if num_batches else 0.0

    def _collate(self, batch_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Right-pad a variable-length micro-batch to its own longest sequence.

        Pads input_ids with the tokenizer pad id, attention_mask with 0, and
        labels with -100 (ignored by the loss). Far cheaper than padding every
        row to max_seq_length up front."""
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        max_len = max(item["input_ids"].size(0) for item in batch_items)

        ids, masks, labels = [], [], []
        for item in batch_items:
            seq = item["input_ids"]
            pad = max_len - seq.size(0)
            ids.append(torch.cat([seq, torch.full((pad,), pad_id, dtype=torch.long)]))
            masks.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(ids).to(self.device),
            "attention_mask": torch.stack(masks).to(self.device),
            "labels": torch.stack(labels).to(self.device),
        }

    def _train_step(
        self,
        batch_items: list[dict[str, Any]],
        optimizer: Any,
    ) -> float:
        """Execute a single training step on a micro-batch."""
        batch = self._collate(batch_items)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],  # prompt tokens are -100 → loss on response only
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        optimizer.step()

        return loss.item()

    def _save_adapter(self, epoch: int) -> None:
        """Save adapter checkpoint for the given epoch."""
        path = str(Path(self.config.output_dir) / f"epoch_{epoch}")
        self._save_adapter_to(path)

    def _save_adapter_to(self, path: str) -> None:
        """Save the LoRA adapter (and tokenizer) to *path*."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

        logger.info("Adapter saved to %s", path)


__all__ = [
    "HAS_TORCH",
    "LoRATrainer",
    "LoRATrainingConfig",
]
