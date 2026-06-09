"""LoRATrainer — fine-tune local models via LoRA/QLoRA from DeskMate SFT pairs.

The LoRA stack is **Unsloth** (``FastLanguageModel``): faster, lower-VRAM
training that supports NVIDIA (CUDA), Intel (XPU — Arc / Core-Ultra iGPU) and
AMD, including 4-bit QLoRA. When Unsloth isn't installed the trainer transparently
falls back to a plain ``transformers`` + ``peft`` path, so the feature degrades
rather than breaks. All heavy imports are guarded so this module imports cleanly
without the ``[training]`` extra.
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

# Unsloth is the primary backend. Importing it patches transformers/peft for
# speed + low VRAM and exposes FastLanguageModel for one-call model+LoRA setup.
try:
    from unsloth import FastLanguageModel

    HAS_UNSLOTH = True
except Exception:  # noqa: BLE001 — broad: unsloth import can fail on env/HW probe
    HAS_UNSLOTH = False
    FastLanguageModel = None  # type: ignore[assignment,misc]

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


def missing_training_deps() -> list[str]:
    """Names of the packages that are NOT importable for training.

    Empty list ⇒ a usable LoRA stack is present. ``torch`` is always required.
    On top of that we need EITHER Unsloth (preferred) OR the
    ``transformers`` + ``peft`` fallback — so we only report something missing
    when neither complete stack is available."""
    missing: list[str] = []
    if not HAS_TORCH:
        missing.append("torch")
    # A complete stack = unsloth, OR (transformers AND peft).
    if not HAS_UNSLOTH and not (HAS_TRANSFORMERS and HAS_PEFT):
        if not HAS_TRANSFORMERS or not HAS_PEFT:
            missing.append("unsloth")  # the one-line install that covers it all
    return missing


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _select_device(hint: str | None = None) -> str:
    """Select the best available PyTorch device.

    Priority: explicit *hint* > cuda > xpu (Intel GPU) > mps > cpu.

    ``xpu`` is Intel's GPU backend, built into official PyTorch since 2.5 — an
    Intel Arc / Core-Ultra iGPU shows up here when an XPU-enabled torch wheel is
    installed (the default CPU wheel reports ``xpu.is_available() == False``).
    """
    if hint is not None:
        return hint
    if not HAS_TORCH or torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_compiler(names: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Return the first available ``(cxx_path, cc_path)`` from *names*.

    Searches PATH, then the active conda/Python env's ``Library/bin`` (where
    conda-installed Intel ``icx`` lives), then a few well-known Windows dirs."""
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import sys  # noqa: PLC0415

    # Windows conda layout puts python.exe in the env ROOT (…/envs/<name>);
    # POSIX puts it in …/<name>/bin. Handle both.
    exe_dir = os.path.dirname(sys.executable)
    env_root = exe_dir if os.name == "nt" else os.path.dirname(exe_dir)
    search_dirs: list[str] = []
    # System Intel oneAPI install(s) FIRST — newest version first. Not
    # version-pinned: we glob whatever "compiler\<ver>\bin" the Base Toolkit
    # dropped. Preferred over the conda `dpcpp_impl` icx, whose SYCL headers are
    # incomplete (missing ur_api.h) and can't actually build XPU kernels.
    for base in (
        os.environ.get("DESKMATE_ONEAPI"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     "Intel", "oneAPI"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                     "Intel", "oneAPI"),
    ):
        if base and os.path.isdir(os.path.join(base, "compiler")):
            for d in sorted(glob.glob(os.path.join(base, "compiler", "*", "bin")),
                            reverse=True):
                search_dirs.append(d)
    search_dirs += [
        os.path.join(env_root, "Library", "bin"),
        os.path.join(env_root, "bin"),
        r"C:\TDM-GCC-64\bin", r"C:\mingw64\bin", r"C:\msys64\mingw64\bin",
    ]
    for cxx, cc in names:
        # search_dirs FIRST (system oneAPI before the conda dpcpp_impl build,
        # whose SYCL headers are incomplete); only then fall back to PATH.
        cxx_path = None
        for d in search_dirs:
            cand = os.path.join(d, cxx + ".exe")
            if os.path.isfile(cand):
                cxx_path = cand
                break
        if not cxx_path:
            cxx_path = shutil.which(cxx)
        if not cxx_path:
            continue
        cc_path = os.path.join(os.path.dirname(cxx_path), cc + ".exe")
        if not os.path.isfile(cc_path):
            cc_path = shutil.which(cc) or cxx_path
        return cxx_path, cc_path
    return None


def _find_ze_path() -> str | None:
    """Locate a Level-Zero SDK root (a dir with ``include/level_zero/ze_api.h``
    and ``lib/ze_loader.lib``). Triton-XPU reads it via ``ZE_PATH``.

    Searches an existing ``ZE_PATH``/``LEVEL_ZERO_V1_SDK_PATH``, the active
    conda env's ``Library`` (where we install the SDK headers), and any Intel
    oneAPI install. Returns the root, or ``None`` if the headers aren't found —
    the Base Toolkit ships ``ze_loader.dll`` (runtime) but NOT the SDK headers,
    so they're installed separately (see docs)."""
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415

    def _ok(root: str) -> bool:
        # A clean ZE root has the L0 headers but NOT a `sycl/` subdir — Triton
        # prepends `<ZE_PATH>/include` to the compile, so a `sycl/` there (e.g.
        # the conda dpcpp_impl's incomplete headers) would shadow the system
        # oneAPI SYCL headers and break the build with `__spirv_*` errors.
        if not (root and os.path.isfile(
                os.path.join(root, "include", "level_zero", "ze_api.h"))):
            return False
        return not os.path.isdir(os.path.join(root, "include", "sycl"))

    from ... import paths as _paths  # noqa: PLC0415

    candidates = [
        os.environ.get("LEVEL_ZERO_V1_SDK_PATH"),
        os.environ.get("ZE_PATH"),
        # DeskMate's dedicated, uncontaminated SDK dir (setup-intel-xpu.bat
        # installs here); preferred precisely because it has no `sycl/`.
        str(_paths.root() / "level-zero-sdk"),
    ]
    for base in (
        os.environ.get("DESKMATE_ONEAPI"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     "Intel", "oneAPI"),
    ):
        if base:
            candidates += sorted(glob.glob(os.path.join(base, "*", )), reverse=True)
    for root in candidates:
        if _ok(root or ""):
            return root
    return None


def _find_vcvars() -> str | None:
    """Locate ``vcvarsall.bat`` (MSVC) via ``vswhere`` or well-known paths.

    Override with ``DESKMATE_VCVARS``. Returns the bat path, or ``None``."""
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    cand = os.environ.get("DESKMATE_VCVARS")
    if cand and os.path.isfile(cand):
        return cand

    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            out = subprocess.run(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            if out:
                bat = os.path.join(out, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                if os.path.isfile(bat):
                    return bat
        except Exception:  # noqa: BLE001 - vswhere is best-effort
            pass

    # Fallback: glob common install roots (Build Tools / Community / …).
    for pf in (pf86, os.environ.get("ProgramFiles", r"C:\Program Files")):
        for bat in sorted(glob.glob(os.path.join(
                pf, "Microsoft Visual Studio", "*", "*",
                "VC", "Auxiliary", "Build", "vcvarsall.bat")), reverse=True):
            if os.path.isfile(bat):
                return bat
    return None


def _ensure_msvc_env() -> bool:
    """Make sure the MSVC build environment (``INCLUDE``/``LIB``/``PATH``) is in
    this process — Intel's ``icpx`` is a clang-cl driver and links against the
    MSVC standard library, so without it Triton's kernel build dies with
    ``'climits' file not found``.

    When DeskMate's UI is started via ``start-deskmate-train.bat`` these are
    already set (we no-op). But for a plain ``deskmate ui`` launch they're
    absent, so we run ``vcvarsall.bat x64`` in a subshell, diff the environment,
    and import the new ``INCLUDE``/``LIB``/``LIBPATH``/``PATH`` entries into
    ``os.environ`` for this process. Returns True if INCLUDE/LIB are present
    afterwards.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    if os.name != "nt":
        return True  # icpx on Linux/mac doesn't need MSVC
    if os.environ.get("INCLUDE") and os.environ.get("LIB"):
        return True  # already loaded (e.g. launched via the wrapper script)

    bat = _find_vcvars()
    if not bat:
        logger.warning(
            "MSVC build tools (vcvarsall.bat) not found; icpx can't find its C++ "
            "standard library and Triton kernel build will fail. Install Visual "
            "Studio Build Tools ('Desktop development with C++'), or launch via "
            "scripts/start-deskmate-train.bat, or set DESKMATE_VCVARS."
        )
        return False

    # Run vcvars + `set` via a temp .bat. Doing this through a file (rather than
    # a `cmd /c "call … & set"` one-liner) avoids cmd's quote-stripping rules,
    # which mangle the space-containing "Program Files (x86)" path. A marker
    # line separates vcvars' own banner from the environment dump.
    marker = "__DESKMATE_VCVARS_OK__"
    script = (
        "@echo off\r\n"
        f'call "{bat}" x64 >nul 2>&1\r\n'
        f"echo {marker}\r\n"
        "set\r\n"
    )
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", suffix=".bat", delete=False, encoding="ascii") as fh:
            fh.write(script)
            tmp = fh.name
        proc = subprocess.run(
            ["cmd", "/c", tmp], capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to run vcvarsall.bat (%s): %s", bat, exc)
        return False
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    imported = 0
    seen_marker = False
    for line in proc.stdout.splitlines():
        if marker in line:
            seen_marker = True
            continue
        if not seen_marker or "=" not in line:
            continue
        key, _, val = line.partition("=")
        ku = key.upper()
        if ku in ("INCLUDE", "LIB", "LIBPATH"):
            os.environ[key] = val
            imported += 1
        elif ku == "PATH":
            # Merge: prepend any new MSVC dirs not already on our PATH.
            cur = os.environ.get("PATH", "")
            cur_set = {p.lower() for p in cur.split(os.pathsep)}
            new = [p for p in val.split(os.pathsep)
                   if p and p.lower() not in cur_set]
            if new:
                os.environ["PATH"] = os.pathsep.join(new) + os.pathsep + cur
    ok = bool(os.environ.get("INCLUDE") and os.environ.get("LIB"))
    if ok:
        logger.info("Loaded MSVC build environment from %s (%d vars).", bat, imported)
    else:
        logger.warning("Ran vcvarsall.bat but INCLUDE/LIB still unset (%s).", bat)
    return ok


def _ensure_cxx_compiler(device: str | None = None) -> str | None:
    """Make sure the RIGHT C++ compiler is wired up for Triton's JIT.

    Unsloth/Triton compile kernels at runtime via a C++ compiler; on Windows
    there's none on PATH by default, so the first step dies with "Failed to find
    C++ compiler". **Backend matters:** the Intel XPU backend emits SYCL kernels
    that need Intel's ``icpx`` with ``-fsycl``. Crucially, Triton only adds the
    ``-fsycl`` flag when the compiler it resolves is *exactly* ``icpx`` found on
    PATH **with ``CXX`` unset** (see ``triton/runtime/build.py``); if we pin
    ``CXX=icx`` it silently drops ``-fsycl`` and the SYCL headers fail to compile
    ("cannot use 'throw' with exceptions disabled"). So on ``xpu`` we:

    * put a *complete* ``icpx`` (system oneAPI preferred over the conda
      ``dpcpp_impl`` build, whose SYCL headers are incomplete) first on PATH,
    * **clear** ``CXX``/``CC`` so Triton takes its ``icpx + -fsycl`` path,
    * point ``ZE_PATH`` at a Level-Zero SDK so ``level_zero/ze_api.h`` resolves.

    On non-XPU devices any common compiler is fine and we just set ``CXX``/``CC``.
    Returns the chosen ``cxx`` path, or ``None`` when none is suitable.
    """
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    intel = [("icx", "icx"), ("icpx", "icx")]
    others = [("cl", "cl"), ("g++", "gcc"), ("clang++", "clang")]

    if device == "xpu":
        # icpx is a clang-cl driver — it needs the MSVC stdlib (climits, …).
        # Load it into this process if a plain `deskmate ui` launch didn't.
        _ensure_msvc_env()
        # Prefer icpx (Triton's SYCL path keys on it); fall back to icx for the
        # PATH/dir, but Triton itself will look up `icpx` on PATH.
        found = _find_compiler([("icpx", "icpx"), ("icx", "icx")])
        if not found:
            logger.warning(
                "Intel XPU training needs the icpx compiler (g++/MSVC can't "
                "build SYCL kernels). Install the Intel oneAPI Base Toolkit "
                "(DPC++/C++ Compiler), or set DESKMATE_ONEAPI to its root."
            )
            return None
        cxx_path, _cc = found
        bindir = os.path.dirname(cxx_path)
        if bindir and bindir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        # icpx (clang-cl mode) finds its own runtime libs (libmmd.lib, …) via
        # the MSVC-style `LIB` env var, NOT relative to the exe. Add oneAPI's
        # own lib dirs to LIB, or linking fails with `LNK1104: libmmd.lib`.
        cmplr_root = os.path.dirname(bindir)  # …/compiler/<ver>
        oneapi_libs = [
            os.path.join(cmplr_root, "lib"),
            os.path.join(cmplr_root, "lib", "clang", "lib", "windows"),
        ]
        cur_lib = os.environ.get("LIB", "")
        cur_lib_set = {p.lower() for p in cur_lib.split(os.pathsep)}
        add_lib = [d for d in oneapi_libs
                   if os.path.isdir(d) and d.lower() not in cur_lib_set]
        if add_lib:
            os.environ["LIB"] = (os.pathsep.join(add_lib)
                                 + (os.pathsep + cur_lib if cur_lib else ""))
        # Let Triton resolve `icpx` itself and add -fsycl; a pinned CXX would
        # suppress that flag. Clear both.
        os.environ.pop("CXX", None)
        os.environ.pop("CC", None)
        # Level-Zero SDK headers (ze_api.h) — the Base Toolkit doesn't ship them.
        if not (os.environ.get("ZE_PATH") and os.path.isfile(
                os.path.join(os.environ["ZE_PATH"], "include", "level_zero", "ze_api.h"))):
            ze = _find_ze_path()
            if ze:
                os.environ["ZE_PATH"] = ze
                logger.info("Using Level-Zero SDK (ZE_PATH): %s", ze)
            else:
                logger.warning(
                    "Level-Zero SDK headers (level_zero/ze_api.h) not found; "
                    "Triton XPU kernel compilation will fail. Install the "
                    "level-zero Windows SDK (see docs/16) into the env's "
                    "Library, or set ZE_PATH."
                )
        icpx = shutil.which("icpx")
        logger.info("Intel XPU compiler for Triton: icpx=%s (CXX unset, -fsycl)", icpx)
        return icpx or cxx_path

    # --- non-XPU: any common compiler, pinned via CXX/CC ---
    if os.environ.get("CXX") and shutil.which(os.environ["CXX"]):
        return os.environ["CXX"]
    found = _find_compiler(intel + others)
    if not found:
        logger.warning(
            "No C++ compiler found (icx/cl/g++/clang++). Triton kernel "
            "compilation will fail — set CXX to a C++ compiler before training."
        )
        return None

    cxx_path, cc_path = found
    os.environ["CXX"] = cxx_path
    os.environ["CC"] = cc_path
    bindir = os.path.dirname(cxx_path)
    if bindir and bindir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    logger.info("Using C++ compiler for Triton JIT: %s (device=%s)", cxx_path, device)
    return cxx_path


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
        # Prefer Unsloth (faster, lower VRAM, multi-backend incl. Intel XPU);
        # fall back to plain transformers + peft when it isn't installed.
        self._use_unsloth = HAS_UNSLOTH
        self._lora_applied = False
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

        # Triton (Unsloth's kernel JIT) needs a C++ compiler on PATH; ensure the
        # right one is wired up before any kernel compiles. On XPU this MUST be
        # Intel icx/icpx (SYCL) — g++ cannot build the kernels.
        _ensure_cxx_compiler(self.device)

        # Always free GPU/iGPU memory when training ends — success, failure or
        # OOM — since this runs in the long-lived UI process (see _release_memory).
        try:
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
        finally:
            optimizer = None  # noqa: F841 - drop optimizer state before cache clear
            self._release_memory()

    # -- Internal helpers ----------------------------------------------------

    def _release_memory(self) -> None:
        """Free the model/optimizer and empty the accelerator cache.

        Training runs IN-PROCESS inside the long-lived UI server, so unless we
        drop our references and clear the allocator cache, the base model +
        LoRA + optimizer state keep occupying GPU/iGPU memory until the whole
        process exits. We null out the references, run a GC pass, then call the
        backend-specific ``empty_cache`` so the freed blocks are returned."""
        self.model = None
        self.tokenizer = None
        self._lora_applied = False

        import gc  # noqa: PLC0415
        gc.collect()

        if not HAS_TORCH or torch is None:
            return
        try:
            dev = (self.device or "").split(":")[0]
            if dev == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif dev == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif dev == "mps" and hasattr(torch.backends, "mps") \
                    and torch.backends.mps.is_available():
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            logger.debug("empty_cache on %s failed: %s", self.device, exc)

    def _ensure_tokenizer(self) -> None:
        """Lazily load the tokenizer.

        With the Unsloth backend the tokenizer is loaded together with the model
        in :meth:`_load_model`; this only handles the transformers fallback.
        """
        if self.tokenizer is not None:
            return
        if self._use_unsloth:
            self._load_model()  # Unsloth returns (model, tokenizer) together
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
        """Load the base model (and, for Unsloth, the tokenizer) for fine-tuning."""
        if self.model is not None:
            return

        if self._use_unsloth:
            self._load_model_unsloth()
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

        # cuda/auto: let accelerate shard via device_map. For single-device
        # backends (xpu / mps / cpu) device_map string support varies by
        # accelerate version, so we load without a map and move the model
        # explicitly below — reliable across versions and backends.
        explicit_move_device: str | None = None
        if self.device in ("cuda", "auto"):
            model_kwargs["device_map"] = "auto"
        else:
            explicit_move_device = self.device

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # 4-bit quantized weights are already placed by bitsandbytes; moving
        # them with .to() is unsupported, so only move a non-quantized model.
        if explicit_move_device is not None and not self.config.use_4bit:
            self.model = self.model.to(explicit_move_device)

        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _load_model_unsloth(self) -> None:
        """Load model + tokenizer + LoRA via Unsloth's FastLanguageModel.

        One call returns an optimized model and its tokenizer; a second wraps it
        with LoRA. Unsloth picks the device itself (CUDA / XPU / etc.) and, when
        ``use_4bit`` is set, applies its own 4-bit QLoRA path."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # auto: bf16/fp16 per device capability
            load_in_4bit=self.config.use_4bit,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            use_gradient_checkpointing=(
                "unsloth" if self.config.gradient_checkpointing else False
            ),
        )
        self._lora_applied = True
        logger.info(
            "Unsloth LoRA: model=%s rank=%d alpha=%d 4bit=%s targets=%s",
            self.model_name, self.config.lora_rank, self.config.lora_alpha,
            self.config.use_4bit, self.config.target_modules,
        )

    def _apply_lora(self) -> None:
        """Wrap the loaded model with LoRA adapters.

        With Unsloth this already happened in :meth:`_load_model_unsloth`, so
        this only runs the ``peft`` fallback path."""
        if self._use_unsloth or getattr(self, "_lora_applied", False):
            return
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
            "LoRA applied (peft fallback): rank=%d, alpha=%d, targets=%s",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.target_modules,
        )

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
        # Move inputs to where the model actually lives. Unsloth (and accelerate
        # device_map) place the model itself, so trust the model's device over
        # our _select_device() guess; fall back to self.device otherwise.
        dev = self.device
        try:
            dev = next(self.model.parameters()).device
        except Exception:  # noqa: BLE001 — model not loaded yet / no params
            pass
        return {
            "input_ids": torch.stack(ids).to(dev),
            "attention_mask": torch.stack(masks).to(dev),
            "labels": torch.stack(labels).to(dev),
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
    "HAS_PEFT",
    "HAS_TORCH",
    "HAS_TRANSFORMERS",
    "HAS_UNSLOTH",
    "LoRATrainer",
    "LoRATrainingConfig",
    "missing_training_deps",
]
