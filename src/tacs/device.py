"""Device selection and reproducibility helpers shared by every entry point."""

import os
import platform
import random
import sys

import torch


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr before printing paths.

    This repository lives under a path containing Vietnamese diacritics, and the
    default Windows console codec (cp1252) raises UnicodeEncodeError the moment a
    script prints it. Every CLI in this package calls this first.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def resolve_device(prefer: str = "auto") -> torch.device:
    """Return the device to train on.

    ``auto`` picks CUDA when available and falls back to CPU. Asking for ``cuda``
    explicitly on a machine without CUDA raises instead of silently running on CPU,
    which would otherwise turn a 20-minute run into an overnight one unnoticed.
    """
    choice = prefer.strip().lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"prefer={prefer!r} but CUDA is unavailable "
                f"(torch {torch.__version__}). Use prefer='auto' or 'cpu'."
            )
        return torch.device(choice)
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device preference {prefer!r}; use 'auto', 'cpu' or 'cuda[:i]'")


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch; return a seeded generator for DataLoader shuffling."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dependency, but keep seeding best-effort
        pass
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def dataloader_defaults(device: torch.device, num_workers: int = 0) -> dict[str, object]:
    """DataLoader kwargs tuned per platform.

    ``pin_memory`` only helps when copying to CUDA. Workers use the spawn start method
    on Windows, so ``persistent_workers`` is what keeps the respawn cost off every epoch.
    """
    options: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return options


def describe_environment() -> dict[str, object]:
    """Collect the environment facts the three of us need to compare across machines."""
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    if torch.cuda.is_available():
        info["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info
