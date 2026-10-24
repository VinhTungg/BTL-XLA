"""Runtime and accelerator memory measurements used by experiments."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import psutil
import torch


MB = 1024**2


@dataclass(frozen=True)
class MemoryMetrics:
    rss_start_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float
    peak_vram_allocated_mb: float | None
    peak_vram_reserved_mb: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "rss_start_mb": self.rss_start_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_delta_mb": self.peak_rss_delta_mb,
            "peak_vram_allocated_mb": self.peak_vram_allocated_mb,
            "peak_vram_reserved_mb": self.peak_vram_reserved_mb,
        }


class PeakMemoryMonitor:
    """Sample process RSS and use PyTorch peak counters for CUDA memory."""

    def __init__(self, device: torch.device, sample_interval_s: float = 0.01) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        self.device = device
        self.sample_interval_s = sample_interval_s
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_rss = 0
        self._peak_rss = 0
        self._metrics: MemoryMetrics | None = None

    def _sample(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._peak_rss = max(self._peak_rss, self.process.memory_info().rss)

    def __enter__(self) -> "PeakMemoryMonitor":
        self._start_rss = self.process.memory_info().rss
        self._peak_rss = self._start_rss
        self._stop_event.clear()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s * 4))
        self._peak_rss = max(self._peak_rss, self.process.memory_info().rss)

        peak_allocated = None
        peak_reserved = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / MB
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / MB

        self._metrics = MemoryMetrics(
            rss_start_mb=round(self._start_rss / MB, 2),
            peak_rss_mb=round(self._peak_rss / MB, 2),
            peak_rss_delta_mb=round((self._peak_rss - self._start_rss) / MB, 2),
            peak_vram_allocated_mb=(
                round(peak_allocated, 2) if peak_allocated is not None else None
            ),
            peak_vram_reserved_mb=(
                round(peak_reserved, 2) if peak_reserved is not None else None
            ),
        )

    def result(self) -> MemoryMetrics:
        if self._metrics is None:
            raise RuntimeError("PeakMemoryMonitor has not completed")
        return self._metrics
