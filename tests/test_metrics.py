import time
import unittest

import torch

from tacs.metrics import PeakMemoryMonitor


class PeakMemoryMonitorTest(unittest.TestCase):
    def test_cpu_monitor_reports_process_peak_without_vram(self) -> None:
        with PeakMemoryMonitor(torch.device("cpu"), sample_interval_s=0.001) as monitor:
            allocation = bytearray(4 * 1024 * 1024)
            allocation[0] = 1
            time.sleep(0.01)

        metrics = monitor.result()
        self.assertGreater(metrics.rss_start_mb, 0)
        self.assertGreaterEqual(metrics.peak_rss_mb, metrics.rss_start_mb)
        self.assertGreaterEqual(metrics.peak_rss_delta_mb, 0)
        self.assertIsNone(metrics.peak_vram_allocated_mb)
        self.assertIsNone(metrics.peak_vram_reserved_mb)
