"""Đo tốc độ và mức sử dụng bộ nhớ của pipeline dữ liệu."""

import argparse
import time
import psutil
from itertools import islice

import torch
from torch.utils.data import DataLoader

from tacs.data import ContextDataset, build_cifar10
from tacs.device import (
    configure_stdout,
    dataloader_defaults,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    """Đọc cấu hình benchmark từ dòng lệnh."""

    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader của TACS."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )

    return parser.parse_args()


def main() -> None:
    """Tạo DataLoader, đọc các batch và in kết quả."""

    configure_stdout()
    args = parse_args()
    device = resolve_device(args.device)

    datasets = build_cifar10(
        root="data",
        split_file="data/splits/cifar10_seed42.json",
    )

    context_dataset = ContextDataset(
        query_dataset=datasets.query_train,
        candidate_dataset=datasets.candidate_pool,
        num_candidates=8,
        seed=42,
    )

    loader = DataLoader(
        context_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_defaults(
            device=device,
            num_workers=args.workers,
        ),
    )

    process = psutil.Process()
    start_memory = process.memory_info().rss
    peak_memory = start_memory

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    sample_count = 0
    batch_count = 0

    for batch in islice(loader, args.batches):
        query = batch["query"].to(
            device,
            non_blocking=device.type == "cuda",
        )

        candidates = batch["candidates"].to(
            device,
            non_blocking=device.type == "cuda",
        )

        sample_count += query.shape[0]
        batch_count += 1

        current_memory = process.memory_info().rss
        peak_memory = max(peak_memory, current_memory)

        # Giữ hai biến được sử dụng trong vòng lặp benchmark.
        _ = candidates

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time

    print(f"Device: {device}")
    print(f"Số batch: {batch_count}")
    print(f"Số query: {sample_count}")
    print(f"Thời gian: {elapsed:.3f} giây")
    print(
        "Tốc độ: "
        f"{sample_count / elapsed:.2f} query/giây"
    )

    print(
        "RAM ban đầu: "
        f"{start_memory / 1024 ** 2:.2f} MB"
    )

    print(
        "RAM cao nhất: "
        f"{peak_memory / 1024 ** 2:.2f} MB"
    )

    print(
        "RAM tăng thêm: "
        f"{(peak_memory - start_memory) / 1024 ** 2:.2f} MB"
    )

    if device.type == "cuda":
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

        print(
            "Đỉnh bộ nhớ GPU: "
            f"{peak_gpu_memory:.2f} MB"
        )


if __name__ == "__main__":
    main()