"""Trực quan hóa batch query và candidates cho TACS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import ContextDataset, build_cifar10
from .device import configure_stdout



def _denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Chuyển ảnh từ chuẩn hóa CIFAR về dạng [0, 1] để hiển thị."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=image.dtype)
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=image.dtype)
    return torch.clamp(image * std[:, None, None] + mean[:, None, None], 0.0, 1.0)


def make_context_grid(
    batch: Mapping[str, torch.Tensor],
    num_queries: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Vẽ lưới query + candidates cho một batch.

    Mỗi hàng là một query; cột đầu tiên là query image, các cột còn lại là
    candidate images của query đó.
    """
    query = batch["query"]
    candidates = batch["candidates"]

    if query.ndim != 4:
        raise ValueError("query phải có shape [B, C, H, W]")
    if candidates.ndim != 5:
        raise ValueError("candidates phải có shape [B, N, C, H, W]")
    if query.shape[0] != candidates.shape[0]:
        raise ValueError("Số query và số batch candidate không khớp")

    if num_queries is None:
        num_queries = query.shape[0]
    num_queries = min(int(num_queries), query.shape[0])
    num_candidates = candidates.shape[1]

    if figsize is None:
        figsize = (2.2 * (num_candidates + 1), 2.8 * num_queries)

    fig, axes = plt.subplots(
        num_queries,
        num_candidates + 1,
        figsize=figsize,
        squeeze=False,
    )

    query_labels = batch.get("target")
    candidate_labels = batch.get("candidate_targets")

    for row in range(num_queries):
        query_image = _denormalize_image(query[row].detach().cpu())
        query_image = query_image.permute(1, 2, 0).numpy()

        axes[row, 0].imshow(query_image)
        axes[row, 0].set_title(
            f"Query {row}\nlabel={int(query_labels[row].item()) if query_labels is not None else 'n/a'}",
            fontsize=8,
        )
        axes[row, 0].axis("off")

        for col in range(num_candidates):
            candidate_image = _denormalize_image(candidates[row, col].detach().cpu())
            candidate_image = candidate_image.permute(1, 2, 0).numpy()
            axis = axes[row, col + 1]
            axis.imshow(candidate_image)
            if candidate_labels is not None:
                axis.set_title(
                    f"C{col}\nlabel={int(candidate_labels[row, col].item())}",
                    fontsize=7,
                )
            else:
                axis.set_title(f"C{col}", fontsize=7)
            axis.axis("off")

    fig.tight_layout()
    return fig


def _sample_batch(
    root: str | Path,
    split_file: str | Path,
    batch_size: int = 4,
    num_candidates: int = 8,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Lấy một batch mẫu từ CIFAR-10 và candidate pool."""
    datasets = build_cifar10(root=root, split_file=split_file)
    context_dataset = ContextDataset(
        query_dataset=datasets.query_train,
        candidate_dataset=datasets.candidate_pool,
        num_candidates=num_candidates,
        seed=seed,
    )
    loader = torch.utils.data.DataLoader(
        context_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader))


def main() -> None:
    """Hiển thị hoặc lưu ảnh minh họa cho query và candidate."""
    configure_stdout()
    parser = argparse.ArgumentParser(description="Hiển thị query + candidates cho TACS.")
    parser.add_argument("--root", type=str, default="data", help="Thư mục chứa CIFAR-10")
    parser.add_argument(
        "--split-file",
        type=str,
        default="data/splits/cifar10_seed42.json",
        help="Đường dẫn file split",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Số query trên một batch")
    parser.add_argument("--num-candidates", type=int, default=8, help="Số candidates mỗi query")
    parser.add_argument("--seed", type=int, default=42, help="Seed candidate sampling")
    parser.add_argument(
        "--output",
        type=str,
        default="query_candidates_preview.png",
        help="Tên file hình ảnh để lưu",
    )
    args = parser.parse_args()

    batch = _sample_batch(
        root=args.root,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )

    fig = make_context_grid(batch, num_queries=args.batch_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Đã lưu trực quan hóa tại: {output_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
