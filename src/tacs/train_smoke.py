import argparse

import torch
from torch.utils.data import DataLoader

from .data import ContextDataset, build_cifar10
from .device import configure_stdout, resolve_device, seed_everything
from .losses import TACSLoss
from .model import TACSModel


def run(
    steps: int,
    seed: int = 7,
    data_root: str = "data",
    split_file: str = "data/splits/cifar10_seed42.json",
    batch_size: int = 4,
    num_candidates: int = 5,
    device_name: str = "auto",
) -> None:
    """Chạy vài bước train trên CIFAR-10 thật để kiểm tra nhanh pipeline."""
    configure_stdout()
    generator = seed_everything(seed)
    device = resolve_device(device_name)
    datasets = build_cifar10(root=data_root, split_file=split_file, download=False)
    context_dataset = ContextDataset(
        query_dataset=datasets.query_train,
        candidate_dataset=datasets.candidate_pool,
        num_candidates=num_candidates,
        seed=seed,
    )
    loader = DataLoader(
        context_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = TACSModel(num_classes=10, embedding_dim=32).to(device)
    objective = TACSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batches = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        query = batch["query"].to(device)
        candidates = batch["candidates"].to(device)
        targets = batch["target"].to(device)
        loss = objective(model(query, candidates), targets)
        optimizer.zero_grad()
        loss.total.backward()
        optimizer.step()
        print(
            f"step={step} total={loss.total.item():.4f} "
            f"task={loss.task.item():.4f} policy={loss.policy.item():.4f} "
            f"reward={loss.mean_reward.item():.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke training TACS trên CIFAR-10 thật.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split-file", type=str, default="data/splits/cifar10_seed42.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(
        steps=args.steps,
        seed=args.seed,
        data_root=args.data_root,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        device_name=args.device,
    )
