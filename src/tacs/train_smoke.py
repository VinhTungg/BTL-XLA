import argparse

import torch

from .losses import TACSLoss
from .model import TACSModel


def run(steps: int, seed: int = 7) -> None:
    torch.manual_seed(seed)
    model = TACSModel(num_classes=3, embedding_dim=32)
    objective = TACSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        query = torch.randn(4, 3, 32, 32)
        candidates = torch.randn(4, 5, 3, 32, 32)
        targets = torch.randint(0, 3, (4,))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    run(args.steps)

