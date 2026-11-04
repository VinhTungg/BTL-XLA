"""Read-only legacy checkpoint diagnostic on real CIFAR-100 validation images.

Compare equal task weights with/without a newly initialised context role.
This is not evaluation accuracy, checkpoint conversion, or a training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from tacs.data import build_cifar100
from tacs.model import DINOv3PatchTokenTaskNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/cifar100_seed42.json"))
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if args.pairs < 1 or args.threads < 1:
        parser.error("--pairs and --threads must be positive")
    if not args.checkpoint.is_file() or not args.split_file.is_file():
        parser.error("Checkpoint and split must already exist; this diagnostic does not create/download data.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        parser.error("Checkpoint must include a config dictionary")
    config = payload["config"]
    if (config.get("backbone"), config.get("dataset")) != ("dinov3_vits16", "cifar100"):
        parser.error("Expected a CIFAR-100/DINOv3 checkpoint")
    if config.get("task_network_version", "legacy_v1") != "legacy_v1":
        parser.error("Expected a legacy_v1 checkpoint for before/after comparison")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        parser.error("Checkpoint is missing model_state_dict")
    task_state = {key.removeprefix("task_network."): value for key, value in state.items()
                  if key.startswith("task_network.")}
    if not task_state:
        parser.error("Checkpoint is missing task_network weights")
    legacy = DINOv3PatchTokenTaskNetwork(100, False, "legacy_v1").eval()
    legacy.load_state_dict(task_state, strict=True)
    fixed = DINOv3PatchTokenTaskNetwork(100, False, "role_v2").eval()
    # Explicit, in-memory diagnostic only; never save this as a resumed checkpoint.
    fixed.load_state_dict({**task_state, "context_role_embedding": fixed.context_role_embedding.detach().clone()}, strict=True)
    del payload, state, task_state

    datasets = build_cifar100(args.data_root, args.split_file, download=False,
                             image_size=256, normalization="dinov3")
    if args.pairs > min(len(datasets.validation), len(datasets.candidate_pool)):
        parser.error("--pairs exceeds the available validation or pool images")
    legacy_deltas, fixed_deltas, query_deltas = [], [], []
    with torch.no_grad():
        for index in range(args.pairs):
            query = datasets.validation[index][0].unsqueeze(0)
            context = datasets.candidate_pool[index][0].unsqueeze(0)
            legacy_deltas.append((legacy(query, context) - legacy(context, query)).abs().max().item())
            fixed_deltas.append((fixed(query, context) - fixed(context, query)).abs().max().item())
            query_deltas.append((legacy(query) - fixed(query)).abs().max().item())

    query, label = datasets.validation[0]
    context, _ = datasets.candidate_pool[0]
    proxy = torch.zeros(1, 384, requires_grad=True)
    loss = nn.functional.cross_entropy(
        fixed(query.unsqueeze(0), context.unsqueeze(0), proxy), torch.tensor([label])
    )
    loss.backward()
    role_grad = fixed.context_role_embedding.grad
    gradient_ok = role_grad is not None and bool(torch.isfinite(role_grad).all())
    role_grad_l1 = role_grad.abs().sum().item() if gradient_ok else 0.0
    proxy_grad_l1 = proxy.grad.abs().sum().item() if proxy.grad is not None else 0.0
    passed = (
        max(legacy_deltas) < 1e-5
        and min(fixed_deltas) > 1e-4
        and max(query_deltas) == 0.0
        and role_grad_l1 > 0.0
        and proxy_grad_l1 > 0.0
        and bool(torch.isfinite(loss))
    )
    result = {
        "passed": passed,
        "checkpoint": str(args.checkpoint),
        "data": "real CIFAR-100 validation queries and candidate-pool contexts; no test evaluation",
        "pairs": args.pairs,
        "seed": args.seed,
        "legacy_swap_max_abs_logit_delta_per_pair": legacy_deltas,
        "role_v2_swap_max_abs_logit_delta_per_pair": fixed_deltas,
        "query_only_max_abs_logit_delta": max(query_deltas),
        "role_embedding_gradient_l1": role_grad_l1,
        "context_proxy_gradient_l1": proxy_grad_l1,
        "warning": "Structural check only. role_v2 is untrained; no accuracy improvement is claimed.",
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Context-role diagnostic failed; inspect the reported tolerances and gradients.")


if __name__ == "__main__":
    main()
