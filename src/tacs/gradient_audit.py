"""Train-mode gradient probe on real validation images; NEVER optimizer.step.

Frozen checkpoint, eval image transforms, train-mode Gumbel selection. This
measures local gradients, not historical training dynamics or test accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import SplitIndices, cifar_transforms
from .device import configure_stdout, resolve_device, seed_everything
from .losses import TACSLoss
from .model import TACSModel, create_task_network, create_vision_encoder
from .retrieval import CosineSimilarityRetriever, TopKContextDataset
from .selector_audit import encode_bank, validate_provenance
from .train import _autocast_context, _load_selected_context, _load_standard_encoder, _selected_pool_positions, _sha256_file


def standardize_reward(reward, eps=1e-6):
    reward = reward.float()
    return (reward - reward.mean()) / (reward.std(unbiased=False) + eps)


def reward_window_summary(reward_batches):
    """Compare microbatch normalization to ONE hypothetical whole window."""
    batches = [torch.as_tensor(r, dtype=torch.float32) for r in reward_batches]
    if not batches or any(r.ndim != 1 or not len(r) or not bool(torch.isfinite(r).all()) for r in batches):
        raise ValueError("Need nonempty finite reward vectors")
    rewards = torch.cat(batches)
    micro = torch.cat([standardize_reward(r) for r in batches])
    whole = standardize_reward(rewards)
    return {
        "queries": len(rewards), "mean_reward": rewards.mean().item(),
        "reward_std": rewards.std(unbiased=False).item(),
        "reward_min": rewards.min().item(), "reward_max": rewards.max().item(),
        "positive_reward_fraction": (rewards > 0).float().mean().item(),
        "mean_abs_micro_vs_window_advantage_delta": (micro - whole).abs().mean().item(),
        "micro_vs_window_advantage_sign_flip_fraction": (micro * whole < 0).float().mean().item(),
        "negative_reward_positive_micro_advantage_count": int(((rewards < 0) & (micro > 0)).sum()),
        "positive_reward_negative_micro_advantage_count": int(((rewards > 0) & (micro < 0)).sum()),
        "all_positive_reward_microbatches": sum(bool((r > 0).all()) for r in batches),
        "all_negative_reward_microbatches": sum(bool((r < 0).all()) for r in batches),
        "zero_variance_microbatches": sum(bool(r.std(unbiased=False) == 0) for r in batches),
        "note": "Whole-window advantage is descriptive only; no global-batch gradient or training comparison was run. Relative advantage may reward a below-zero reward that is better than its microbatch peers.",
    }


def gradient_pair_summary(task, policy, policy_weight=1.0):
    """Norms and alignment of matching parameter-gradient lists, CPU tensors."""
    if len(task) != len(policy) or not task:
        raise ValueError("Gradient lists must be nonempty and aligned")
    task_sq = policy_sq = dot = 0.0
    task_nonzero = policy_nonzero = task_connected = policy_connected = 0
    for left, right in zip(task, policy):
        if left is not None:
            task_connected += 1
            if not bool(torch.isfinite(left).all()):
                raise ValueError("Non-finite task gradient; lower loss scale or use --no-amp")
            left = left.float()
            task_sq += float((left * left).sum(dtype=torch.float64))
            task_nonzero += int(torch.count_nonzero(left))
        if right is not None:
            policy_connected += 1
            if not bool(torch.isfinite(right).all()):
                raise ValueError("Non-finite policy gradient; lower loss scale or use --no-amp")
            right = right.float()
            policy_sq += float((right * right).sum(dtype=torch.float64))
            policy_nonzero += int(torch.count_nonzero(right))
        if left is not None and right is not None:
            if left.shape != right.shape:
                raise ValueError("Gradient shapes do not match")
            dot += float((left * right).sum(dtype=torch.float64))
    task_norm, policy_norm = math.sqrt(task_sq), math.sqrt(policy_sq)
    combined_sq = task_sq + policy_weight ** 2 * policy_sq + 2 * policy_weight * dot
    return {
        "task_l2": task_norm, "policy_l2": policy_norm,
        "weighted_policy_to_task_l2_ratio": abs(policy_weight) * policy_norm / task_norm if task_norm else None,
        "task_policy_cosine": dot / (task_norm * policy_norm) if task_norm and policy_norm else None,
        "combined_l2": math.sqrt(max(0., combined_sq)),
        "task_connected_tensors": task_connected, "policy_connected_tensors": policy_connected,
        "task_nonzero_elements": task_nonzero, "policy_nonzero_elements": policy_nonzero,
        "finite": True,
    }


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_gradients(loss, inputs, scale, retain_graph):
    gradients = torch.autograd.grad(loss * scale, inputs, retain_graph=retain_graph, allow_unused=True)
    # Transfer/de-scale immediately so two full parameter gradients do not stay on GPU.
    return [None if g is None else g.detach().float().cpu().div_(scale) for g in gradients]


def accumulate(accumulator, gradients, divisor):
    for index, grad in enumerate(gradients):
        if grad is not None:
            if accumulator[index] is None:
                accumulator[index] = grad.clone().div_(divisor)
            else:
                accumulator[index].add_(grad, alpha=1 / divisor)


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    metrics_path = args.checkpoint.parent / "metrics.json"
    for path in (args.checkpoint, metrics_path, args.split_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = _sha256_file(args.checkpoint)
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    split_hash = _sha256_file(args.split_file)
    cfg = validate_provenance(payload, metrics, split_hash)
    split = SplitIndices.load(args.split_file)
    split.validate(50_000, "cifar100")
    if cfg.backbone != "dinov3_vits16":
        raise ValueError("Real gradient probe currently requires DINOv3 ViT-S/16")
    if args.batches != cfg.gradient_accumulation_steps or args.batch_size != cfg.batch_size:
        raise ValueError("Use the checkpoint microbatch size and accumulation-window length")
    count = args.batches * args.batch_size
    if count > len(split.validation):
        raise ValueError("Probe exceeds validation size")
    for key in ("query_train", "candidate_pool", "validation"):
        if len(getattr(split, key)) != metrics["data"]["full_sizes"][key]:
            raise ValueError(f"Split size mismatch: {key}")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp = cfg.amp and device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = TACSModel(cfg.num_classes, cfg.embedding_dim, cfg.selector_temperature, True,
                      encoder=create_vision_encoder(cfg.backbone, cfg.embedding_dim, False),
                      task_network=create_task_network(cfg.backbone, cfg.num_classes, cfg.embedding_dim, False, cfg.task_network_version))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    epoch = payload["epoch"]
    del payload
    before = state_digest(model)
    # Only need derivatives wrt selector; frozen task parameters remain constants
    # in the same forward function. Proxy-to-selector autograd is NOT disabled.
    model.task_network.requires_grad_(False)
    from torchvision.datasets import CIFAR100
    _, transform = cifar_transforms("cifar100", cfg.image_size, "dinov3")
    data = CIFAR100(str(args.data_root), train=True, download=False, transform=transform)
    pool = Subset(data, split.candidate_pool)
    generator = torch.Generator().manual_seed(args.seed)
    positions = torch.randperm(len(split.validation), generator=generator)[:count].tolist()
    queries = Subset(data, [split.validation[i] for i in positions])
    topk = None
    standard_hash = None
    if cfg.baseline == "two_stage":
        if args.standard_encoder is None:
            raise ValueError("Two-Stage requires --standard-encoder from the same pilot")
        cfg.standard_encoder_path = str(args.standard_encoder)
        standard, _, standard_hash = _load_standard_encoder(cfg, device, split_hash)
        bank = encode_bank(standard, pool, device, args.batch_size, False, "retrieval bank", True)
        retriever = CosineSimilarityRetriever(bank, torch.tensor(pool.indices),
                                             torch.tensor([data.targets[i] for i in pool.indices]), pool)
        topk = TopKContextDataset(queries, pool, retriever, standard, device,
                                 args.batch_size, min(cfg.topk_candidates, len(pool))).cached_positions
        del standard, bank, retriever
    model.to(device)
    bank = encode_bank(model.selector.encoder, pool, device, args.batch_size, amp, "selector bank").to(device).detach()
    model.train()  # Restore after bank encoder.eval(); Gumbel sampling, not eval argmax.
    parameters = [p for p in model.selector.parameters() if p.requires_grad]
    task_acc, policy_acc = [None] * len(parameters), [None] * len(parameters)
    criterion = TACSLoss(cfg.policy_weight)
    records, rewards = [], []
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": epoch, "baseline": cfg.baseline, "split_sha256": split_hash,
        "metrics_sha256": _sha256_file(metrics_path), "standard_encoder_sha256": standard_hash,
        "seed": args.seed, "query_indices": queries.indices, "queries": count,
        "batches": args.batches, "batch_size": args.batch_size,
        "device": str(device), "amp": amp, "fixed_loss_scale": args.loss_scale,
        "torch_version": str(torch.__version__), "training_config": metrics["config"],
        "selector_train_mode": True, "task_train_mode": True, "task_parameters_frozen_for_probe": True,
        "candidate_bank_detached": True, "test_evaluated": False, "optimizer_steps": 0,
        "data": "real validation queries, evaluation transforms; full real candidate pool",
        "state_sha256_before": before,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loader = DataLoader(queries, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with (args.output / "batches.jsonl").open("x", encoding="utf-8") as output:
        for index, (query, target) in enumerate(loader):
            seed_everything(args.seed + 1000 + index)
            query, target = query.to(device), target.to(device)
            local_topk = None if topk is None else topk[index * args.batch_size:(index + 1) * args.batch_size].to(device)
            candidates = bank if local_topk is None else bank[local_topk]
            with _autocast_context(device, amp):
                selection = model.selector.forward_embedding_bank(query, candidates)
                chosen = _selected_pool_positions({} if local_topk is None else {"candidate_positions": local_topk}, selection[1])
                context = _load_selected_context(pool, chosen, device)
                prediction = model.forward_embedding_bank(query, candidates, context, selection)
                loss = criterion(prediction, target)
                context_ce = F.cross_entropy(prediction.context_logits, target, reduction="none")
                no_context_ce = F.cross_entropy(prediction.no_context_logits, target, reduction="none")
            reward = (no_context_ce - context_ce).detach().float().cpu()
            advantage = standardize_reward(reward)
            if not bool(torch.isfinite(loss.total)):
                raise ValueError("Non-finite loss")
            inputs = parameters + [prediction.selector_scores]
            task_all = cpu_gradients(loss.task, inputs, args.loss_scale, True)
            policy_all = cpu_gradients(loss.policy, inputs, args.loss_scale, False)
            task_grads, task_scores = task_all[:-1], task_all[-1]
            policy_grads, policy_scores = policy_all[:-1], policy_all[-1]
            stats = gradient_pair_summary(task_grads, policy_grads, cfg.policy_weight)
            if task_scores is None or policy_scores is None:
                raise ValueError("Missing task/policy score gradient")
            score_stats = gradient_pair_summary([task_scores], [policy_scores], cfg.policy_weight)
            probabilities = prediction.selector_scores.detach().float().cpu().softmax(-1)
            weights = prediction.selection_weights.detach().float().cpu()
            expected = advantage[:, None] * (probabilities - weights) / len(target)
            accumulate(task_acc, task_grads, args.batches)
            accumulate(policy_acc, policy_grads, args.batches)
            record = {
                "batch": index, "gumbel_seed": args.seed + 1000 + index,
                "query_indices": queries.indices[index * args.batch_size:(index + 1) * args.batch_size],
                "targets": target.detach().cpu().tolist(), "selected_pool_positions": chosen.detach().cpu().tolist(),
                "context_ce": context_ce.detach().float().cpu().tolist(),
                "no_context_ce": no_context_ce.detach().float().cpu().tolist(),
                "reward": reward.tolist(), "microbatch_advantage": advantage.tolist(),
                "task_loss": float(loss.task.detach()), "policy_loss": float(loss.policy.detach()),
                "mean_reward": float(loss.mean_reward), "encoder_gradients": stats,
                "score_gradients": score_stats,
                "policy_score_gradient_max_abs_error_vs_formula": (policy_scores - expected).abs().max().item(),
                "sampled_vs_argmax_fraction": (weights.argmax(-1) == probabilities.argmax(-1)).float().mean().item(),
            }
            rewards.append(reward)
            records.append(record)
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(f"[gradient] {index + 1}/{args.batches} task={stats['task_l2']:.6g} policy={stats['policy_l2']:.6g} cosine={stats['task_policy_cosine']} reward={reward.mean().item():.6g}", flush=True)
            del task_all, policy_all, task_grads, policy_grads, inputs, prediction, loss, selection, context, context_ce, no_context_ce
    after = state_digest(model)
    file_unchanged = _sha256_file(args.checkpoint) == checkpoint_hash
    if before != after or not file_unchanged:
        raise RuntimeError("State/checkpoint changed during supposedly read-only gradient probe")
    combined = gradient_pair_summary(task_acc, policy_acc, cfg.policy_weight)
    report = {
        **manifest, "status": "complete", "state_sha256_after": after,
        "model_state_unchanged": before == after, "checkpoint_file_unchanged": file_unchanged,
        "reward_window": reward_window_summary(rewards),
        "accumulated_encoder_gradients": combined,
        "all_batches_have_nonzero_task_gradient": all(r["encoder_gradients"]["task_l2"] > 0 for r in records),
        "all_batches_have_nonzero_policy_gradient": all(r["encoder_gradients"]["policy_l2"] > 0 for r in records),
        "opposing_gradient_batches": sum((r["encoder_gradients"]["task_policy_cosine"] or 0) < 0 for r in records),
        "max_policy_score_gradient_formula_error": max(r["policy_score_gradient_max_abs_error_vs_formula"] for r in records),
        "elapsed_s": time.perf_counter() - started,
        "limitations": ["One fixed checkpoint and one sampled accumulation window; not historical training evidence.",
                        "Validation images use evaluation transforms, not train augmentation.",
                        "Task parameters are constants; selector gradients still flow through task network via the production zero-value proxy.",
                        "AMP uses declared fixed diagnostic loss scale, not a restored training GradScaler state.",
                        "No optimizer step; nonzero/aligned gradients alone do not establish useful learning.",
                        "Microbatch vs window comparison is reward/advantage only, not an alternate full-window parameter gradient."],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--standard-encoder", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/cifar100_seed42.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--loss-scale", type=float, default=128.)
    args = parser.parse_args()
    if min(args.batches, args.batch_size, args.threads) < 1 or not math.isfinite(args.loss_scale) or args.loss_scale <= 0:
        parser.error("Batch counts, threads and finite loss scale must be positive")
    try:
        print(json.dumps(run(args), indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"GRADIENT AUDIT FAILED: {error}\nNo optimizer step was requested; incomplete output is not a completed result.\n")


if __name__ == "__main__":
    main()
