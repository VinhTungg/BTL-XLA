"""Read-only, paired validation audit of a trained role_v2 selector.

No training, test evaluation, checkpoint conversion or implicit downloads.
All interventions use the SAME task weights; no-context here is an ablation,
not the independently trained No-context baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import SplitIndices, cifar_transforms
from .device import configure_stdout, resolve_device, seed_everything
from .model import TACSModel, create_task_network, create_vision_encoder
from .retrieval import CosineSimilarityRetriever, TopKContextDataset
from .train import (
    ExperimentConfig, _autocast_context, _load_selected_context,
    _load_standard_encoder, _selected_pool_positions, _sha256_file,
)


def validate_provenance(payload: dict, metrics: dict, split_hash: str) -> ExperimentConfig:
    """Reject accidental legacy/subset/mismatched artifacts before inference."""
    if not isinstance(payload, dict) or not isinstance(metrics, dict):
        raise ValueError("Checkpoint and metrics.json must contain dictionaries")
    config = payload.get("config")
    if not isinstance(config, dict) or config != metrics.get("config"):
        raise ValueError("Checkpoint config and metrics.json config must match exactly")
    if config.get("task_network_version") != "role_v2":
        raise ValueError("This pilot audit requires role_v2; do not substitute legacy weights")
    cfg = ExperimentConfig(**config)
    if cfg.baseline not in {"tacs", "two_stage"} or cfg.dataset != "cifar100":
        raise ValueError("Expected a CIFAR-100 tacs/two_stage checkpoint")
    data = metrics.get("data", {})
    if data.get("source") != cfg.dataset or data.get("split_sha256") != split_hash:
        raise ValueError("Dataset/split SHA256 does not match training provenance")
    if cfg.dry_run or cfg.max_train_samples or cfg.max_val_samples:
        raise ValueError("Subset/dry-run training is not the full-data pilot")
    sizes = data.get("full_sizes", {})
    effective = data.get("effective_sizes", {})
    if any(not sizes.get(k) or sizes[k] != effective.get(k)
           for k in ("query_train", "candidate_pool", "validation")):
        raise ValueError("Incomplete dataset size provenance")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("Checkpoint is missing model_state_dict")
    if not math.isfinite(float(payload.get("val_accuracy", float("nan")))):
        raise ValueError("Checkpoint is missing finite val_accuracy")
    return cfg


def summarize(rows: list[dict], random_trials: int) -> dict:
    if not rows or random_trials < 1:
        raise ValueError("Need validation rows and at least one random trial")
    n = len(rows)
    strategies = ["selected", "no_context"] + [f"random_{i}" for i in range(random_trials)]
    summary = {}
    for name in strategies:
        summary[name] = {
            "accuracy": sum(r[name + "_correct"] for r in rows) / n,
            "mean_cross_entropy": sum(r[name + "_ce"] for r in rows) / n,
        }
    # Average random trials WITHIN each query before paired comparisons.
    for reference in ("no_context", "random"):
        accuracy_deltas, rewards = [], []
        for row in rows:
            refs = ["no_context"] if reference == "no_context" else strategies[2:]
            accuracy_deltas.append(row["selected_correct"] - sum(row[s + "_correct"] for s in refs) / len(refs))
            rewards.append(sum(row[s + "_ce"] for s in refs) / len(refs) - row["selected_ce"])
        summary["selected_vs_" + reference] = {
            "accuracy_delta_pp": 100 * sum(accuracy_deltas) / n,
            "mean_ce_reduction": sum(rewards) / n,
            "queries_with_lower_ce": sum(x > 0 for x in rewards),
            "queries_with_higher_ce": sum(x < 0 for x in rewards),
        }
    counts = Counter(r["selected_pool_position"] for r in rows)
    summary["selection"] = {
        "unique_contexts": len(counts),
        "top1_share": counts.most_common(1)[0][1] / n,
        "top10_share": sum(v for _, v in counts.most_common(10)) / n,
        "most_frequent_pool_positions": counts.most_common(10),
        "same_label_fraction": sum(r["selected_label"] == r["target"] for r in rows) / n,
        "mean_policy_entropy": sum(r["policy_entropy"] for r in rows) / n,
        "mean_top1_probability": sum(r["top1_probability"] for r in rows) / n,
    }
    return summary


def encode_bank(encoder, dataset, device, batch_size, amp, label, normalize=False):
    banks = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    with torch.no_grad():
        for step, (images, _) in enumerate(loader, 1):
            with _autocast_context(device, amp):
                bank = encoder(images.to(device))
                if normalize:
                    bank = F.normalize(bank, dim=-1)
            if not bool(torch.isfinite(bank).all()):
                raise ValueError(f"Non-finite {label} embeddings")
            banks.append(bank.detach().cpu())
            if step == 1 or step % 100 == 0 or step == len(loader):
                print(f"[{label}] {min(step * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return torch.cat(banks)


def run(args) -> dict:
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output}")
    for path in (args.checkpoint, args.split_file, args.checkpoint.parent / "metrics.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    metrics = json.loads((args.checkpoint.parent / "metrics.json").read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    split_hash = _sha256_file(args.split_file)
    cfg = validate_provenance(payload, metrics, split_hash)
    split = SplitIndices.load(args.split_file)
    split.validate(50_000, "cifar100")
    for key in ("query_train", "candidate_pool", "validation"):
        if len(getattr(split, key)) != metrics["data"]["full_sizes"][key]:
            raise ValueError(f"Split size mismatch: {key}")
    if args.max_queries > len(split.validation):
        raise ValueError("--max-queries exceeds validation size")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    amp = cfg.amp and device.type == "cuda" and not args.no_amp
    # Match the trainer's CUDA matmul policy, without changing checkpoint weights.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = TACSModel(
        cfg.num_classes, cfg.embedding_dim, cfg.selector_temperature, True,
        encoder=create_vision_encoder(cfg.backbone, cfg.embedding_dim, pretrained=False),
        task_network=create_task_network(cfg.backbone, cfg.num_classes, cfg.embedding_dim,
                                         pretrained=False, task_network_version=cfg.task_network_version),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    checkpoint_accuracy = float(payload["val_accuracy"])
    checkpoint_epoch = payload.get("epoch")
    del payload
    standard = None
    standard_hash = None
    if cfg.baseline == "two_stage":
        if args.standard_encoder is None:
            raise ValueError("Two-Stage requires --standard-encoder from the SAME pilot's no_context run")
        cfg.standard_encoder_path = str(args.standard_encoder)
        standard, _, standard_hash = _load_standard_encoder(cfg, torch.device("cpu"), split_hash)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch, "checkpoint_val_accuracy": checkpoint_accuracy,
        "split_sha256": split_hash, "standard_encoder_sha256": standard_hash,
        "baseline": cfg.baseline, "training_seed": cfg.seed, "audit_seed": args.seed,
        "task_network_version": cfg.task_network_version,
        "training_config": metrics["config"],
        "metrics_sha256": _sha256_file(args.checkpoint.parent / "metrics.json"),
        "device": str(device), "amp": amp, "torch_version": str(torch.__version__),
        "batch_size": args.batch_size, "random_trials": args.random_trials,
        "test_evaluated": False, "weights_updated": False,
    }
    if args.check_only:
        return {**manifest, "status": "checkpoint_and_split_preflight_passed",
                "warning": "No data inference performed; accuracy is not reverified."}

    # Instantiate only the real training partition. Never open the CIFAR test dataset.
    from torchvision.datasets import CIFAR100
    _, transform = cifar_transforms("cifar100", cfg.image_size,
                                    "dinov3" if cfg.backbone == "dinov3_vits16" else "cifar")
    real_data = CIFAR100(str(args.data_root), train=True, transform=transform, download=False)
    pool = Subset(real_data, split.candidate_pool)
    generator = torch.Generator().manual_seed(args.seed)
    positions = list(range(len(split.validation)))
    if args.max_queries:
        positions = torch.randperm(len(positions), generator=generator)[:args.max_queries].tolist()
    queries = Subset(real_data, [split.validation[i] for i in positions])
    topk = None
    if standard is not None:
        print("[retrieval] Rebuilding standard-encoder bank in FP32 (same as trainer)", flush=True)
        standard.to(device)
        bank = encode_bank(standard, pool, device, args.batch_size, False, "retrieval bank", normalize=True)
        retriever = CosineSimilarityRetriever(bank, torch.tensor(pool.indices),
                                              torch.tensor([real_data.targets[i] for i in pool.indices]), pool)
        topk = TopKContextDataset(queries, pool, retriever, standard, device,
                                 args.batch_size, min(cfg.topk_candidates, len(pool))).cached_positions
        del standard, retriever, bank
    model.to(device).eval()
    bank = encode_bank(model.selector.encoder, pool, device, args.batch_size, amp, "selector bank").to(device)
    candidate_count = len(pool) if topk is None else topk.shape[1]
    # Precompute each trial independently, so RNG does not depend on inference batch size.
    random_local = torch.stack([torch.randint(candidate_count, (len(queries),), generator=generator)
                                for _ in range(args.random_trials)], dim=1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    offset = 0
    loader = DataLoader(queries, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with (args.output / "queries.jsonl").open("x", encoding="utf-8") as output, torch.no_grad():
        for query, target in loader:
            query, target = query.to(device), target.to(device)
            count = len(target)
            local_topk = None if topk is None else topk[offset:offset + count].to(device)
            candidate_bank = bank if local_topk is None else bank[local_topk]
            with _autocast_context(device, amp):
                selection = model.selector.forward_embedding_bank(query, candidate_bank)
                if not bool(torch.isfinite(selection[0]).all()):
                    raise ValueError("Non-finite selector scores")
                selected = _selected_pool_positions({} if local_topk is None else
                                                    {"candidate_positions": local_topk}, selection[1])
                probabilities = selection[0].float().softmax(-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
                strategies = {"selected": selected, "no_context": None}
                for trial in range(args.random_trials):
                    local = random_local[offset:offset + count, trial].to(device)
                    strategies[f"random_{trial}"] = local if local_topk is None else local_topk.gather(1, local[:, None]).squeeze(1)
                predictions = {}
                for name, chosen in strategies.items():
                    context = None if chosen is None else _load_selected_context(pool, chosen, device)
                    logits = model.task_network(query, context)
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError(f"Non-finite logits: {name}")
                    predictions[name] = (logits.argmax(-1).cpu(), F.cross_entropy(logits.float(), target, reduction="none").cpu())
            for i in range(count):
                pool_pos = int(selected[i])
                row = {"query_index": queries.indices[offset + i], "target": int(target[i]),
                       "selected_pool_position": pool_pos, "selected_cifar_index": pool.indices[pool_pos],
                       "selected_label": real_data.targets[pool.indices[pool_pos]],
                       "policy_entropy": float(entropy[i]), "top1_probability": float(probabilities[i].max())}
                for name, (pred, loss) in predictions.items():
                    row[name + "_prediction"] = int(pred[i])
                    row[name + "_correct"] = int(pred[i]) == row["target"]
                    row[name + "_ce"] = float(loss[i])
                    if strategies[name] is not None:
                        row[name + "_pool_position"] = int(strategies[name][i])
                rows.append(row)
                output.write(json.dumps(row) + "\n")
            offset += count
            if offset == count or offset % (args.batch_size * 25) == 0 or offset == len(queries):
                output.flush()
                print(f"[audit] {offset}/{len(queries)} queries | elapsed {time.perf_counter() - started:.1f}s", flush=True)
    summary = summarize(rows, args.random_trials)
    full = len(queries) == len(split.validation)
    delta = summary["selected"]["accuracy"] - checkpoint_accuracy
    report = {**manifest, "status": "complete", "queries": len(rows), "full_validation": full,
              "candidate_pool_size": len(pool), "candidates_per_query": candidate_count,
              "random_scope": "full_pool" if topk is None else "same_retrieval_topk",
              "results": summary, "elapsed_s": time.perf_counter() - started,
              "validation_reproduction": {"checked": full, "accuracy_delta_pp": 100 * delta if full else None,
                  "within_tolerance": abs(delta) <= args.accuracy_tolerance if full else None,
                  "tolerance": args.accuracy_tolerance},
              "limitations": ["Single checkpoint/seed; not evidence of improvement over the paper.",
                              "No-context intervention uses selector-run task weights, not a separately trained baseline.",
                              "Random is averaged across trials on the same queries, not extra independent queries.",
                              "No training-time gradient measurement. Concentration alone does not prove collapse.",
                              "Two-stage encoder schema/split checked; old metrics do not record its hash, so original identity needs artifact provenance."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/cifar100_seed42.json"))
    parser.add_argument("--standard-encoder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=256, help="0 = full validation; otherwise seeded random subset")
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--accuracy-tolerance", type=float, default=0.001, help="Fraction, not percent; default 0.1 percentage points")
    args = parser.parse_args()
    if args.max_queries < 0 or min(args.batch_size, args.threads, args.random_trials) < 1:
        parser.error("Sizes must be positive; only --max-queries may be zero")
    if not math.isfinite(args.accuracy_tolerance) or not 0 <= args.accuracy_tolerance <= 1:
        parser.error("--accuracy-tolerance must be finite and within [0,1]")
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"AUDIT FAILED: {error}\nNo checkpoint was changed. Incomplete outputs must not be treated as results.\n")
    print(json.dumps(report, indent=2))
    if report.get("validation_reproduction", {}).get("within_tolerance") is False:
        parser.exit(2, "Validation accuracy differs from checkpoint beyond tolerance; inspect report before interpretation.\n")


if __name__ == "__main__":
    main()
