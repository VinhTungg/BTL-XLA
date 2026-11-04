"""Run reproducible multi-seed TACS experiments and aggregate their metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .device import configure_stdout
from .model import TASK_NETWORK_VERSIONS
from .train import METRICS_SCHEMA_VERSION


BASELINES = ("no_context", "random", "similarity", "tacs", "two_stage")
DEPENDENT_BASELINES = frozenset({"similarity", "two_stage"})
DEFAULT_SEEDS = (42, 43, 44)
SUMMARY_METRICS = (
    "best_val_accuracy",
    "test_accuracy",
    "end_to_end_time_s",
    "peak_rss_mb",
    "peak_vram_allocated_mb",
    "val_latency_ms_per_query",
    "val_throughput_qps",
)


@dataclass(frozen=True)
class RunnerConfig:
    dataset: str = "cifar100"
    backbone: str = "dinov3_vits16"
    task_network_version: str = "role_v2"
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    baselines: tuple[str, ...] = BASELINES
    outputs_root: Path = Path("outputs")
    data_root: Path = Path("data")
    split_file: Path | None = None
    device: str = "auto"
    epochs: int | None = None
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    lr: float = 1e-3
    min_lr: float = 0.0
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    topk_candidates: int = 32
    selector_temperature: float = 0.1
    policy_weight: float = 1.0
    num_workers: int = 0
    eval_test: bool = True
    dry_run: bool = False
    pretrained_backbone: bool = True
    amp: bool = True
    resume: bool = True
    force: bool = False
    plan_only: bool = False

    def __post_init__(self) -> None:
        if self.task_network_version not in TASK_NETWORK_VERSIONS:
            raise ValueError(f"Unsupported task_network_version: {self.task_network_version}")

    @property
    def configured_epochs(self) -> int:
        if self.epochs is not None:
            return self.epochs
        return 100 if self.backbone == "dinov3_vits16" else 5

    @property
    def executed_epochs(self) -> int:
        return 1 if self.dry_run else self.configured_epochs

    @property
    def resolved_split_file(self) -> Path:
        if self.split_file is not None:
            return self.split_file
        return Path("data") / "splits" / f"{self.dataset}_seed42.json"

    @property
    def experiment_root(self) -> Path:
        root = self.outputs_root / self.dataset / self.backbone
        if self.backbone != "tiny" and self.task_network_version != "legacy_v1":
            root = root / self.task_network_version
        return root

    def seed_root(self, seed: int) -> Path:
        return self.experiment_root / f"seed_{seed}"

    def run_dir(self, seed: int, baseline: str) -> Path:
        return self.seed_root(seed) / baseline

    def standard_encoder_path(self, seed: int) -> Path:
        return self.run_dir(seed, "no_context") / "best_encoder.pt"


def ordered_baselines(baselines: Sequence[str]) -> tuple[str, ...]:
    """Return canonical order so the standard encoder is trained first."""
    requested = set(baselines)
    return tuple(name for name in BASELINES if name in requested)


def build_train_command(config: RunnerConfig, seed: int, baseline: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tacs.train",
        "--baseline",
        baseline,
        "--dataset",
        config.dataset,
        "--backbone",
        config.backbone,
        "--task-network-version",
        config.task_network_version,
        "--seed",
        str(seed),
        "--batch-size",
        str(config.batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--lr",
        str(config.lr),
        "--min-lr",
        str(config.min_lr),
        "--warmup-epochs",
        str(config.warmup_epochs),
        "--weight-decay",
        str(config.weight_decay),
        "--device",
        config.device,
        "--data-root",
        str(config.data_root),
        "--split-file",
        str(config.resolved_split_file),
        "--output-dir",
        str(config.run_dir(seed, baseline)),
        "--topk-candidates",
        str(config.topk_candidates),
        "--selector-temperature",
        str(config.selector_temperature),
        "--policy-weight",
        str(config.policy_weight),
        "--num-workers",
        str(config.num_workers),
    ]
    if config.epochs is not None:
        command.extend(("--epochs", str(config.epochs)))
    if baseline in DEPENDENT_BASELINES:
        command.extend(
            ("--standard-encoder", str(config.standard_encoder_path(seed)))
        )
    if config.eval_test:
        command.append("--eval-test")
    if config.dry_run:
        command.append("--dry-run")
    if not config.pretrained_backbone:
        command.append("--no-pretrained-backbone")
    if not config.amp:
        command.append("--no-amp")
    if not config.resume or config.force:
        command.append("--no-resume")
    return command


def load_metrics(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_complete_run(
    config: RunnerConfig,
    seed: int,
    baseline: str,
) -> bool:
    run_dir = config.run_dir(seed, baseline)
    payload = load_metrics(run_dir / "metrics.json")
    if payload is None:
        return False
    run_config = payload.get("config", {})
    if not isinstance(run_config, dict):
        return False
    run_config = dict(run_config)
    run_config.setdefault("task_network_version", "legacy_v1")
    recipe = payload.get("training_recipe", {})
    expected_config = {
        "dataset": config.dataset,
        "backbone": config.backbone,
        "task_network_version": config.task_network_version,
        "pretrained_backbone": config.pretrained_backbone,
        "seed": seed,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lr": config.lr,
        "min_lr": config.min_lr,
        "warmup_epochs": config.warmup_epochs,
        "weight_decay": config.weight_decay,
        "split_file": str(config.resolved_split_file),
        "topk_candidates": config.topk_candidates,
        "selector_temperature": config.selector_temperature,
        "policy_weight": config.policy_weight,
        "dry_run": config.dry_run,
        "num_workers": config.num_workers,
        "eval_test": config.eval_test,
        "amp": config.amp,
    }
    required_matches = (
        payload.get("metrics_schema_version") == METRICS_SCHEMA_VERSION,
        payload.get("baseline") == baseline,
        payload.get("seed") == seed,
        all(run_config.get(key) == value for key, value in expected_config.items()),
        recipe.get("executed_epochs") == config.executed_epochs,
        (not config.eval_test or payload.get("test_result") is not None),
        (run_dir / "checkpoint_best.pt").is_file(),
        (run_dir / "best_encoder.pt").is_file(),
    )
    return all(required_matches)


def _latest_history_value(payload: dict[str, Any], key: str) -> float | None:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        return None
    value = history[-1].get(key) if isinstance(history[-1], dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def metric_values(payload: dict[str, Any]) -> dict[str, float | None]:
    test_result = payload.get("test_result")
    memory = payload.get("memory") or {}
    return {
        "best_val_accuracy": _number_or_none(payload.get("best_val_acc")),
        "test_accuracy": _number_or_none(
            test_result.get("accuracy") if isinstance(test_result, dict) else None
        ),
        "end_to_end_time_s": _number_or_none(payload.get("end_to_end_time_s")),
        "peak_rss_mb": _number_or_none(memory.get("peak_rss_mb")),
        "peak_vram_allocated_mb": _number_or_none(
            memory.get("peak_vram_allocated_mb")
        ),
        "val_latency_ms_per_query": _latest_history_value(
            payload, "val_latency_ms_per_query"
        ),
        "val_throughput_qps": _latest_history_value(
            payload, "val_throughput_qps"
        ),
    }


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def aggregate_metrics(
    config: RunnerConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    by_baseline: dict[str, list[dict[str, float | None]]] = {
        baseline: [] for baseline in ordered_baselines(config.baselines)
    }
    for seed in config.seeds:
        for baseline in ordered_baselines(config.baselines):
            metrics_path = config.run_dir(seed, baseline) / "metrics.json"
            payload = load_metrics(metrics_path)
            if payload is None:
                continue
            run_config = payload.get("config", {})
            if not isinstance(run_config, dict) or run_config.get(
                "task_network_version", "legacy_v1"
            ) != config.task_network_version:
                raise RuntimeError(f"Task network version mismatch in {metrics_path}")
            values = metric_values(payload)
            runs.append(
                {
                    "seed": seed,
                    "baseline": baseline,
                    "metrics_path": str(metrics_path),
                    **values,
                }
            )
            by_baseline[baseline].append(values)

    summaries: list[dict[str, Any]] = []
    for baseline, baseline_runs in by_baseline.items():
        row: dict[str, Any] = {
            "baseline": baseline,
            "num_runs": len(baseline_runs),
        }
        for metric in SUMMARY_METRICS:
            values = [
                value
                for run in baseline_runs
                if (value := run[metric]) is not None
            ]
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            )
        summaries.append(row)

    document = {
        "summary_schema_version": 1,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "std_definition": "sample standard deviation (ddof=1); 0 for one run",
        "experiment": {
            "dataset": config.dataset,
            "backbone": config.backbone,
            "task_network_version": config.task_network_version,
            "seeds": list(config.seeds),
            "baselines": list(ordered_baselines(config.baselines)),
            "split_file": str(config.resolved_split_file),
            "configured_epochs": config.configured_epochs,
            "executed_epochs": config.executed_epochs,
            "eval_test": config.eval_test,
            "dry_run": config.dry_run,
        },
        "runs": runs,
        "baselines": summaries,
    }
    return document, summaries


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}"


def write_summaries(config: RunnerConfig) -> tuple[Path, Path, Path]:
    document, rows = aggregate_metrics(config)
    root = config.experiment_root
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "multiseed_summary.json"
    csv_path = root / "multiseed_summary.csv"
    markdown_path = root / "multiseed_summary.md"
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_fields = ["baseline", "num_runs"] + [
        field
        for metric in SUMMARY_METRICS
        for field in (f"{metric}_n", f"{metric}_mean", f"{metric}_std")
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    columns = (
        ("Validation accuracy", "best_val_accuracy"),
        ("Test accuracy", "test_accuracy"),
        ("Time (s)", "end_to_end_time_s"),
        ("Peak RSS (MB)", "peak_rss_mb"),
        ("Peak VRAM (MB)", "peak_vram_allocated_mb"),
        ("Val latency (ms/query)", "val_latency_ms_per_query"),
        ("Val throughput (q/s)", "val_throughput_qps"),
    )
    lines = [
        "# Multi-seed experiment summary",
        "",
        f"Dataset: `{config.dataset}` · Backbone: `{config.backbone}` · "
        f"Seeds: `{', '.join(map(str, config.seeds))}`",
        "",
        "Values are mean ± sample standard deviation across available seeds.",
        "",
        "| Baseline | Runs | " + " | ".join(label for label, _ in columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    for row in rows:
        cells = [
            _format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
            for _, metric in columns
        ]
        lines.append(
            f"| {row['baseline']} | {row['num_runs']} | " + " | ".join(cells) + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def _require_standard_encoder(config: RunnerConfig, seed: int) -> None:
    if config.standard_encoder_path(seed).is_file():
        return
    raise RuntimeError(
        f"Seed {seed} cần standard encoder tại {config.standard_encoder_path(seed)}. "
        "Hãy thêm baseline no_context hoặc chạy no_context cho seed này trước."
    )


def run_all(config: RunnerConfig) -> tuple[Path, Path, Path] | None:
    baselines = ordered_baselines(config.baselines)
    for seed in config.seeds:
        for baseline in baselines:
            if (
                not config.plan_only
                and baseline in DEPENDENT_BASELINES
                and "no_context" not in baselines
            ):
                _require_standard_encoder(config, seed)
            command = build_train_command(config, seed, baseline)
            label = f"seed={seed} baseline={baseline}"
            if not config.force and is_complete_run(config, seed, baseline):
                print(f"[SKIP] {label}: complete run already exists")
                continue
            print(f"[RUN]  {label}")
            print("       " + subprocess.list2cmdline(command))
            if not config.plan_only:
                config.run_dir(seed, baseline).mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)
    if config.plan_only:
        return None
    paths = write_summaries(config)
    print("[DONE] Summary:")
    for path in paths:
        print(f"       {path}")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Chạy 5 mô hình TACS trên nhiều seed và tổng hợp mean ± std."
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument(
        "--backbone", choices=("tiny", "dinov3_vits16"), default="dinov3_vits16"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--task-network-version", choices=TASK_NETWORK_VERSIONS, default="role_v2")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--topk-candidates", type=int, default=32)
    parser.add_argument("--selector-temperature", type=float, default=0.1)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-eval-test", action="store_true", help="Không đánh giá test set"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Tắt CUDA AMP")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không tiếp tục checkpoint của run đang dở",
    )
    parser.add_argument("--force", action="store_true", help="Chạy lại cả run hoàn chỉnh")
    parser.add_argument("--plan-only", action="store_true", help="Chỉ in lệnh sẽ chạy")
    args = parser.parse_args(argv)

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds không được chứa giá trị trùng")
    if len(set(args.baselines)) != len(args.baselines):
        parser.error("--baselines không được chứa giá trị trùng")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs phải lớn hơn 0")
    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps phải lớn hơn 0")
    resolved_epochs = args.epochs or (100 if args.backbone == "dinov3_vits16" else 5)
    if args.warmup_epochs < 0 or args.warmup_epochs >= resolved_epochs:
        parser.error("--warmup-epochs phải thỏa 0 <= warmup-epochs < epochs")
    if args.lr <= 0:
        parser.error("--lr phải lớn hơn 0")
    if args.min_lr < 0 or args.min_lr >= args.lr:
        parser.error("--min-lr phải thỏa 0 <= min-lr < lr")
    if args.weight_decay < 0 or args.policy_weight < 0:
        parser.error("weight decay và policy weight không được âm")
    if args.topk_candidates <= 0 or args.selector_temperature <= 0:
        parser.error("top-k và selector temperature phải lớn hơn 0")
    if args.num_workers < 0:
        parser.error("--num-workers không được âm")

    return RunnerConfig(
        dataset=args.dataset,
        backbone=args.backbone,
        task_network_version=args.task_network_version,
        seeds=tuple(args.seeds),
        baselines=tuple(args.baselines),
        outputs_root=args.outputs_root,
        data_root=args.data_root,
        split_file=args.split_file,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        topk_candidates=args.topk_candidates,
        selector_temperature=args.selector_temperature,
        policy_weight=args.policy_weight,
        num_workers=args.num_workers,
        eval_test=not args.no_eval_test,
        dry_run=args.dry_run,
        pretrained_backbone=not args.no_pretrained_backbone,
        amp=not args.no_amp,
        resume=not args.no_resume,
        force=args.force,
        plan_only=args.plan_only,
    )


def main() -> None:
    configure_stdout()
    run_all(parse_args())


if __name__ == "__main__":
    main()
