"""Validate and summarize the five-model CIFAR-10 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


BASELINES = ("no_context", "random", "similarity", "tacs", "two_stage")
DISPLAY_NAMES = {
    "no_context": "No-context",
    "random": "Random-context",
    "similarity": "Similarity-context",
    "tacs": "TACS Full",
    "two_stage": "Two-Stage TACS",
}
EXPECTED_CIFAR10_SIZES = {
    "query_train": 36_000,
    "candidate_pool": 9_000,
    "validation": 5_000,
    "test": 10_000,
}


class BenchmarkValidationError(ValueError):
    """Raised when experiment artifacts cannot form a fair comparison."""


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _require(payload: dict[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise BenchmarkValidationError(f"{context} thiếu trường {key!r}")
    return payload[key]


def load_metric_file(path: str | Path) -> dict[str, Any]:
    metric_path = Path(path)
    if not metric_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metrics: {metric_path}")
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkValidationError(f"Metrics không phải JSON object: {metric_path}")
    return payload


def validate_metric(
    payload: dict[str, Any],
    expected_baseline: str,
    require_complete: bool = True,
) -> None:
    context = f"metrics[{expected_baseline}]"
    if payload.get("metrics_schema_version") != 2:
        raise BenchmarkValidationError(
            f"{context} không dùng metrics schema v2; cần chạy lại bằng trainer hiện tại"
        )
    if payload.get("baseline") != expected_baseline:
        raise BenchmarkValidationError(
            f"{context} có baseline={payload.get('baseline')!r}"
        )

    data = _require(payload, "data", context)
    if data.get("source") != "cifar10":
        raise BenchmarkValidationError(f"{context} không có provenance CIFAR-10")
    if not data.get("split_sha256"):
        raise BenchmarkValidationError(f"{context} thiếu split_sha256")

    full_sizes = data.get("full_sizes")
    effective_sizes = data.get("effective_sizes")
    if require_complete and (
        full_sizes != EXPECTED_CIFAR10_SIZES
        or effective_sizes != EXPECTED_CIFAR10_SIZES
    ):
        raise BenchmarkValidationError(
            f"{context} dùng dry-run/tập con: effective_sizes={effective_sizes}"
        )

    history = _require(payload, "history", context)
    if not isinstance(history, list) or not history:
        raise BenchmarkValidationError(f"{context} không có history")
    required_history_keys = {
        "val_acc",
        "val_latency_ms_per_query",
        "val_throughput_qps",
        "train_peak_rss_mb",
        "val_peak_rss_mb",
    }
    missing_history = required_history_keys.difference(history[-1])
    if missing_history:
        raise BenchmarkValidationError(
            f"{context} thiếu metrics benchmark: {sorted(missing_history)}"
        )

    memory = _require(payload, "memory", context)
    if "peak_rss_mb" not in memory or "peak_vram_allocated_mb" not in memory:
        raise BenchmarkValidationError(f"{context} thiếu peak RAM/VRAM")

    config = _require(payload, "config", context)
    if config.get("baseline") != expected_baseline:
        raise BenchmarkValidationError(f"{context} config baseline không khớp")
    runtime = _require(payload, "runtime", context)
    if not runtime.get("device") or not runtime.get("torch_version"):
        raise BenchmarkValidationError(f"{context} thiếu thông tin runtime/device")

    selection = _require(payload, "selection", context)
    candidate_pool_size = data.get("effective_sizes", {}).get("candidate_pool")
    if expected_baseline == "tacs":
        if (
            selection.get("strategy") != "full_pool_tacs"
            or selection.get("selector_candidates") != candidate_pool_size
        ):
            raise BenchmarkValidationError(f"{context} không phải TACS Full toàn pool")
    elif expected_baseline == "two_stage":
        selector_candidates = selection.get("selector_candidates", 0)
        if (
            selection.get("strategy") != "cosine_topk_then_tacs"
            or selection.get("retrieval_candidates_scanned") != candidate_pool_size
            or not 0 < selector_candidates <= candidate_pool_size
        ):
            raise BenchmarkValidationError(f"{context} không phải Two-Stage hợp lệ")

    if require_complete and not payload.get("test_result"):
        raise BenchmarkValidationError(
            f"{context} chưa có test_result; phải train/evaluate với --eval-test"
        )


def validate_comparison(
    metrics: dict[str, dict[str, Any]],
    require_complete: bool = True,
) -> None:
    missing = set(BASELINES).difference(metrics)
    if missing:
        raise BenchmarkValidationError(f"Thiếu mô hình: {sorted(missing)}")

    for baseline in BASELINES:
        validate_metric(metrics[baseline], baseline, require_complete=require_complete)

    comparable_fields = {
        "seed": lambda item: item["seed"],
        "epochs": lambda item: item["epochs"],
        "split_sha256": lambda item: item["data"]["split_sha256"],
        "batch_size": lambda item: item["config"]["batch_size"],
        "embedding_dim": lambda item: item["config"]["embedding_dim"],
        "num_classes": lambda item: item["config"]["num_classes"],
        "lr": lambda item: item["config"]["lr"],
        "weight_decay": lambda item: item["config"]["weight_decay"],
        "device": lambda item: item["runtime"]["device"],
        "torch_version": lambda item: item["runtime"]["torch_version"],
        "torch_num_threads": lambda item: item["runtime"]["torch_num_threads"],
        "gpu_name": lambda item: item["runtime"]["gpu_name"],
    }
    for field, getter in comparable_fields.items():
        values = {getter(metrics[baseline]) for baseline in BASELINES}
        if len(values) != 1:
            raise BenchmarkValidationError(
                f"So sánh không công bằng: {field} khác nhau: {sorted(values, key=str)}"
            )

    tacs_objective = {
        (
            metrics[baseline]["selector_temperature"],
            metrics[baseline]["policy_weight"],
        )
        for baseline in ("tacs", "two_stage")
    }
    if len(tacs_objective) != 1:
        raise BenchmarkValidationError(
            "So sánh không công bằng: TACS Full và Two-Stage khác temperature/policy weight"
        )


def _average(history: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(record[key]) for record in history), 4)


def summarize_metric(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload["history"]
    selection = payload["selection"]
    test_result = payload.get("test_result") or {}
    memory = payload["memory"]
    config = payload["config"]
    runtime = payload["runtime"]
    return {
        "baseline": payload["baseline"],
        "model": DISPLAY_NAMES[payload["baseline"]],
        "seed": payload["seed"],
        "epochs": payload["epochs"],
        "batch_size": config["batch_size"],
        "embedding_dim": config["embedding_dim"],
        "device": runtime["device"],
        "gpu_name": runtime["gpu_name"],
        "torch_version": runtime["torch_version"],
        "best_val_accuracy": payload["best_val_acc"],
        "test_accuracy": test_result.get("accuracy"),
        "avg_train_throughput_qps": _average(history, "throughput_qps"),
        "avg_val_throughput_qps": _average(history, "val_throughput_qps"),
        "avg_val_latency_ms_per_query": _average(
            history, "val_latency_ms_per_query"
        ),
        "data_setup_time_s": payload["data_setup_time_s"],
        "training_time_s": payload["total_time_s"],
        "end_to_end_time_s": payload["end_to_end_time_s"],
        "peak_rss_mb": memory["peak_rss_mb"],
        "peak_vram_allocated_mb": memory["peak_vram_allocated_mb"],
        "peak_vram_reserved_mb": memory["peak_vram_reserved_mb"],
        "selection_strategy": selection["strategy"],
        "retrieval_candidates_scanned": selection[
            "retrieval_candidates_scanned"
        ],
        "selector_candidates": selection["selector_candidates"],
        "split_sha256": payload["data"]["split_sha256"],
    }


def build_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [summarize_metric(metrics[baseline]) for baseline in BASELINES]


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Mô hình | Val Acc | Test Acc | Latency (ms/query) | Throughput (q/s) | Peak RAM (MB) | Peak VRAM (MB) | Retrieval / Selector |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test_accuracy = row["test_accuracy"]
        test_text = "N/A" if test_accuracy is None else f"{test_accuracy * 100:.2f}%"
        vram = row["peak_vram_allocated_mb"]
        vram_text = "N/A" if vram is None else f"{vram:.2f}"
        lines.append(
            f"| {row['model']} | {row['best_val_accuracy'] * 100:.2f}% | "
            f"{test_text} | {row['avg_val_latency_ms_per_query']:.4f} | "
            f"{row['avg_val_throughput_qps']:.2f} | {row['peak_rss_mb']:.2f} | "
            f"{vram_text} | {row['retrieval_candidates_scanned']} / "
            f"{row['selector_candidates']} |"
        )
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_table(rows) + "\n", encoding="utf-8")


def plot_comparison(rows: list[dict[str, Any]], path: str | Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Thiếu matplotlib; cài project với extra [viz] hoặc dùng --skip-plot"
        ) from error

    labels = [row["model"] for row in rows]
    accuracy = [
        100
        * (
            row["test_accuracy"]
            if row["test_accuracy"] is not None
            else row["best_val_accuracy"]
        )
        for row in rows
    ]
    latency = [row["avg_val_latency_ms_per_query"] for row in rows]
    use_vram = any(row["peak_vram_allocated_mb"] is not None for row in rows)
    memory = [
        (
            row["peak_vram_allocated_mb"]
            if use_vram and row["peak_vram_allocated_mb"] is not None
            else row["peak_rss_mb"]
        )
        for row in rows
    ]

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = (
        (accuracy, "Accuracy (%)", "Test Accuracy"),
        (latency, "ms/query", "End-to-end Validation Latency"),
        (memory, "MB", "Peak VRAM" if use_vram else "Peak RAM"),
    )
    for axis, (values, ylabel, title) in zip(axes, panels):
        axis.bar(labels, values, color=("#4472C4", "#70AD47", "#ED7D31", "#A5A5A5", "#5B9BD5"))
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", linestyle=":", alpha=0.5)

    figure.suptitle("CIFAR-10: So sánh 5 mô hình TACS", fontweight="bold")
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Cho phép dry-run/thiếu test_result; chỉ dùng để kiểm tra pipeline",
    )
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    root = Path(args.outputs_root)
    metrics = {
        baseline: load_metric_file(root / baseline / "metrics.json")
        for baseline in BASELINES
    }
    validate_comparison(metrics, require_complete=not args.allow_incomplete)
    rows = build_rows(metrics)

    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "all_models_benchmark.csv")
    write_json(rows, output_dir / "all_models_benchmark.json")
    write_markdown(rows, output_dir / "all_models_benchmark.md")
    if not args.skip_plot:
        plot_comparison(rows, output_dir / "all_models_comparison.png")

    print(markdown_table(rows))
    print(f"Đã lưu benchmark tại: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
