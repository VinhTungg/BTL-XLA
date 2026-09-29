"""Phân tích, tổng hợp và trực quan hóa kết quả 3 baseline Tuần 2 (Gói 5)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

def configure_stdout():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")



def load_metrics(output_dir: str | Path) -> dict:
    p = Path(output_dir) / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def generate_comparison_table(
    no_ctx: dict,
    random_ctx: dict,
    sim_ctx: dict,
) -> str:
    """Tạo bảng so sánh Markdown tổng hợp 3 baseline."""
    lines = [
        "| Chỉ số thực nghiệm | Baseline 1 (No-context) | Baseline 2 (Random-context) | Baseline 3 (Similarity-context) |",
        "|:---|:---:|:---:|:---:|",
    ]

    # Best val acc
    lines.append(
        f"| **Validation Accuracy (Best)** | **{no_ctx['best_val_acc'] * 100:.2f}%** | **{random_ctx['best_val_acc'] * 100:.2f}%** | **{sim_ctx['best_val_acc'] * 100:.2f}%** |"
    )

    # Test acc
    no_test = no_ctx.get("test_result", {}).get("accuracy", 0.0) if no_ctx.get("test_result") else 0.0
    rand_test = random_ctx.get("test_result", {}).get("accuracy", 0.0) if random_ctx.get("test_result") else 0.0
    sim_test = sim_ctx.get("test_result", {}).get("accuracy", 0.0) if sim_ctx.get("test_result") else 0.0
    lines.append(
        f"| **Test Accuracy (Độc lập)** | **{no_test * 100:.2f}%** | **{rand_test * 100:.2f}%** | **{sim_test * 100:.2f}%** |"
    )

    # Final Train & Val Loss
    no_last = no_ctx["history"][-1]
    rand_last = random_ctx["history"][-1]
    sim_last = sim_ctx["history"][-1]
    lines.append(
        f"| **Final Train Loss** | {no_last['train_loss']:.4f} | {rand_last['train_loss']:.4f} | {sim_last['train_loss']:.4f} |"
    )
    lines.append(
        f"| **Final Val Loss** | {no_last['val_loss']:.4f} | {rand_last['val_loss']:.4f} | {sim_last['val_loss']:.4f} |"
    )

    # Average Throughput
    no_thru = sum(h["throughput_qps"] for h in no_ctx["history"]) / len(no_ctx["history"])
    rand_thru = sum(h["throughput_qps"] for h in random_ctx["history"]) / len(random_ctx["history"])
    sim_thru = sum(h["throughput_qps"] for h in sim_ctx["history"]) / len(sim_ctx["history"])
    lines.append(
        f"| **Throughput trung bình (query/s)** | {no_thru:.1f} q/s | {rand_thru:.1f} q/s | {sim_thru:.1f} q/s |"
    )

    # Total Training Time
    lines.append(
        f"| **Tổng thời gian huấn luyện** | {no_ctx['total_time_s']:.2f}s | {random_ctx['total_time_s']:.2f}s | {sim_ctx['total_time_s']:.2f}s |"
    )

    # RAM Peak
    no_ram = max(h["rss_mb"] for h in no_ctx["history"])
    rand_ram = max(h["rss_mb"] for h in random_ctx["history"])
    sim_ram = max(h["rss_mb"] for h in sim_ctx["history"])
    lines.append(
        f"| **Dung lượng RAM chiếm dụng** | {no_ram:.1f} MB | {rand_ram:.1f} MB | {sim_ram:.1f} MB |"
    )

    return "\n".join(lines)


def plot_curves(
    no_ctx: dict,
    random_ctx: dict,
    sim_ctx: dict,
    output_path: str | Path = "docs/baseline_comparison_curves.png",
) -> None:
    """Vẽ biểu đồ so sánh Loss và Accuracy của 3 baseline."""
    epochs_no = [h["epoch"] for h in no_ctx["history"]]
    epochs_rand = [h["epoch"] for h in random_ctx["history"]]
    epochs_sim = [h["epoch"] for h in sim_ctx["history"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Train Loss & Val Loss
    ax_loss = axes[0]
    ax_loss.plot(epochs_no, [h["train_loss"] for h in no_ctx["history"]], "b--", label="No-ctx (Train)")
    ax_loss.plot(epochs_no, [h["val_loss"] for h in no_ctx["history"]], "b-o", label="No-ctx (Val)")

    ax_loss.plot(epochs_rand, [h["train_loss"] for h in random_ctx["history"]], "g--", label="Random (Train)")
    ax_loss.plot(epochs_rand, [h["val_loss"] for h in random_ctx["history"]], "g-s", label="Random (Val)")

    ax_loss.plot(epochs_sim, [h["train_loss"] for h in sim_ctx["history"]], "r--", label="Similarity (Train)")
    ax_loss.plot(epochs_sim, [h["val_loss"] for h in sim_ctx["history"]], "r-^", label="Similarity (Val)")

    ax_loss.set_title("Cross-Entropy Loss qua các Epochs", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=9)

    # Subplot 2: Train Acc & Val Acc
    ax_acc = axes[1]
    ax_acc.plot(epochs_no, [h["val_acc"] * 100 for h in no_ctx["history"]], "b-o", label="No-ctx Val Acc")
    ax_acc.plot(epochs_rand, [h["val_acc"] * 100 for h in random_ctx["history"]], "g-s", label="Random Val Acc")
    ax_acc.plot(epochs_sim, [h["val_acc"] * 100 for h in sim_ctx["history"]], "r-^", label="Similarity Val Acc")

    ax_acc.set_title("Validation Accuracy (%) qua các Epochs", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=10)

    fig.suptitle("So sánh thực nghiệm 3 Baseline (M2 - Tuần 2)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Đã lưu biểu đồ so sánh tại: {out.resolve()}")


def main() -> None:
    configure_stdout()
    no_ctx = load_metrics("outputs/no_context")
    rand_ctx = load_metrics("outputs/random")
    sim_ctx = load_metrics("outputs/similarity")

    table_md = generate_comparison_table(no_ctx, rand_ctx, sim_ctx)
    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP SO SÁNH 3 BASELINE TUẦN 2 (M2)")
    print("=" * 70)
    print(table_md)
    print("=" * 70 + "\n")

    plot_curves(no_ctx, rand_ctx, sim_ctx, "docs/baseline_comparison_curves.png")


if __name__ == "__main__":
    main()
