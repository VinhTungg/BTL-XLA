"""Hạ tầng huấn luyện và đánh giá dùng chung cho cả 3 baseline tuần 2.

Hỗ trợ 3 baseline:
1. no_context: Chỉ dùng query embedding, context vector = 0.
2. random: Lấy 1 candidate ngẫu nhiên từ candidate pool (tái lập theo seed).
3. similarity: Lấy 1 candidate có cosine similarity cao nhất từ candidate pool qua encoder chuẩn đã khóa.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .baselines import BaselineModel
from .data import ContextDataset, build_cifar10
from .device import configure_stdout, resolve_device
from .model import TinyVisionEncoder
from .retrieval import (
    CosineSimilarityRetriever,
    SimilarityContextDataset,
    precompute_candidate_embeddings,
)


@dataclass
class ExperimentConfig:
    """Cấu hình thí nghiệm tuần 2."""

    baseline: str = "no_context"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 64
    num_classes: int = 10
    device: str = "auto"
    data_root: str = "data"
    split_file: str = "data/splits/cifar10_seed42.json"
    output_dir: str = "outputs/no_context"
    standard_encoder_path: str = "outputs/no_context/best_encoder.pt"
    dry_run: bool = False
    num_workers: int = 0
    eval_test: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)


def get_current_rss_mb() -> float:
    """Đo dung lượng RAM Resident Set Size (RSS) đang chiếm dụng theo MB."""
    process = psutil.Process()
    return round(process.memory_info().rss / (1024 * 1024), 2)


def create_mock_datasets(
    num_train: int = 200,
    num_candidate: int = 50,
    num_val: int = 50,
    num_test: int = 50,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """Tạo mock dataset dùng cho unit test hoặc dry-run khi chưa tải xong CIFAR-10 thật."""
    generator = torch.Generator().manual_seed(seed)
    total = num_train + num_candidate + num_val + num_test
    images = torch.randn(total, 3, 32, 32, generator=generator)
    targets = torch.randint(0, num_classes, (total,), generator=generator)
    full = TensorDataset(images, targets)

    train_set = Subset(full, list(range(0, num_train)))
    candidate_set = Subset(full, list(range(num_train, num_train + num_candidate)))
    val_set = Subset(full, list(range(num_train + num_candidate, num_train + num_candidate + num_val)))
    test_set = Subset(full, list(range(num_train + num_candidate + num_val, total)))
    return train_set, candidate_set, val_set, test_set


def build_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader | None, Dataset]:
    """Khởi tạo DataLoader tương ứng với từng baseline."""
    torch.manual_seed(config.seed)

    # Thử nạp CIFAR-10 thật
    use_mock = False
    try:
        datasets = build_cifar10(root=config.data_root, split_file=config.split_file, download=False)
        query_train = datasets.query_train
        candidate_pool = datasets.candidate_pool
        validation = datasets.validation
        test_set = datasets.test
    except Exception:
        use_mock = True
        n_tr = 400 if not config.dry_run else 16
        n_ca = 100 if not config.dry_run else 8
        n_va = 100 if not config.dry_run else 8
        n_te = 100 if not config.dry_run else 8
        query_train, candidate_pool, validation, test_set = create_mock_datasets(
            num_train=n_tr, num_candidate=n_ca, num_val=n_va, num_test=n_te, seed=config.seed
        )

    # Giới hạn số mẫu cho dry-run hoặc config (cho fast testing)
    if config.dry_run:
        max_tr = config.max_train_samples or 16
        max_va = config.max_val_samples or 8
        query_train = Subset(query_train, list(range(min(len(query_train), max_tr))))
        validation = Subset(validation, list(range(min(len(validation), max_va))))
    else:
        if config.max_train_samples and len(query_train) > config.max_train_samples:
            query_train = Subset(query_train, list(range(config.max_train_samples)))
        if config.max_val_samples and len(validation) > config.max_val_samples:
            validation = Subset(validation, list(range(config.max_val_samples)))


    # Tạo train/val dataset theo baseline
    if config.baseline == "no_context":
        train_ds = query_train
        val_ds = validation
        test_ds = test_set

    elif config.baseline == "random":
        train_ds = ContextDataset(
            query_dataset=query_train,
            candidate_dataset=candidate_pool,
            num_candidates=1,
            seed=config.seed,
        )
        val_ds = ContextDataset(
            query_dataset=validation,
            candidate_dataset=candidate_pool,
            num_candidates=1,
            seed=config.seed,
        )
        test_ds = ContextDataset(
            query_dataset=test_set,
            candidate_dataset=candidate_pool,
            num_candidates=1,
            seed=config.seed,
        )

    elif config.baseline == "similarity":
        # Khóa encoder chuẩn từ Gói 2
        encoder = TinyVisionEncoder(config.embedding_dim).to(device)
        encoder_path = Path(config.standard_encoder_path)
        if encoder_path.exists():
            state = torch.load(encoder_path, map_location=device, weights_only=True)
            encoder.load_state_dict(state)
        encoder.eval()

        # Cache candidate pool embeddings
        cache_dir = Path(config.output_dir).parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "candidate_pool_embeddings.pt"

        if not cache_file.exists():
            precompute_candidate_embeddings(
                candidate_dataset=candidate_pool,
                encoder=encoder,
                batch_size=config.batch_size,
                device=device,
                output_path=cache_file,
            )

        retriever = CosineSimilarityRetriever.from_cache(cache_file, candidate_dataset=candidate_pool)

        train_ds = SimilarityContextDataset(
            query_dataset=query_train,
            candidate_dataset=candidate_pool,
            retriever=retriever,
            encoder=encoder,
            device=device,
            batch_size=config.batch_size,
        )
        val_ds = SimilarityContextDataset(
            query_dataset=validation,
            candidate_dataset=candidate_pool,
            retriever=retriever,
            encoder=encoder,
            device=device,
            batch_size=config.batch_size,
        )
        test_ds = SimilarityContextDataset(
            query_dataset=test_set,
            candidate_dataset=candidate_pool,
            retriever=retriever,
            encoder=encoder,
            device=device,
            batch_size=config.batch_size,
        )
    else:
        raise ValueError(f"Unknown baseline: {config.baseline}")

    bs = config.batch_size if not config.dry_run else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        if config.eval_test
        else None
    )

    return train_loader, val_loader, test_loader, candidate_pool


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Giải nén batch thống nhất cho cả dạng tuple và dictionary."""
    if isinstance(batch, dict):
        query = batch["query"]
        target = batch["target"]
        candidates = batch.get("candidates")
        return query, target, candidates
    elif isinstance(batch, (list, tuple)):
        query, target = batch[0], batch[1]
        return query, target, None
    raise TypeError(f"Không hỗ trợ dạng batch: {type(batch)}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Huấn luyện 1 epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.perf_counter()

    for batch in loader:
        query, target, candidates = unpack_batch(batch)
        query = query.to(device)
        target = target.to(device)
        if candidates is not None:
            candidates = candidates.to(device)

        optimizer.zero_grad()
        logits = model(query, candidates)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * query.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == target).sum().item()
        total += query.size(0)

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    throughput = total / max(1e-5, elapsed)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "throughput": round(throughput, 2),
        "time": round(elapsed, 2),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Đánh giá mô hình (không cập nhật gradient)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            query, target, candidates = unpack_batch(batch)
            query = query.to(device)
            target = target.to(device)
            if candidates is not None:
                candidates = candidates.to(device)

            logits = model(query, candidates)
            loss = criterion(logits, target)

            total_loss += loss.item() * query.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += query.size(0)

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    throughput = total / max(1e-5, elapsed)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "throughput": round(throughput, 2),
        "time": round(elapsed, 2),
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Chạy toàn bộ quy trình huấn luyện và đánh giá cho 1 baseline."""
    configure_stdout()
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")

    train_loader, val_loader, test_loader, candidate_pool = build_dataloaders(config, device)

    model = BaselineModel(
        num_classes=config.num_classes,
        embedding_dim=config.embedding_dim,
        baseline_type=config.baseline,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history: list[dict[str, Any]] = []
    total_train_start = time.perf_counter()

    epochs = config.epochs if not config.dry_run else 1
    print(f"\n[{config.baseline.upper()}] Starting training {epochs} epochs on device '{device}'...")


    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_res = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate(model, val_loader, criterion, device)
        rss_mb = get_current_rss_mb()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_acc": train_res["accuracy"],
            "val_loss": val_res["loss"],
            "val_acc": val_res["accuracy"],
            "throughput_qps": train_res["throughput"],
            "epoch_time_s": train_res["time"],
            "rss_mb": rss_mb,
        }
        history.append(epoch_record)

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_res['loss']:.4f}, Acc: {train_res['accuracy'] * 100:.2f}% | "
            f"Val Loss: {val_res['loss']:.4f}, Acc: {val_res['accuracy'] * 100:.2f}% | "
            f"{train_res['throughput']} q/s | RAM: {rss_mb} MB"
        )

        # Lưu checkpoint tốt nhất
        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "val_loss": val_res["loss"],
                "config": asdict(config),
            }
            torch.save(checkpoint_data, out_dir / "checkpoint_best.pt")

            # Lưu encoder chuẩn (đặc biệt quan trọng ở No-context cho Gói 2)
            torch.save(model.encoder.state_dict(), out_dir / "best_encoder.pt")

    total_time = round(time.perf_counter() - total_train_start, 2)

    # Đánh giá test set (nếu yêu cầu)
    test_result = None
    if test_loader is not None:
        best_ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_result = evaluate(model, test_loader, criterion, device)
        print(f"  --> [TEST EVALUATION] Loss: {test_result['loss']:.4f}, Acc: {test_result['accuracy'] * 100:.2f}%")

    metrics_payload = {
        "baseline": config.baseline,
        "seed": config.seed,
        "epochs": epochs,
        "best_val_acc": round(best_val_acc, 4),
        "total_time_s": total_time,
        "test_result": test_result,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[{config.baseline.upper()}] Done. Output saved to: {out_dir}\n")
    return metrics_payload


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="CLI Huấn luyện 3 Baseline Tuần 2.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="no_context",
        choices=["no_context", "random", "similarity"],
        help="Loại baseline cần chạy",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Số epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", type=str, default="data", help="Đường dẫn CIFAR-10")
    parser.add_argument("--split-file", type=str, default="data/splits/cifar10_seed42.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất kết quả")
    parser.add_argument("--standard-encoder", type=str, default="outputs/no_context/best_encoder.pt")
    parser.add_argument("--dry-run", action="store_true", help="Chạy kiểm thử ngắn 1 epoch batch nhỏ")
    parser.add_argument("--eval-test", action="store_true", help="Đánh giá test set sau khi hoàn thành")

    args = parser.parse_args()
    out = args.output_dir if args.output_dir else f"outputs/{args.baseline}"
    return ExperimentConfig(
        baseline=args.baseline,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        split_file=args.split_file,
        output_dir=out,
        standard_encoder_path=args.standard_encoder,
        dry_run=args.dry_run,
        eval_test=args.eval_test,
    )


def main() -> None:
    configure_stdout()
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
