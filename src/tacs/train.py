"""Hạ tầng huấn luyện và đánh giá dùng chung cho cả 3 baseline tuần 2.

Hỗ trợ 3 baseline:
1. no_context: Chỉ dùng query embedding, context vector = 0.
2. random: Lấy 1 candidate ngẫu nhiên từ candidate pool (tái lập theo seed).
3. similarity: Lấy 1 candidate có cosine similarity cao nhất từ candidate pool qua encoder chuẩn đã khóa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .baselines import BaselineModel
from .data import ContextDataset, FullPoolContextDataset, build_cifar10
from .device import configure_stdout, resolve_device
from .metrics import PeakMemoryMonitor
from .model import TACSModel, TinyVisionEncoder
from .retrieval import (
    CosineSimilarityRetriever,
    SimilarityContextDataset,
    TopKContextDataset,
    precompute_candidate_embeddings,
)


ENCODER_CHECKPOINT_SCHEMA_VERSION = 1
EMBEDDING_CACHE_SCHEMA_VERSION = 1


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
    topk_candidates: int = 32
    selector_temperature: float = 0.1
    policy_weight: float = 1.0
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_indices_sha256(candidate_dataset: Dataset) -> str:
    indices = (
        candidate_dataset.indices
        if isinstance(candidate_dataset, Subset)
        else range(len(candidate_dataset))
    )
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _load_standard_encoder(
    config: ExperimentConfig,
    device: torch.device,
    expected_split_sha256: str | None = None,
) -> tuple[TinyVisionEncoder, Path, str]:
    encoder_path = Path(config.standard_encoder_path)
    if not encoder_path.is_file():
        raise RuntimeError(
            "Similarity/Two-Stage yêu cầu standard encoder đã huấn luyện trên "
            f"CIFAR-10, nhưng không tìm thấy: {encoder_path}"
        )

    payload = torch.load(encoder_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not {
        "encoder_state_dict",
        "metadata",
    }.issubset(payload):
        raise RuntimeError(
            f"Standard encoder {encoder_path} là checkpoint legacy/không có provenance. "
            "Hãy train lại baseline no_context bằng trainer hiện tại."
        )

    metadata = payload["metadata"]
    data_metadata = metadata.get("data", {}) if isinstance(metadata, dict) else {}
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != ENCODER_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("baseline") != "no_context"
        or metadata.get("embedding_dim") != config.embedding_dim
        or data_metadata.get("source") != "cifar10"
        or (
            expected_split_sha256 is not None
            and data_metadata.get("split_sha256") != expected_split_sha256
        )
    ):
        raise RuntimeError(
            f"Standard encoder {encoder_path} không khớp schema, baseline, "
            "embedding_dim hoặc provenance CIFAR-10."
        )

    full_sizes = data_metadata.get("full_sizes", {})
    effective_sizes = data_metadata.get("effective_sizes", {})
    if not config.dry_run and (
        effective_sizes.get("query_train") != full_sizes.get("query_train")
        or effective_sizes.get("validation") != full_sizes.get("validation")
    ):
        raise RuntimeError(
            f"Standard encoder {encoder_path} được tạo bởi dry-run/tập con. "
            "Hãy train no_context trên toàn bộ query_train và validation."
        )

    encoder = TinyVisionEncoder(config.embedding_dim).to(device)
    try:
        encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise RuntimeError(
            f"Trọng số standard encoder không tương thích: {encoder_path}"
        ) from error
    encoder.eval()
    return encoder, encoder_path, _sha256_file(encoder_path)


def _embedding_cache_metadata(
    config: ExperimentConfig,
    candidate_pool: Dataset,
    split_sha256: str | None,
    encoder_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "dataset": "cifar10",
        "split_sha256": split_sha256,
        "candidate_count": len(candidate_pool),
        "candidate_indices_sha256": _candidate_indices_sha256(candidate_pool),
        "encoder_sha256": encoder_sha256,
        "embedding_dim": config.embedding_dim,
        "transform": "cifar10-evaluation-v1",
        "normalization": "l2",
    }


def _load_or_rebuild_retriever(
    config: ExperimentConfig,
    candidate_pool: Dataset,
    encoder: TinyVisionEncoder,
    device: torch.device,
    split_sha256: str | None,
    encoder_sha256: str,
) -> CosineSimilarityRetriever:
    cache_dir = Path(config.output_dir).parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "candidate_pool_embeddings.pt"
    expected_metadata = _embedding_cache_metadata(
        config,
        candidate_pool,
        split_sha256,
        encoder_sha256,
    )

    try:
        return CosineSimilarityRetriever.from_cache(
            cache_file,
            candidate_dataset=candidate_pool,
            expected_metadata=expected_metadata,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"Rebuilding candidate embedding cache: {error}")

    precompute_candidate_embeddings(
        candidate_dataset=candidate_pool,
        encoder=encoder,
        batch_size=config.batch_size,
        device=device,
        output_path=cache_file,
        metadata=expected_metadata,
    )
    return CosineSimilarityRetriever.from_cache(
        cache_file,
        candidate_dataset=candidate_pool,
        expected_metadata=expected_metadata,
    )


def build_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader | None, Dataset, dict[str, Any]]:
    """Khởi tạo DataLoader tương ứng với từng baseline."""
    torch.manual_seed(config.seed)

    try:
        datasets = build_cifar10(
            root=config.data_root,
            split_file=config.split_file,
            download=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Không thể nạp CIFAR-10. Lần chạy đã dừng trước khi "
            "tạo output. Hãy kiểm tra --data-root và --split-file."
        ) from error

    query_train = datasets.query_train
    candidate_pool = datasets.candidate_pool
    validation = datasets.validation
    test_set = datasets.test
    split_path = Path(config.split_file)
    split_sha256 = (
        hashlib.sha256(split_path.read_bytes()).hexdigest()
        if split_path.is_file()
        else None
    )

    full_sizes = {
        "query_train": len(query_train),
        "candidate_pool": len(candidate_pool),
        "validation": len(validation),
        "test": len(test_set),
    }

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
        test_ds = test_set if config.eval_test else None

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
        test_ds = (
            ContextDataset(
                query_dataset=test_set,
                candidate_dataset=candidate_pool,
                num_candidates=1,
                seed=config.seed,
            )
            if config.eval_test
            else None
        )

    elif config.baseline == "similarity":
        encoder, _, encoder_sha256 = _load_standard_encoder(
            config,
            device,
            split_sha256,
        )
        retriever = _load_or_rebuild_retriever(
            config,
            candidate_pool,
            encoder,
            device,
            split_sha256,
            encoder_sha256,
        )

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
        test_ds = (
            SimilarityContextDataset(
                query_dataset=test_set,
                candidate_dataset=candidate_pool,
                retriever=retriever,
                encoder=encoder,
                device=device,
                batch_size=config.batch_size,
            )
            if config.eval_test
            else None
        )

    elif config.baseline == "tacs":
        train_ds = FullPoolContextDataset(
            query_dataset=query_train,
            candidate_dataset=candidate_pool,
        )
        val_ds = FullPoolContextDataset(
            query_dataset=validation,
            candidate_dataset=candidate_pool,
        )
        test_ds = (
            FullPoolContextDataset(
                query_dataset=test_set,
                candidate_dataset=candidate_pool,
            )
            if config.eval_test
            else None
        )

    elif config.baseline == "two_stage":
        encoder, _, encoder_sha256 = _load_standard_encoder(
            config,
            device,
            split_sha256,
        )
        retriever = _load_or_rebuild_retriever(
            config,
            candidate_pool,
            encoder,
            device,
            split_sha256,
            encoder_sha256,
        )
        train_ds = TopKContextDataset(
            query_dataset=query_train,
            candidate_dataset=candidate_pool,
            retriever=retriever,
            encoder=encoder,
            device=device,
            batch_size=config.batch_size,
            top_k=min(config.topk_candidates, len(candidate_pool)),
        )
        val_ds = TopKContextDataset(
            query_dataset=validation,
            candidate_dataset=candidate_pool,
            retriever=retriever,
            encoder=encoder,
            device=device,
            batch_size=config.batch_size,
            top_k=min(config.topk_candidates, len(candidate_pool)),
        )
        test_ds = (
            TopKContextDataset(
                query_dataset=test_set,
                candidate_dataset=candidate_pool,
                retriever=retriever,
                encoder=encoder,
                device=device,
                batch_size=config.batch_size,
                top_k=min(config.topk_candidates, len(candidate_pool)),
            )
            if config.eval_test
            else None
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
        if test_ds is not None
        else None
    )

    selection_info: dict[str, Any]
    if config.baseline == "no_context":
        selection_info = {
            "strategy": "no_context",
            "retrieval_candidates_scanned": 0,
            "selector_candidates": 0,
        }
    elif config.baseline == "random":
        selection_info = {
            "strategy": "random_context",
            "retrieval_candidates_scanned": 0,
            "selector_candidates": 1,
        }
    elif config.baseline == "similarity":
        selection_info = {
            "strategy": "cosine_top1",
            "retrieval_candidates_scanned": len(candidate_pool),
            "selector_candidates": 1,
            "retrieval_cache": "validated_standard_encoder_embedding_cache",
        }
    elif config.baseline == "tacs":
        selection_info = {
            "strategy": "full_pool_tacs",
            "retrieval_candidates_scanned": 0,
            "selector_candidates": len(candidate_pool),
            "candidate_bank_refresh": "once_per_train_or_evaluation_epoch",
            "candidate_bank_gradient": False,
        }
    elif config.baseline == "two_stage":
        selection_info = {
            "strategy": "cosine_topk_then_tacs",
            "retrieval_candidates_scanned": len(candidate_pool),
            "selector_candidates": min(config.topk_candidates, len(candidate_pool)),
            "candidate_bank_refresh": "per_batch_topk_candidates",
            "candidate_bank_gradient": False,
            "retrieval_cache": "validated_standard_encoder_embedding_cache",
        }

    data_info = {
        "source": "cifar10",
        "data_root": config.data_root,
        "split_file": config.split_file,
        "split_sha256": split_sha256,
        "full_sizes": full_sizes,
        "effective_sizes": {
            "query_train": len(train_ds),
            "candidate_pool": len(candidate_pool),
            "validation": len(val_ds),
            "test": len(test_set),
        },
        "selection": selection_info,
    }
    return train_loader, val_loader, test_loader, candidate_pool, data_info


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


def _encode_full_candidate_bank(
    model: TACSModel,
    candidate_dataset: Dataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode the complete pool once without duplicating images per query.

    The detached bank is refreshed for every train/evaluation epoch. Gradients
    still update the shared selector encoder through the query branch.
    """
    encoder = model.selector.encoder
    was_training = encoder.training
    encoder.eval()
    candidate_loader = DataLoader(
        candidate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in candidate_loader:
            embeddings.append(encoder(images.to(device)).detach())
    encoder.train(was_training)
    return torch.cat(embeddings, dim=0)


def _full_candidate_bank_for_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor | None:
    if not isinstance(loader.dataset, FullPoolContextDataset):
        return None
    if not isinstance(model, TACSModel):
        raise TypeError("FullPoolContextDataset yêu cầu TACSModel")
    return _encode_full_candidate_bank(
        model,
        loader.dataset.candidate_dataset,
        device,
        batch_size=loader.batch_size or 1,
    )


def _train_one_epoch_impl(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    device: torch.device,
) -> dict[str, float]:
    """Huấn luyện 1 epoch."""
    model.train()
    selection_started_at = time.perf_counter()
    full_candidate_bank = _full_candidate_bank_for_loader(model, loader, device)
    runtime_selection_time = time.perf_counter() - selection_started_at
    loop_started_at = time.perf_counter()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        query, target, candidates = unpack_batch(batch)
        query = query.to(device)
        target = target.to(device)
        if candidates is not None:
            candidates = candidates.to(device)

        optimizer.zero_grad()
        if full_candidate_bank is not None:
            logits = model.forward_embedding_bank(query, full_candidate_bank)
        else:
            logits = model(query, candidates)
        loss_value = criterion(logits, target)
        loss = loss_value.total if hasattr(loss_value, "total") else loss_value
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * query.size(0)
        pred_logits = logits.context_logits if hasattr(logits, "context_logits") else logits
        preds = pred_logits.argmax(dim=-1)
        correct += (preds == target).sum().item()
        total += query.size(0)

    model_time = time.perf_counter() - loop_started_at
    selection_time = (
        runtime_selection_time
        if full_candidate_bank is not None
        else float(getattr(loader.dataset, "precompute_time_s", 0.0))
    )
    actual_time = model_time + runtime_selection_time
    end_to_end_time = model_time + selection_time
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    throughput = total / max(1e-5, end_to_end_time)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "throughput": round(throughput, 2),
        "time": round(actual_time, 2),
        "end_to_end_time_s": round(end_to_end_time, 4),
        "model_time_s": round(model_time, 4),
        "selection_precompute_time_s": round(selection_time, 4),
        "latency_ms_per_query": round(
            end_to_end_time * 1000 / max(1, total),
            4,
        ),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    device: torch.device,
) -> dict[str, float | None]:
    """Train one epoch and include peak process/CUDA memory."""
    with PeakMemoryMonitor(device) as memory_monitor:
        result = _train_one_epoch_impl(model, loader, optimizer, criterion, device)
    return {**result, **memory_monitor.result().as_dict()}


def _evaluate_impl(
    model: nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
) -> dict[str, float]:
    """Đánh giá mô hình (không cập nhật gradient)."""
    model.eval()
    selection_started_at = time.perf_counter()
    full_candidate_bank = _full_candidate_bank_for_loader(model, loader, device)
    runtime_selection_time = time.perf_counter() - selection_started_at
    loop_started_at = time.perf_counter()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            query, target, candidates = unpack_batch(batch)
            query = query.to(device)
            target = target.to(device)
            if candidates is not None:
                candidates = candidates.to(device)

            if full_candidate_bank is not None:
                logits = model.forward_embedding_bank(query, full_candidate_bank)
            else:
                logits = model(query, candidates)
            loss_value = criterion(logits, target)
            loss = loss_value.total if hasattr(loss_value, "total") else loss_value

            total_loss += loss.item() * query.size(0)
            pred_logits = logits.context_logits if hasattr(logits, "context_logits") else logits
            preds = pred_logits.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += query.size(0)

    model_time = time.perf_counter() - loop_started_at
    selection_time = (
        runtime_selection_time
        if full_candidate_bank is not None
        else float(getattr(loader.dataset, "precompute_time_s", 0.0))
    )
    actual_time = model_time + runtime_selection_time
    end_to_end_time = model_time + selection_time
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    throughput = total / max(1e-5, end_to_end_time)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "throughput": round(throughput, 2),
        "time": round(actual_time, 2),
        "end_to_end_time_s": round(end_to_end_time, 4),
        "model_time_s": round(model_time, 4),
        "selection_precompute_time_s": round(selection_time, 4),
        "latency_ms_per_query": round(
            end_to_end_time * 1000 / max(1, total),
            4,
        ),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
) -> dict[str, float | None]:
    """Evaluate one split and include peak process/CUDA memory."""
    with PeakMemoryMonitor(device) as memory_monitor:
        result = _evaluate_impl(model, loader, criterion, device)
    return {**result, **memory_monitor.result().as_dict()}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Chạy toàn bộ quy trình huấn luyện và đánh giá cho 1 baseline."""
    configure_stdout()
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    data_setup_started_at = time.perf_counter()
    with PeakMemoryMonitor(device) as data_setup_memory_monitor:
        train_loader, val_loader, test_loader, candidate_pool, data_info = (
            build_dataloaders(config, device)
        )
    data_setup_time = time.perf_counter() - data_setup_started_at
    data_setup_memory = data_setup_memory_monitor.result().as_dict()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")
    print(
        f"Data source: {data_info['source']} | "
        f"effective sizes: {data_info['effective_sizes']}"
    )

    # Retrieval setup may instantiate an encoder internally. Resetting here
    # gives TACS Full and Two-Stage identical model initialisation for one seed.
    torch.manual_seed(config.seed)

    if config.baseline in {"tacs", "two_stage"}:
        model = TACSModel(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            temperature=config.selector_temperature,
            detach_candidate_embeddings=True,
        ).to(device)

        def tacs_criterion(outputs, targets):
            from .losses import TACSLoss
            return TACSLoss(policy_weight=config.policy_weight)(outputs, targets)

        criterion = tacs_criterion
    else:
        model = BaselineModel(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            baseline_type=config.baseline,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

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
        rss_mb = max(train_res["peak_rss_mb"], val_res["peak_rss_mb"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_acc": train_res["accuracy"],
            "val_loss": val_res["loss"],
            "val_acc": val_res["accuracy"],
            "throughput_qps": train_res["throughput"],
            "epoch_time_s": train_res["time"],
            "train_end_to_end_time_s": train_res["end_to_end_time_s"],
            "train_model_time_s": train_res["model_time_s"],
            "train_selection_precompute_time_s": train_res[
                "selection_precompute_time_s"
            ],
            "train_latency_ms_per_query": train_res["latency_ms_per_query"],
            "val_throughput_qps": val_res["throughput"],
            "val_time_s": val_res["time"],
            "val_end_to_end_time_s": val_res["end_to_end_time_s"],
            "val_model_time_s": val_res["model_time_s"],
            "val_selection_precompute_time_s": val_res[
                "selection_precompute_time_s"
            ],
            "val_latency_ms_per_query": val_res["latency_ms_per_query"],
            "train_peak_rss_mb": train_res["peak_rss_mb"],
            "val_peak_rss_mb": val_res["peak_rss_mb"],
            "train_peak_rss_delta_mb": train_res["peak_rss_delta_mb"],
            "val_peak_rss_delta_mb": val_res["peak_rss_delta_mb"],
            "train_peak_vram_allocated_mb": train_res[
                "peak_vram_allocated_mb"
            ],
            "val_peak_vram_allocated_mb": val_res["peak_vram_allocated_mb"],
            "train_peak_vram_reserved_mb": train_res["peak_vram_reserved_mb"],
            "val_peak_vram_reserved_mb": val_res["peak_vram_reserved_mb"],
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
            encoder = model.selector.encoder if isinstance(model, TACSModel) else model.encoder
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "metadata": {
                        "schema_version": ENCODER_CHECKPOINT_SCHEMA_VERSION,
                        "baseline": config.baseline,
                        "embedding_dim": config.embedding_dim,
                        "data": data_info,
                    },
                },
                out_dir / "best_encoder.pt",
            )

    training_time = time.perf_counter() - total_train_start
    total_time = round(training_time, 2)

    # Đánh giá test set (nếu yêu cầu)
    test_result = None
    if test_loader is not None:
        best_ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_result = evaluate(model, test_loader, criterion, device)
        print(f"  --> [TEST EVALUATION] Loss: {test_result['loss']:.4f}, Acc: {test_result['accuracy'] * 100:.2f}%")

    phase_peak_rss = [data_setup_memory["peak_rss_mb"]]
    phase_peak_vram_allocated = [data_setup_memory["peak_vram_allocated_mb"]]
    phase_peak_vram_reserved = [data_setup_memory["peak_vram_reserved_mb"]]
    for record in history:
        phase_peak_rss.extend(
            [record["train_peak_rss_mb"], record["val_peak_rss_mb"]]
        )
        phase_peak_vram_allocated.extend(
            [
                record["train_peak_vram_allocated_mb"],
                record["val_peak_vram_allocated_mb"],
            ]
        )
        phase_peak_vram_reserved.extend(
            [
                record["train_peak_vram_reserved_mb"],
                record["val_peak_vram_reserved_mb"],
            ]
        )
    if test_result is not None:
        phase_peak_rss.append(test_result["peak_rss_mb"])
        phase_peak_vram_allocated.append(test_result["peak_vram_allocated_mb"])
        phase_peak_vram_reserved.append(test_result["peak_vram_reserved_mb"])

    non_null_vram_allocated = [
        value for value in phase_peak_vram_allocated if value is not None
    ]
    non_null_vram_reserved = [
        value for value in phase_peak_vram_reserved if value is not None
    ]
    memory_summary = {
        "data_setup": data_setup_memory,
        "peak_rss_mb": max(phase_peak_rss),
        "peak_vram_allocated_mb": (
            max(non_null_vram_allocated) if non_null_vram_allocated else None
        ),
        "peak_vram_reserved_mb": (
            max(non_null_vram_reserved) if non_null_vram_reserved else None
        ),
    }
    run_time = time.perf_counter() - data_setup_started_at
    runtime_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }

    metrics_payload = {
        "metrics_schema_version": 2,
        "baseline": config.baseline,
        "seed": config.seed,
        "epochs": epochs,
        "config": asdict(config),
        "runtime": runtime_info,
        "data": data_info,
        "selection": data_info["selection"],
        "selector_temperature": (
            config.selector_temperature
            if config.baseline in {"tacs", "two_stage"}
            else None
        ),
        "policy_weight": (
            config.policy_weight
            if config.baseline in {"tacs", "two_stage"}
            else None
        ),
        "best_val_acc": round(best_val_acc, 4),
        "data_setup_time_s": round(data_setup_time, 4),
        "total_time_s": total_time,
        "end_to_end_time_s": round(run_time, 4),
        "memory": memory_summary,
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
    parser = argparse.ArgumentParser(description="CLI Huấn luyện baseline và mô hình TACS.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="no_context",
        choices=["no_context", "random", "similarity", "tacs", "two_stage"],
        help="Loại baseline hoặc mô hình cần chạy",
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
    parser.add_argument(
        "--topk-candidates",
        type=int,
        default=32,
        help="Số candidate cosine top-k đầu vào cho Two-Stage (TACS luôn dùng toàn pool)",
    )
    parser.add_argument(
        "--selector-temperature",
        type=float,
        default=0.1,
        help="Temperature Gumbel-Softmax của TACS selector",
    )
    parser.add_argument(
        "--policy-weight",
        type=float,
        default=1.0,
        help="Trọng số policy loss trong TACSLoss",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy ngắn 1 epoch trên một phần dữ liệu đã chọn",
    )
    parser.add_argument("--eval-test", action="store_true", help="Đánh giá test set sau khi hoàn thành")

    args = parser.parse_args()
    if args.topk_candidates <= 0:
        parser.error("--topk-candidates phải lớn hơn 0")

    if args.selector_temperature <= 0:
        parser.error("--selector-temperature phải lớn hơn 0")
    if args.policy_weight < 0:
        parser.error("--policy-weight không được âm")

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
        topk_candidates=args.topk_candidates,
        selector_temperature=args.selector_temperature,
        policy_weight=args.policy_weight,
        dry_run=args.dry_run,
        eval_test=args.eval_test,
    )


def main() -> None:
    configure_stdout()
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
