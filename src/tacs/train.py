"""Hạ tầng huấn luyện và đánh giá dùng chung cho cả 3 baseline tuần 2.

Hỗ trợ 3 baseline dùng chung Task Network patch-token:
1. no_context: Chỉ dùng chuỗi patch token của query.
2. random: Ghép query với 1 candidate ngẫu nhiên từ candidate pool.
3. similarity: Ghép query với candidate có cosine similarity cao nhất.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .baselines import BaselineModel
from .data import (
    ContextDataset,
    FullPoolContextDataset,
    build_cifar10,
    build_cifar100,
    cifar_num_classes,
    normalize_cifar_name,
)
from .device import configure_stdout, resolve_device
from .metrics import PeakMemoryMonitor
from .model import (
    TASK_NETWORK_VERSIONS,
    TACSModel,
    backbone_embedding_dim,
    create_task_network,
    create_vision_encoder,
)
from .retrieval import (
    CosineSimilarityRetriever,
    SimilarityContextDataset,
    TopKContextDataset,
    precompute_candidate_embeddings,
)


ENCODER_CHECKPOINT_SCHEMA_VERSION = 1
EMBEDDING_CACHE_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 6


@dataclass
class ExperimentConfig:
    """Cấu hình thí nghiệm tuần 2."""

    baseline: str = "no_context"
    dataset: str = "cifar10"
    backbone: str = "tiny"
    task_network_version: str = "role_v2"
    pretrained_backbone: bool = True
    seed: int = 42
    epochs: int | None = None
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    advantage_scope: str = "microbatch"
    lr: float = 1e-3
    min_lr: float = 0.0
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    embedding_dim: int | None = None
    image_size: int | None = None
    num_classes: int | None = None
    device: str = "auto"
    data_root: str = "data"
    split_file: str | None = None
    output_dir: str | None = None
    standard_encoder_path: str | None = None
    topk_candidates: int = 32
    selector_temperature: float = 0.1
    policy_weight: float = 1.0
    dry_run: bool = False
    num_workers: int = 0
    eval_test: bool = False
    amp: bool = True
    resume: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    def __post_init__(self) -> None:
        self.dataset = normalize_cifar_name(self.dataset)
        if self.advantage_scope not in {"microbatch", "effective_batch"}:
            raise ValueError("advantage_scope must be microbatch or effective_batch")
        if self.advantage_scope == "effective_batch" and self.baseline not in {"tacs", "two_stage"}:
            raise ValueError("effective_batch advantage requires tacs/two_stage")
        if self.task_network_version not in TASK_NETWORK_VERSIONS:
            raise ValueError(f"Unsupported task_network_version: {self.task_network_version}")
        if self.backbone not in {"tiny", "dinov3_vits16"}:
            raise ValueError(f"Backbone không được hỗ trợ: {self.backbone}")
        if self.epochs is None:
            self.epochs = 100 if self.backbone == "dinov3_vits16" else 5
        elif self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.min_lr < 0 or self.min_lr >= self.lr:
            raise ValueError("min_lr must satisfy 0 <= min_lr < lr")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        expected_classes = cifar_num_classes(self.dataset)
        if self.num_classes is None:
            self.num_classes = expected_classes
        elif self.num_classes != expected_classes:
            raise ValueError(
                f"{self.dataset} requires num_classes={expected_classes}, "
                f"received {self.num_classes}"
            )

        expected_embedding_dim = backbone_embedding_dim(self.backbone)
        if self.embedding_dim is None:
            self.embedding_dim = expected_embedding_dim
        elif self.backbone != "tiny" and self.embedding_dim != expected_embedding_dim:
            raise ValueError(
                f"{self.backbone} requires embedding_dim={expected_embedding_dim}, "
                f"received {self.embedding_dim}"
            )

        expected_image_size = 32 if self.backbone == "tiny" else 256
        if self.image_size is None:
            self.image_size = expected_image_size
        elif self.image_size != expected_image_size:
            raise ValueError(
                f"{self.backbone} requires image_size={expected_image_size}, "
                f"received {self.image_size}"
            )

        if self.split_file is None:
            self.split_file = f"data/splits/{self.dataset}_seed42.json"
        if self.backbone == "tiny":
            base = "outputs" if self.dataset == "cifar10" else f"outputs/{self.dataset}"
        else:
            base = f"outputs/{self.dataset}/{self.backbone}"
            if self.task_network_version != "legacy_v1":
                base = f"{base}/{self.task_network_version}"
        if self.output_dir is None:
            self.output_dir = f"{base}/{self.baseline}"
        if self.standard_encoder_path is None:
            self.standard_encoder_path = f"{base}/no_context/best_encoder.pt"

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("task_network_version", "legacy_v1")
        return cls(**data)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a checkpoint atomically so interruption keeps the previous file."""
    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


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
) -> tuple[nn.Module, Path, str]:
    dataset_label = config.dataset.replace("cifar", "CIFAR-")
    encoder_path = Path(config.standard_encoder_path)
    if not encoder_path.is_file():
        raise RuntimeError(
            "Similarity/Two-Stage yêu cầu standard encoder đã huấn luyện trên "
            f"{dataset_label}, nhưng không tìm thấy: {encoder_path}"
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
        or metadata.get("backbone", "tiny") != config.backbone
        or data_metadata.get("source") != config.dataset
        or (
            expected_split_sha256 is not None
            and data_metadata.get("split_sha256") != expected_split_sha256
        )
    ):
        raise RuntimeError(
            f"Standard encoder {encoder_path} không khớp schema, baseline, "
            f"embedding_dim hoặc provenance {config.dataset}."
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

    encoder = create_vision_encoder(
        config.backbone,
        embedding_dim=config.embedding_dim,
        pretrained=False,
    ).to(device)
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
        "dataset": config.dataset,
        "split_sha256": split_sha256,
        "candidate_count": len(candidate_pool),
        "candidate_indices_sha256": _candidate_indices_sha256(candidate_pool),
        "encoder_sha256": encoder_sha256,
        "embedding_dim": config.embedding_dim,
        "backbone": config.backbone,
        "image_size": config.image_size,
        "transform": f"{config.dataset}-evaluation-v1",
        "normalization": "l2",
    }


def _load_or_rebuild_retriever(
    config: ExperimentConfig,
    candidate_pool: Dataset,
    encoder: nn.Module,
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
    dataset_label = config.dataset.replace("cifar", "CIFAR-")

    try:
        dataset_builder = build_cifar10 if config.dataset == "cifar10" else build_cifar100
        datasets = dataset_builder(
            root=config.data_root,
            split_file=config.split_file,
            download=False,
            image_size=config.image_size,
            normalization=("dinov3" if config.backbone == "dinov3_vits16" else "cifar"),
        )
    except Exception as error:
        raise RuntimeError(
            f"Không thể nạp {dataset_label}. Lần chạy đã dừng trước khi "
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
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
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
            "downstream_input": "query_patch_tokens",
        }
    elif config.baseline == "random":
        selection_info = {
            "strategy": "random_context",
            "retrieval_candidates_scanned": 0,
            "selector_candidates": 1,
            "downstream_input": "concatenated_query_context_patch_tokens",
        }
    elif config.baseline == "similarity":
        selection_info = {
            "strategy": "cosine_top1",
            "retrieval_candidates_scanned": len(candidate_pool),
            "selector_candidates": 1,
            "retrieval_cache": "validated_standard_encoder_embedding_cache",
            "downstream_input": "concatenated_query_context_patch_tokens",
        }
    elif config.baseline == "tacs":
        selection_info = {
            "strategy": "full_pool_tacs",
            "retrieval_candidates_scanned": 0,
            "selector_candidates": len(candidate_pool),
            "candidate_bank_refresh": "once_per_train_or_evaluation_epoch",
            "candidate_bank_gradient": False,
            "downstream_input": "concatenated_query_context_patch_tokens",
            "selected_context_loading": "hard_top1_image_only",
            "gradient_estimator": "straight_through_embedding_proxy",
            "selector_score_function": "raw_dot_product",
        }
    elif config.baseline == "two_stage":
        selection_info = {
            "strategy": "cosine_topk_then_tacs",
            "retrieval_candidates_scanned": len(candidate_pool),
            "selector_candidates": min(config.topk_candidates, len(candidate_pool)),
            "candidate_bank_refresh": "once_per_train_or_evaluation_epoch",
            "candidate_batch_representation": "topk_indices_into_shared_embedding_bank",
            "candidate_bank_gradient": False,
            "retrieval_cache": "validated_standard_encoder_embedding_cache",
            "downstream_input": "concatenated_query_context_patch_tokens",
            "selected_context_loading": "hard_top1_image_only",
            "gradient_estimator": "straight_through_embedding_proxy",
            "selector_score_function": "raw_dot_product",
        }

    data_info = {
        "source": config.dataset,
        "backbone": config.backbone,
        "image_size": config.image_size,
        "normalization": (
            "dinov3" if config.backbone == "dinov3_vits16" else "cifar"
        ),
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
    amp_enabled: bool = False,
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
        pin_memory=device.type == "cuda",
    )
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in candidate_loader:
            images = images.to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                embeddings.append(encoder(images).detach())
    encoder.train(was_training)
    return torch.cat(embeddings, dim=0)


def _candidate_bank_for_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> torch.Tensor | None:
    if not isinstance(loader.dataset, (FullPoolContextDataset, TopKContextDataset)):
        return None
    if not isinstance(model, TACSModel):
        raise TypeError("Candidate embedding bank yêu cầu TACSModel")
    return _encode_full_candidate_bank(
        model,
        loader.dataset.candidate_dataset,
        device,
        batch_size=loader.batch_size or 1,
        amp_enabled=amp_enabled,
    )


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _selected_pool_positions(
    batch: Any,
    selection_weights: torch.Tensor,
) -> torch.Tensor:
    """Map a hard local Gumbel choice to positions in the shared pool."""

    local_positions = selection_weights.detach().argmax(dim=-1)
    if isinstance(batch, dict) and "candidate_positions" in batch:
        topk_positions = batch["candidate_positions"].to(selection_weights.device)
        return topk_positions.gather(1, local_positions.unsqueeze(1)).squeeze(1)
    return local_positions


def _load_selected_context(
    candidate_dataset: Dataset,
    pool_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Load only the hard-selected images required by the Task Network."""

    images = [
        candidate_dataset[int(position)][0]
        for position in pool_positions.detach().cpu().tolist()
    ]
    return torch.stack(images).to(device, non_blocking=True)


def _forward_replay_batch(model, batch, loader, bank, device):
    """Forward the same materialized query and pool indices on either pass."""
    query, target, candidates = unpack_batch(batch)
    query, target = query.to(device), target.to(device)
    if bank is None:
        return model(query, None if candidates is None else candidates.to(device)), target
    embeddings = bank
    if isinstance(batch, dict) and "candidate_positions" in batch:
        embeddings = bank[batch["candidate_positions"].to(device)]
    selection = model.selector.forward_embedding_bank(query, embeddings)
    context = _load_selected_context(loader.dataset.candidate_dataset,
                                    _selected_pool_positions(batch, selection[1]), device)
    return model.forward_embedding_bank(query, embeddings, selected_context=context, selection=selection), target


def _train_effective_epoch(model, loader, optimizer, criterion, device, scaler, amp, accumulation):
    from itertools import islice
    from .effective_batch import backward_effective_window
    from .losses import TACSLoss
    if accumulation < 1 or not isinstance(criterion, TACSLoss) or not isinstance(model, TACSModel):
        raise ValueError("effective_batch requires positive accumulation, TACSModel and TACSLoss")
    model.train()
    started = time.perf_counter()
    bank = _candidate_bank_for_loader(model, loader, device, amp_enabled=amp)
    selection_time = time.perf_counter() - started
    loop_started = time.perf_counter()
    iterator = iter(loader)
    total = correct = steps = 0
    loss_sum = task_sum = policy_sum = replay_delta = 0.
    while batches := list(islice(iterator, accumulation)):
        optimizer.zero_grad(set_to_none=True)
        try:
            result = backward_effective_window(
                model, batches, lambda batch: _forward_replay_batch(model, batch, loader, bank, device),
                criterion, lambda: _autocast_context(device, amp), scaler)
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        total += result['samples']
        correct += result['correct']
        loss_sum += result['loss_sum']
        task_sum += result['task_loss_sum']
        policy_sum += result['policy_loss_sum']
        replay_delta = max(replay_delta, result['replay_max_reward_delta'])
        if steps == 1 or steps % 25 == 0:
            print(f"[effective-batch] step={steps} samples={total} reward_std={result['reward_std']:.6g} replay_delta={replay_delta:.3g}", flush=True)
    model_time = time.perf_counter() - loop_started
    elapsed = model_time + selection_time
    return dict(loss=round(loss_sum / max(1, total), 4), accuracy=round(correct / max(1, total), 4),
                throughput=round(total / max(1e-5, elapsed), 2), time=round(elapsed, 2),
                end_to_end_time_s=elapsed, model_time_s=model_time, selection_precompute_time_s=selection_time,
                latency_ms_per_query=elapsed * 1000 / max(1, total), optimizer_steps=steps,
                task_loss=task_sum / max(1, total), policy_loss=policy_sum / max(1, total),
                replay_max_reward_delta=replay_delta)


def _train_one_epoch_impl(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_accumulation_steps: int = 1,
    advantage_scope: str = "microbatch",
) -> dict[str, float]:
    """Huấn luyện 1 epoch."""
    if advantage_scope == "effective_batch":
        return _train_effective_epoch(model, loader, optimizer, criterion, device,
                                      scaler, amp_enabled, gradient_accumulation_steps)
    if advantage_scope != "microbatch":
        raise ValueError("Unknown advantage_scope")
    model.train()
    selection_started_at = time.perf_counter()
    candidate_bank = _candidate_bank_for_loader(
        model, loader, device, amp_enabled=amp_enabled
    )
    runtime_selection_time = time.perf_counter() - selection_started_at
    loop_started_at = time.perf_counter()
    total_loss = 0.0
    correct = 0
    total = 0
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    num_batches = len(loader)
    final_window_size = num_batches % gradient_accumulation_steps
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        query, target, candidates = unpack_batch(batch)
        query = query.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if candidates is not None:
            candidates = candidates.to(device, non_blocking=True)

        with _autocast_context(device, amp_enabled):
            if candidate_bank is not None:
                candidate_embeddings = candidate_bank
                if isinstance(batch, dict) and "candidate_positions" in batch:
                    candidate_embeddings = candidate_bank[
                        batch["candidate_positions"].to(device, non_blocking=True)
                    ]
                selection = model.selector.forward_embedding_bank(
                    query,
                    candidate_embeddings,
                )
                pool_positions = _selected_pool_positions(batch, selection[1])
                selected_context = _load_selected_context(
                    loader.dataset.candidate_dataset,
                    pool_positions,
                    device,
                )
                logits = model.forward_embedding_bank(
                    query,
                    candidate_embeddings,
                    selected_context=selected_context,
                    selection=selection,
                )
            else:
                logits = model(query, candidates)
            loss_value = criterion(logits, target)
            loss = loss_value.total if hasattr(loss_value, "total") else loss_value
        window_size = gradient_accumulation_steps
        if final_window_size and batch_index > num_batches - final_window_size:
            window_size = final_window_size
        backward_loss = loss / window_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        should_step = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == num_batches
        )
        if should_step:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        total_loss += loss.item() * query.size(0)
        pred_logits = logits.context_logits if hasattr(logits, "context_logits") else logits
        preds = pred_logits.argmax(dim=-1)
        correct += (preds == target).sum().item()
        total += query.size(0)

    model_time = time.perf_counter() - loop_started_at
    selection_time = (
        runtime_selection_time
        if candidate_bank is not None
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
        "optimizer_steps": optimizer_steps,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    gradient_accumulation_steps: int = 1,
    advantage_scope: str = "microbatch",
) -> dict[str, float | None]:
    """Train one epoch and include peak process/CUDA memory."""
    with PeakMemoryMonitor(device) as memory_monitor:
        result = _train_one_epoch_impl(
            model,
            loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=gradient_accumulation_steps,
            advantage_scope=advantage_scope,
        )
    return {**result, **memory_monitor.result().as_dict()}


def _evaluate_impl(
    model: nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Đánh giá mô hình (không cập nhật gradient)."""
    model.eval()
    selection_started_at = time.perf_counter()
    candidate_bank = _candidate_bank_for_loader(
        model, loader, device, amp_enabled=amp_enabled
    )
    runtime_selection_time = time.perf_counter() - selection_started_at
    loop_started_at = time.perf_counter()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            query, target, candidates = unpack_batch(batch)
            query = query.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if candidates is not None:
                candidates = candidates.to(device, non_blocking=True)

            with _autocast_context(device, amp_enabled):
                if candidate_bank is not None:
                    candidate_embeddings = candidate_bank
                    if isinstance(batch, dict) and "candidate_positions" in batch:
                        candidate_embeddings = candidate_bank[
                            batch["candidate_positions"].to(
                                device, non_blocking=True
                            )
                        ]
                    selection = model.selector.forward_embedding_bank(
                        query,
                        candidate_embeddings,
                    )
                    pool_positions = _selected_pool_positions(batch, selection[1])
                    selected_context = _load_selected_context(
                        loader.dataset.candidate_dataset,
                        pool_positions,
                        device,
                    )
                    logits = model.forward_embedding_bank(
                        query,
                        candidate_embeddings,
                        selected_context=selected_context,
                        selection=selection,
                    )
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
        if candidate_bank is not None
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
    amp_enabled: bool = False,
) -> dict[str, float | None]:
    """Evaluate one split and include peak process/CUDA memory."""
    with PeakMemoryMonitor(device) as memory_monitor:
        result = _evaluate_impl(
            model, loader, criterion, device, amp_enabled=amp_enabled
        )
    return {**result, **memory_monitor.result().as_dict()}


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: ExperimentConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create the AdamW + cosine recipe reported in the paper."""

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    cosine_epochs = config.epochs - config.warmup_epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=config.min_lr
    )
    if config.warmup_epochs:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler
    return optimizer, scheduler


RESUME_CONFIG_KEYS = (
    "baseline",
    "dataset",
    "backbone",
    "task_network_version",
    "seed",
    "embedding_dim",
    "image_size",
    "num_classes",
    "lr",
    "min_lr",
    "weight_decay",
    "gradient_accumulation_steps",
    "batch_size",
    "warmup_epochs",
    "topk_candidates",
    "selector_temperature",
    "policy_weight",
    "advantage_scope",
)


def _restore_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]], float]:
    """Restore an interrupted run from its best compatible checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint resume không hợp lệ: {checkpoint_path}")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError(
            f"Checkpoint resume thiếu config: {checkpoint_path}. "
            "Dùng --no-resume để train lại từ đầu."
        )
    checkpoint_config = dict(checkpoint_config)
    checkpoint_config.setdefault("task_network_version", "legacy_v1")
    checkpoint_config.setdefault("advantage_scope", "microbatch")
    current_config = asdict(config)
    mismatches = [
        key
        for key in RESUME_CONFIG_KEYS
        if checkpoint_config.get(key) != current_config.get(key)
    ]
    if "task_network_version" in mismatches:
        raise RuntimeError(
            "Checkpoint không khớp task_network_version. Chọn output directory khác "
            "cho kiến trúc mới; chỉ dùng --task-network-version legacy_v1 "
            "để tiếp tục checkpoint legacy."
        )
    if mismatches:
        raise RuntimeError(
            "Checkpoint không khớp cấu hình hiện tại tại: "
            f"{', '.join(mismatches)}. Dùng --no-resume hoặc output directory khác."
        )

    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "val_accuracy",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(
            f"Checkpoint resume thiếu trường bắt buộc: {sorted(missing)}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler.is_enabled() and isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
    rng_state = checkpoint.get("torch_rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state.cpu())
    cuda_rng_states = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and isinstance(cuda_rng_states, list):
        torch.cuda.set_rng_state_all(cuda_rng_states)

    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        history = []
    epoch = int(checkpoint["epoch"])
    best_val_acc = float(
        checkpoint.get("best_val_accuracy", checkpoint["val_accuracy"])
    )
    elapsed_time = float(checkpoint.get("elapsed_training_time_s", 0.0))
    return epoch + 1, best_val_acc, history, elapsed_time


def _validate_output_architecture(config: ExperimentConfig) -> None:
    """Reject cross-version overwrites, including --no-resume, before data setup."""
    out_dir = Path(config.output_dir)
    config_path = out_dir / "config.json"
    if config_path.is_file():
        try:
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot validate existing config: {config_path}") from exc
    else:
        checkpoint_path = next(
            (out_dir / name for name in ("checkpoint_latest.pt", "checkpoint_best.pt")
             if (out_dir / name).is_file()),
            None,
        )
        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        previous_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(previous_config, dict):
        raise RuntimeError(f"Missing/invalid existing config in {out_dir}; use another output directory.")
    previous_version = previous_config.get("task_network_version", "legacy_v1")
    if previous_config.get("advantage_scope", "microbatch") != config.advantage_scope:
        raise RuntimeError("Output belongs to a different advantage_scope; use another output directory")
    if previous_version != config.task_network_version:
        raise RuntimeError(
            f"Output {out_dir} belongs to task_network_version={previous_version}, "
            f"not {config.task_network_version}. Use another output directory; "
            "--no-resume does not override this architecture safeguard."
        )


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Chạy toàn bộ quy trình huấn luyện và đánh giá cho 1 baseline."""
    configure_stdout()
    _validate_output_architecture(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = bool(config.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_setup_started_at = time.perf_counter()
    with PeakMemoryMonitor(device) as data_setup_memory_monitor:
        train_loader, val_loader, test_loader, candidate_pool, data_info = (
            build_dataloaders(config, device)
        )
    data_setup_time = time.perf_counter() - data_setup_started_at
    data_setup_memory = data_setup_memory_monitor.result().as_dict()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Data source: {data_info['source']} | "
        f"effective sizes: {data_info['effective_sizes']}"
    )

    # Retrieval setup may instantiate an encoder internally. Resetting here
    # gives TACS Full and Two-Stage identical model initialisation for one seed.
    torch.manual_seed(config.seed)
    encoder = create_vision_encoder(
        config.backbone,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained_backbone,
    ).to(device)
    task_network = create_task_network(
        config.backbone,
        num_classes=config.num_classes,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained_backbone,
        task_network_version=config.task_network_version,
    ).to(device)

    if config.baseline in {"tacs", "two_stage"}:
        model = TACSModel(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            temperature=config.selector_temperature,
            detach_candidate_embeddings=True,
            encoder=encoder,
            task_network=task_network,
        ).to(device)

        from .losses import TACSLoss
        criterion = TACSLoss(policy_weight=config.policy_weight)
    else:
        model = BaselineModel(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            baseline_type=config.baseline,
            encoder=encoder,
            task_network=task_network,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    history: list[dict[str, Any]] = []
    epochs = config.epochs if not config.dry_run else 1
    start_epoch = 1
    resumed_from_epoch: int | None = None
    previous_training_time = 0.0
    best_checkpoint_path = out_dir / "checkpoint_best.pt"
    latest_checkpoint_path = out_dir / "checkpoint_latest.pt"
    resume_checkpoint_path = (
        latest_checkpoint_path
        if latest_checkpoint_path.is_file()
        else best_checkpoint_path
    )
    if config.resume and resume_checkpoint_path.is_file():
        (
            start_epoch,
            best_val_acc,
            history,
            previous_training_time,
        ) = _restore_training_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            device,
        )
        resumed_from_epoch = start_epoch - 1

    # Failed resume must not replace the original run's configuration.
    config.save(out_dir / "config.json")
    total_train_start = time.perf_counter()
    precision_label = "AMP FP16" if amp_enabled else "FP32"
    if resumed_from_epoch is None:
        progress_label = f"starting at epoch 1/{epochs}"
    else:
        progress_label = f"resuming after epoch {resumed_from_epoch}/{epochs}"
    print(
        f"\n[{config.baseline.upper()}] {progress_label} on device '{device}' "
        f"with {precision_label}..."
    )

    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_res = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            advantage_scope=config.advantage_scope,
        )
        val_res = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        rss_mb = max(train_res["peak_rss_mb"], val_res["peak_rss_mb"])

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "train_loss": train_res["loss"],
            "advantage_scope": config.advantage_scope,
            "train_task_loss": train_res.get("task_loss"),
            "train_policy_loss": train_res.get("policy_loss"),
            "replay_max_reward_delta": train_res.get("replay_max_reward_delta"),
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
            "optimizer_steps": train_res["optimizer_steps"],
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
            f"LR: {learning_rate:.3e} | {train_res['throughput']} q/s | "
            f"RAM: {rss_mb} MB"
        )

        # Lưu checkpoint tốt nhất
        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            best_checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "val_loss": val_res["loss"],
                "config": asdict(config),
            }
            _atomic_torch_save(best_checkpoint_data, best_checkpoint_path)

            # Lưu encoder chuẩn (đặc biệt quan trọng ở No-context cho Gói 2)
            encoder = model.selector.encoder if isinstance(model, TACSModel) else model.encoder
            _atomic_torch_save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "metadata": {
                        "schema_version": ENCODER_CHECKPOINT_SCHEMA_VERSION,
                        "baseline": config.baseline,
                        "embedding_dim": config.embedding_dim,
                        "backbone": config.backbone,
                        "image_size": config.image_size,
                        "data": data_info,
                    },
                },
                out_dir / "best_encoder.pt",
            )

        latest_checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_accuracy": val_res["accuracy"],
            "best_val_accuracy": best_val_acc,
            "val_loss": val_res["loss"],
            "config": asdict(config),
            "history": history,
            "elapsed_training_time_s": (
                previous_training_time + time.perf_counter() - total_train_start
            ),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
        }
        _atomic_torch_save(latest_checkpoint_data, latest_checkpoint_path)

    training_time = previous_training_time + time.perf_counter() - total_train_start
    total_time = round(training_time, 2)

    # Đánh giá test set (nếu yêu cầu)
    test_result = None
    if test_loader is not None:
        best_ckpt = torch.load(
            best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_result = evaluate(
            model,
            test_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
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
    run_time = previous_training_time + time.perf_counter() - data_setup_started_at
    runtime_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "precision": "amp_fp16" if amp_enabled else "fp32",
        "resumed_from_epoch": resumed_from_epoch,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }

    metrics_payload = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
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
        "training_recipe": {
            "optimizer": "AdamW",
            "scheduler": (
                "LinearWarmup+CosineAnnealingLR"
                if config.warmup_epochs
                else "CosineAnnealingLR"
            ),
            "initial_lr": config.lr,
            "minimum_lr": config.min_lr,
            "weight_decay": config.weight_decay,
            "scheduler_t_max": config.epochs - config.warmup_epochs,
            "warmup_epochs": config.warmup_epochs,
            "micro_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": (
                config.batch_size * config.gradient_accumulation_steps
            ),
            "advantage_scope": config.advantage_scope,
            "advantage_batch_size": (config.batch_size * config.gradient_accumulation_steps
                                     if config.advantage_scope == "effective_batch" else config.batch_size),
            "effective_batch_replay": config.advantage_scope == "effective_batch",
            "configured_epochs": config.epochs,
            "executed_epochs": epochs,
            "amp_enabled": amp_enabled,
            "resume_enabled": config.resume,
            "resumed_from_epoch": resumed_from_epoch,
        },
        "model_architecture": {
            "downstream_task_network": "patch_token_transformer",
            "backbone": config.backbone,
            "task_network_version": config.task_network_version,
            "image_role_encoding": (
                "query_and_context_type_embeddings" if config.backbone == "tiny"
                else "context_role_embedding" if config.task_network_version == "role_v2"
                else "none"
            ),
            "retrieval_encoder_trainable": False,
            "selector_encoder_trainable": config.baseline
            in {"tacs", "two_stage"},
        },
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Bộ dữ liệu thật dùng cho thí nghiệm",
    )
    parser.add_argument(
        "--backbone",
        default="tiny",
        choices=["tiny", "dinov3_vits16"],
        help="Backbone lightweight hoặc DINOv3 ViT-S/16 của paper",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Không tải pretrained weights (chỉ dùng cho kiểm tra kỹ thuật)",
    )
    parser.add_argument(
        "--task-network-version",
        choices=TASK_NETWORK_VERSIONS,
        default="role_v2",
        help="DINOv3 role_v2 phân biệt query/context; legacy_v1 chỉ để đọc/tiếp tục run cũ",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Tắt CUDA automatic mixed precision",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Bỏ qua checkpoint hiện có và train lại từ epoch đầu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số epochs (mặc định: tiny=5, DINOv3=100 theo paper)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--advantage-scope", choices=["microbatch", "effective_batch"], default="microbatch")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Số micro-batch được tích lũy trước mỗi optimizer step",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Learning rate tối thiểu của cosine annealing",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Số epoch linear warm-up trước cosine annealing",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay của AdamW",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", type=str, default="data", help="Thư mục dữ liệu CIFAR")
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất kết quả")
    parser.add_argument("--standard-encoder", type=str, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Số worker của mỗi DataLoader",
    )
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
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs phải lớn hơn 0")
    if args.lr <= 0:
        parser.error("--lr phải lớn hơn 0")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps phải lớn hơn 0")
    resolved_epochs = args.epochs or (100 if args.backbone == "dinov3_vits16" else 5)
    if args.warmup_epochs < 0 or args.warmup_epochs >= resolved_epochs:
        parser.error("--warmup-epochs phải thỏa 0 <= warmup-epochs < epochs")
    if args.min_lr < 0 or args.min_lr >= args.lr:
        parser.error("--min-lr phải thỏa 0 <= min-lr < lr")
    if args.weight_decay < 0:
        parser.error("--weight-decay không được âm")
    if args.num_workers < 0:
        parser.error("--num-workers không được âm")
    if args.topk_candidates <= 0:
        parser.error("--topk-candidates phải lớn hơn 0")

    if args.selector_temperature <= 0:
        parser.error("--selector-temperature phải lớn hơn 0")
    if args.policy_weight < 0:
        parser.error("--policy-weight không được âm")

    return ExperimentConfig(
        baseline=args.baseline,
        dataset=args.dataset,
        backbone=args.backbone,
        task_network_version=args.task_network_version,
        pretrained_backbone=not args.no_pretrained_backbone,
        amp=not args.no_amp,
        resume=not args.no_resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        advantage_scope=args.advantage_scope,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        split_file=args.split_file,
        output_dir=args.output_dir,
        standard_encoder_path=args.standard_encoder,
        num_workers=args.num_workers,
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
