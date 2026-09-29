"""Module trích xuất embedding và truy xuất Cosine Similarity cho Similarity-context baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def extract_normalized_features(
    encoder: nn.Module,
    images: Tensor,
) -> Tensor:
    """Trích xuất và chuẩn hóa L2 feature vector."""
    features = encoder(images)
    return F.normalize(features, dim=-1)


def precompute_candidate_embeddings(
    candidate_dataset: Dataset,
    encoder: nn.Module,
    batch_size: int = 128,
    device: torch.device | str = "cpu",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Tiền tính toán toàn bộ embedding cho candidate pool và lưu cache."""
    device = torch.device(device)
    encoder = encoder.to(device)
    encoder.eval()

    loader = DataLoader(
        candidate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_indices: list[int] = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            feats = extract_normalized_features(encoder, images)
            all_embeddings.append(feats.cpu())
            all_targets.append(targets.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # Lấy index gốc trong dataset nếu là Subset
    if isinstance(candidate_dataset, Subset):
        orig_indices = list(candidate_dataset.indices)
    else:
        orig_indices = list(range(len(candidate_dataset)))
    indices_tensor = torch.tensor(orig_indices, dtype=torch.long)

    result = {
        "embeddings": embeddings_tensor,
        "targets": targets_tensor,
        "indices": indices_tensor,
        "embedding_dim": embeddings_tensor.shape[-1],
        "num_candidates": len(candidate_dataset),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)

    return result


class CosineSimilarityRetriever:
    """Truy xuất ứng viên có Cosine Similarity lớn nhất trong candidate pool."""

    def __init__(
        self,
        candidate_embeddings: Tensor,
        candidate_indices: Tensor,
        candidate_targets: Tensor,
        candidate_dataset: Dataset | None = None,
    ) -> None:
        self.candidate_embeddings = candidate_embeddings.float()  # [M, D] (đã chuẩn hóa L2)
        self.candidate_indices = candidate_indices
        self.candidate_targets = candidate_targets
        self.candidate_dataset = candidate_dataset

    @classmethod
    def from_cache(
        cls,
        cache_path: str | Path,
        candidate_dataset: Dataset | None = None,
    ) -> "CosineSimilarityRetriever":
        """Nạp retriever từ file cache embedding đã lưu."""
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cls(
            candidate_embeddings=data["embeddings"],
            candidate_indices=data["indices"],
            candidate_targets=data["targets"],
            candidate_dataset=candidate_dataset,
        )

    def retrieve_top1(
        self,
        query_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Truy xuất top-1 ứng viên có Cosine Similarity cao nhất cho mỗi query.

        Args:
            query_embeddings: Tensor [B, D] (đã chuẩn hóa L2)

        Returns:
            scores: Tensor [B] (giá trị Cosine Similarity cao nhất)
            pool_positions: Tensor [B] (vị trí trong candidate pool)
            orig_indices: Tensor [B] (index gốc trong dataset CIFAR-10)
        """
        query_embeddings = query_embeddings.float().to(self.candidate_embeddings.device)
        # Cosine similarity = query_norm @ candidate_norm.T vì cả 2 đã chuẩn hóa L2
        sim_matrix = torch.matmul(query_embeddings, self.candidate_embeddings.T)  # [B, M]
        best_scores, best_positions = torch.max(sim_matrix, dim=-1)  # [B]
        orig_indices = self.candidate_indices[best_positions]
        return best_scores.cpu(), best_positions.cpu(), orig_indices.cpu()


class SimilarityContextDataset(Dataset):
    """Dataset kết hợp Query với ứng viên Similarity-context được truy xuất trước."""

    def __init__(
        self,
        query_dataset: Dataset,
        candidate_dataset: Dataset,
        retriever: CosineSimilarityRetriever,
        encoder: nn.Module | None = None,
        device: torch.device | str = "cpu",
        batch_size: int = 128,
    ) -> None:
        self.query_dataset = query_dataset
        self.candidate_dataset = candidate_dataset
        self.retriever = retriever

        # Tiền tính toán mapping top-1 candidate cho toàn bộ query để tăng tốc training
        self.cached_assignments: list[tuple[int, float]] = []
        if encoder is not None:
            self._precompute_assignments(encoder, device=device, batch_size=batch_size)

    def _precompute_assignments(
        self,
        encoder: nn.Module,
        device: torch.device | str = "cpu",
        batch_size: int = 128,
    ) -> None:
        device = torch.device(device)
        encoder = encoder.to(device)
        encoder.eval()

        loader = DataLoader(
            self.query_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        assignments: list[tuple[int, float]] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                q_embs = extract_normalized_features(encoder, images)
                scores, best_positions, _ = self.retriever.retrieve_top1(q_embs)
                for pos, sc in zip(best_positions.tolist(), scores.tolist()):
                    assignments.append((pos, sc))

        self.cached_assignments = assignments

    def __len__(self) -> int:
        return len(self.query_dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        query_image, query_target = self.query_dataset[index]

        if self.cached_assignments:
            cand_pos, sim_score = self.cached_assignments[index]
        else:
            # Fallback nếu chưa precompute
            cand_pos = 0
            sim_score = 0.0

        cand_image, cand_target = self.candidate_dataset[cand_pos]

        if isinstance(self.candidate_dataset, Subset):
            cand_orig_idx = self.candidate_dataset.indices[cand_pos]
        else:
            cand_orig_idx = cand_pos

        if isinstance(self.query_dataset, Subset):
            query_orig_idx = self.query_dataset.indices[index]
        else:
            query_orig_idx = index

        return {
            "query": query_image,
            "candidates": cand_image.unsqueeze(0),  # [1, 3, 32, 32]
            "target": torch.as_tensor(query_target, dtype=torch.long),
            "query_index": torch.tensor(query_orig_idx, dtype=torch.long),
            "candidate_indices": torch.tensor([cand_orig_idx], dtype=torch.long),
            "candidate_targets": torch.tensor([cand_target], dtype=torch.long),
            "similarity_scores": torch.tensor([sim_score], dtype=torch.float32),
        }
