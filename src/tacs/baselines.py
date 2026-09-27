"""Mô hình Baseline cho Tuần 2: No-context, Random-context, Similarity-context."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import TinyVisionEncoder


class BaselineModel(nn.Module):
    """Mô hình phân loại cho cả 3 baseline với cấu trúc chung tuân thủ WEEK_2_ASSIGNMENT.

    Cấu trúc mạng:
    - Backbone: TinyVisionEncoder (embedding_dim)
    - Classifier: Linear(embedding_dim * 2, embedding_dim) -> GELU -> Linear(embedding_dim, num_classes)

    Chế độ hoạt động:
    - 'no_context': Context vector được đặt bằng vector 0 (torch.zeros_like(query_feat)).
    - 'random': Context vector được mã hóa từ 1 candidate ngẫu nhiên lấy từ pool.
    - 'similarity': Context vector được mã hóa từ 1 candidate có cosine similarity cao nhất.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 64,
        baseline_type: str = "no_context",
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        valid_types = {"no_context", "random", "similarity"}
        if baseline_type not in valid_types:
            raise ValueError(f"baseline_type phải thuộc {valid_types}, nhận '{baseline_type}'")
        self.baseline_type = baseline_type
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.encoder = encoder if encoder is not None else TinyVisionEncoder(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def extract_embedding(self, images: Tensor) -> Tensor:
        """Trích xuất feature vector chuẩn hóa L2 từ encoder."""
        features = self.encoder(images)
        return F.normalize(features, dim=-1)

    def forward(self, query: Tensor, candidates: Tensor | None = None) -> Tensor:
        """Forward pass cho phân loại.

        Args:
            query: Tensor [B, 3, 32, 32]
            candidates: Tensor [B, 1, 3, 32, 32] hoặc [B, 3, 32, 32] hoặc None (cho no_context)

        Returns:
            logits: Tensor [B, num_classes]
        """
        query_feat = self.extract_embedding(query)

        if self.baseline_type == "no_context" or candidates is None:
            context_feat = torch.zeros_like(query_feat)
        else:
            if candidates.ndim == 5:
                cand_img = candidates[:, 0]
            elif candidates.ndim == 4:
                cand_img = candidates
            else:
                raise ValueError(f"candidates phải có 4 hoặc 5 chiều, nhận ndim={candidates.ndim}")
            context_feat = self.extract_embedding(cand_img)

        fused = torch.cat((query_feat, context_feat), dim=-1)
        logits = self.classifier(fused)
        return logits
