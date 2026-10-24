"""Mô hình Baseline cho Tuần 2: No-context, Random-context, Similarity-context."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .model import TinyPatchTokenTaskNetwork, TinyVisionEncoder


class BaselineModel(nn.Module):
    """Baselines sharing the same patch-token downstream architecture as TACS.

    The frozen encoder is used only to build reproducible similarity retrieval
    caches.  Classification always goes through ``task_network`` so the three
    baselines differ from TACS by context selection, not downstream capacity.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 64,
        baseline_type: str = "no_context",
        encoder: nn.Module | None = None,
        task_network: nn.Module | None = None,
        freeze_retrieval_encoder: bool = True,
    ) -> None:
        super().__init__()
        valid_types = {"no_context", "random", "similarity"}
        if baseline_type not in valid_types:
            raise ValueError(f"baseline_type phải thuộc {valid_types}, nhận '{baseline_type}'")
        self.baseline_type = baseline_type
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.encoder = encoder if encoder is not None else TinyVisionEncoder(embedding_dim)
        self.freeze_retrieval_encoder = freeze_retrieval_encoder
        if freeze_retrieval_encoder:
            self.encoder.requires_grad_(False)
        self.task_network = (
            task_network
            if task_network is not None
            else TinyPatchTokenTaskNetwork(num_classes, embedding_dim)
        )

    def train(self, mode: bool = True) -> "BaselineModel":
        super().train(mode)
        if self.freeze_retrieval_encoder:
            self.encoder.eval()
        return self

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
        if self.baseline_type == "no_context" or candidates is None:
            return self.task_network(query)
        if candidates.ndim == 5:
            context = candidates[:, 0]
        elif candidates.ndim == 4:
            context = candidates
        else:
            raise ValueError(
                f"candidates phải có 4 hoặc 5 chiều, nhận ndim={candidates.ndim}"
            )
        return self.task_network(query, context)
