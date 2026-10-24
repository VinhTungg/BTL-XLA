from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyVisionEncoder(nn.Module):
    """Small encoder used for CPU tests; replaceable by a ViT later."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(32, embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.features(images).flatten(1))


class TaskAlignedSelector(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        temperature: float = 0.1,
        detach_candidate_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.encoder = encoder
        self.temperature = temperature
        self.detach_candidate_embeddings = detach_candidate_embeddings

    def forward(self, query: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if candidates.ndim != 5 or query.shape[0] != candidates.shape[0]:
            raise ValueError("expected query [B,C,H,W] and candidates [B,N,C,H,W]")
        batch, count, channels, height, width = candidates.shape
        query_embedding = F.normalize(self.encoder(query), dim=-1)
        flat_candidates = candidates.reshape(batch * count, channels, height, width)
        if self.detach_candidate_embeddings:
            with torch.no_grad():
                candidate_embedding = self.encoder(flat_candidates)
        else:
            candidate_embedding = self.encoder(flat_candidates)
        candidate_embedding = F.normalize(candidate_embedding.reshape(batch, count, -1), dim=-1)
        scores = torch.einsum("bd,bnd->bn", query_embedding, candidate_embedding)
        if self.training:
            weights = F.gumbel_softmax(scores, tau=self.temperature, hard=True, dim=-1)
        else:
            index = scores.argmax(dim=-1)
            weights = F.one_hot(index, num_classes=count).to(scores.dtype)
        selected_embedding = torch.einsum("bn,bnd->bd", weights, candidate_embedding)
        return scores, weights, query_embedding, selected_embedding

    def forward_embedding_bank(
        self,
        query: Tensor,
        candidate_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Score a shared full-pool bank or one bank per query.

        ``candidate_embeddings`` accepts ``[N, D]`` for a bank shared by the
        complete query batch, or ``[B, N, D]`` for per-query candidates.
        """
        if candidate_embeddings.ndim not in {2, 3}:
            raise ValueError("expected candidate embeddings [N,D] or [B,N,D]")

        query_embedding = F.normalize(self.encoder(query), dim=-1)
        candidate_embedding = F.normalize(candidate_embeddings, dim=-1)
        if candidate_embedding.ndim == 2:
            scores = torch.matmul(query_embedding, candidate_embedding.T)
            count = candidate_embedding.shape[0]
        else:
            if candidate_embedding.shape[0] != query.shape[0]:
                raise ValueError("per-query candidate bank must match query batch")
            scores = torch.einsum("bd,bnd->bn", query_embedding, candidate_embedding)
            count = candidate_embedding.shape[1]

        if self.training:
            weights = F.gumbel_softmax(scores, tau=self.temperature, hard=True, dim=-1)
        else:
            index = scores.argmax(dim=-1)
            weights = F.one_hot(index, num_classes=count).to(scores.dtype)

        if candidate_embedding.ndim == 2:
            selected_embedding = torch.matmul(weights, candidate_embedding)
        else:
            selected_embedding = torch.einsum("bn,bnd->bd", weights, candidate_embedding)
        return scores, weights, query_embedding, selected_embedding


@dataclass
class TACSOutput:
    context_logits: Tensor
    no_context_logits: Tensor
    selector_scores: Tensor
    selection_weights: Tensor
    selector_temperature: float


class TACSModel(nn.Module):
    """Classification MVP following the two-module structure in the paper."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        temperature: float = 0.1,
        detach_candidate_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.selector = TaskAlignedSelector(
            TinyVisionEncoder(embedding_dim),
            temperature,
            detach_candidate_embeddings=detach_candidate_embeddings,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, query: Tensor, candidates: Tensor) -> TACSOutput:
        scores, weights, query_embedding, selected_embedding = self.selector(query, candidates)
        context_logits = self.classifier(torch.cat((query_embedding, selected_embedding), dim=-1))
        no_context_logits = self.classifier(torch.cat((query_embedding, torch.zeros_like(query_embedding)), dim=-1))
        return TACSOutput(
            context_logits,
            no_context_logits,
            scores,
            weights,
            self.selector.temperature,
        )

    def forward_embedding_bank(
        self,
        query: Tensor,
        candidate_embeddings: Tensor,
    ) -> TACSOutput:
        scores, weights, query_embedding, selected_embedding = (
            self.selector.forward_embedding_bank(query, candidate_embeddings)
        )
        context_logits = self.classifier(
            torch.cat((query_embedding, selected_embedding), dim=-1)
        )
        no_context_logits = self.classifier(
            torch.cat((query_embedding, torch.zeros_like(query_embedding)), dim=-1)
        )
        return TACSOutput(
            context_logits,
            no_context_logits,
            scores,
            weights,
            self.selector.temperature,
        )
