from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DINOV3_VITS16_MODEL = "vit_small_patch16_dinov3.lvd1689m"
BACKBONE_EMBEDDING_DIMS = {
    "tiny": 64,
    "dinov3_vits16": 384,
}


class TinyVisionEncoder(nn.Module):
    """Small encoder used for CPU tests; replaceable by a ViT later."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = embedding_dim
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


class DINOv3VisionEncoder(nn.Module):
    """ViT-S/16 DINOv3 image encoder backed by the official timm port."""

    output_dim = 384
    image_size = 256

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                "DINOv3 yêu cầu timm>=1.0.20; hãy cài dependencies của project."
            ) from error

        self.model = timm.create_model(
            DINOV3_VITS16_MODEL,
            pretrained=pretrained,
            num_classes=0,
        )
        model_dim = int(getattr(self.model, "num_features", self.output_dim))
        if model_dim != self.output_dim:
            raise RuntimeError(
                f"DINOv3 ViT-S/16 phải trả về {self.output_dim} features, "
                f"nhưng model trả về {model_dim}."
            )

    def forward(self, images: Tensor) -> Tensor:
        embeddings = self.model(images)
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.output_dim:
            raise RuntimeError(
                "DINOv3 encoder phải trả về tensor [batch, 384]."
            )
        return embeddings


class TinyPatchTokenTaskNetwork(nn.Module):
    """Lightweight patch-token classifier used by CPU tests.

    Query and context patches are placed in one token sequence before the
    Transformer blocks, mirroring the classification architecture in Sec. 4.2
    of the paper.  The context image is omitted entirely for the no-context
    reward branch.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        patch_size: int = 4,
        depth: int = 2,
        max_patches: int = 1024,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        num_heads = 4 if embedding_dim % 4 == 0 else 1
        self.output_dim = embedding_dim
        self.patch_embed = nn.Conv2d(
            3,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_patches + 1, embedding_dim)
        )
        self.query_type_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.query_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.context_type_embedding, std=0.02)

    def _patch_tokens(self, images: Tensor) -> Tensor:
        return self.patch_embed(images).flatten(2).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        context: Tensor | None = None,
        context_proxy: Tensor | None = None,
    ) -> Tensor:
        query_tokens = self._patch_tokens(query)
        patch_count = query_tokens.shape[1]
        if patch_count > self.position_embedding.shape[1] - 1:
            raise ValueError("image produces more patches than max_patches")
        patch_positions = self.position_embedding[:, 1 : patch_count + 1]
        query_tokens = query_tokens + patch_positions + self.query_type_embedding

        token_parts = [
            self.cls_token.expand(query.shape[0], -1, -1)
            + self.position_embedding[:, :1],
            query_tokens,
        ]
        if context is not None:
            context_tokens = (
                self._patch_tokens(context)
                + patch_positions
                + self.context_type_embedding
            )
            if context_proxy is not None:
                if context_proxy.shape != (query.shape[0], self.output_dim):
                    raise ValueError(
                        "context_proxy must have shape [batch, embedding_dim]"
                    )
                context_tokens = context_tokens + context_proxy.unsqueeze(1)
            token_parts.append(context_tokens)

        tokens = torch.cat(token_parts, dim=1)
        tokens = self.norm(self.blocks(tokens))
        return self.head(tokens[:, 0])


class DINOv3PatchTokenTaskNetwork(nn.Module):
    """DINOv3 ViT-S/16 classifier that jointly attends query/context patches."""

    output_dim = 384

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = DINOv3VisionEncoder(pretrained=pretrained).model
        self.head = nn.Linear(self.output_dim, num_classes)

    def _forward_tokens(
        self,
        query: Tensor,
        context: Tensor | None,
        context_proxy: Tensor | None,
    ) -> Tensor:
        model = self.backbone
        query_patches = model.patch_embed(query)
        query_tokens, rope = model._pos_embed(query_patches)

        if context is not None:
            context_tokens = model.patch_embed(context)
            if context_tokens.ndim == 4:
                context_tokens = context_tokens.flatten(1, 2)
            if context_proxy is not None:
                if context_proxy.shape != (query.shape[0], self.output_dim):
                    raise ValueError(
                        "context_proxy must have shape [batch, embedding_dim]"
                    )
                context_tokens = context_tokens + context_proxy.unsqueeze(1)
            query_tokens = torch.cat((query_tokens, context_tokens), dim=1)
            if rope is not None:
                rope = torch.cat((rope, rope), dim=0)

        tokens = model.norm_pre(query_tokens)
        for block in model.blocks:
            tokens = block(tokens, rope=rope)
        return model.norm(tokens)

    def forward(
        self,
        query: Tensor,
        context: Tensor | None = None,
        context_proxy: Tensor | None = None,
    ) -> Tensor:
        tokens = self._forward_tokens(query, context, context_proxy)
        features = self.backbone.head_drop(self.backbone.fc_norm(tokens[:, 0]))
        return self.head(features)


def backbone_embedding_dim(backbone: str, tiny_embedding_dim: int = 64) -> int:
    if backbone == "tiny":
        return tiny_embedding_dim
    if backbone not in BACKBONE_EMBEDDING_DIMS:
        raise ValueError(f"Backbone không được hỗ trợ: {backbone}")
    return BACKBONE_EMBEDDING_DIMS[backbone]


def create_vision_encoder(
    backbone: str,
    embedding_dim: int = 64,
    pretrained: bool = True,
) -> nn.Module:
    if backbone == "tiny":
        return TinyVisionEncoder(embedding_dim)
    if backbone == "dinov3_vits16":
        expected_dim = backbone_embedding_dim(backbone)
        if embedding_dim != expected_dim:
            raise ValueError(
                f"{backbone} yêu cầu embedding_dim={expected_dim}, "
                f"nhưng nhận {embedding_dim}."
            )
        return DINOv3VisionEncoder(pretrained=pretrained)
    raise ValueError(f"Backbone không được hỗ trợ: {backbone}")


def create_task_network(
    backbone: str,
    num_classes: int,
    embedding_dim: int = 64,
    pretrained: bool = True,
) -> nn.Module:
    if backbone == "tiny":
        return TinyPatchTokenTaskNetwork(num_classes, embedding_dim)
    if backbone == "dinov3_vits16":
        expected_dim = backbone_embedding_dim(backbone)
        if embedding_dim != expected_dim:
            raise ValueError(
                f"{backbone} yêu cầu embedding_dim={expected_dim}, "
                f"nhưng nhận {embedding_dim}."
            )
        return DINOv3PatchTokenTaskNetwork(num_classes, pretrained=pretrained)
    raise ValueError(f"Backbone không được hỗ trợ: {backbone}")


class TaskAlignedSelector(nn.Module):
    """Learn task utility with the raw dot product from Eq. (1)."""

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
        query_embedding = self.encoder(query)
        flat_candidates = candidates.reshape(batch * count, channels, height, width)
        if self.detach_candidate_embeddings:
            with torch.no_grad():
                candidate_embedding = self.encoder(flat_candidates)
        else:
            candidate_embedding = self.encoder(flat_candidates)
        candidate_embedding = candidate_embedding.reshape(batch, count, -1)
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

        query_embedding = self.encoder(query)
        candidate_embedding = candidate_embeddings
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
    """Selector plus patch-token downstream network from the TACS paper."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        temperature: float = 0.1,
        detach_candidate_embeddings: bool = False,
        encoder: nn.Module | None = None,
        task_network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        selector_encoder = encoder if encoder is not None else TinyVisionEncoder(embedding_dim)
        encoder_dim = int(getattr(selector_encoder, "output_dim", embedding_dim))
        if encoder_dim != embedding_dim:
            raise ValueError(
                f"Encoder output_dim={encoder_dim} không khớp embedding_dim={embedding_dim}"
            )
        self.selector = TaskAlignedSelector(
            selector_encoder,
            temperature,
            detach_candidate_embeddings=detach_candidate_embeddings,
        )
        self.task_network = (
            task_network
            if task_network is not None
            else TinyPatchTokenTaskNetwork(num_classes, embedding_dim)
        )

    def forward(self, query: Tensor, candidates: Tensor) -> TACSOutput:
        scores, weights, _, _ = self.selector(query, candidates)
        selected_context = torch.einsum("bn,bnchw->bchw", weights, candidates)
        context_logits = self.task_network(query, selected_context)
        no_context_logits = self.task_network(query)
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
        selected_context: Tensor | None = None,
        selection: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> TACSOutput:
        scores, weights, _, selected_embedding = (
            selection
            if selection is not None
            else self.selector.forward_embedding_bank(query, candidate_embeddings)
        )
        if selected_context is None:
            raise ValueError(
                "selected_context is required when using an embedding bank"
            )
        # The hard candidate image is used in the forward pass.  The zero-value
        # proxy preserves the straight-through task gradient to the bank-based
        # Gumbel selection without materialising every pool image on the GPU.
        context_proxy = selected_embedding - selected_embedding.detach()
        context_logits = self.task_network(
            query,
            selected_context,
            context_proxy=context_proxy,
        )
        no_context_logits = self.task_network(query)
        return TACSOutput(
            context_logits,
            no_context_logits,
            scores,
            weights,
            self.selector.temperature,
        )
