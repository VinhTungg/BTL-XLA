"""Classification with one fixed cosine-retrieved context per query."""

from torch import Tensor, nn
from torch.nn import functional as F

from ..model import PairedLogits


class CosineTop1Model(nn.Module):
    def __init__(self, task_network: nn.Module) -> None:
        super().__init__()
        self.task_network = task_network

    def forward(self, query: Tensor, candidates: Tensor | None = None) -> PairedLogits:
        if candidates is None or candidates.ndim != 5 or candidates.shape[1] != 1:
            raise ValueError("Cosine top-1 requires exactly one context image per query")
        return PairedLogits(
            context_logits=self.task_network(query, candidates[:, 0]),
            no_context_logits=self.task_network(query),
        )


class PairedClassificationLoss(nn.Module):
    def __init__(self, aux_query_weight: float) -> None:
        super().__init__()
        self.aux_query_weight = aux_query_weight

    def forward(self, output: PairedLogits, targets: Tensor) -> Tensor:
        return (
            F.cross_entropy(output.context_logits, targets)
            + self.aux_query_weight * F.cross_entropy(output.no_context_logits, targets)
        )
