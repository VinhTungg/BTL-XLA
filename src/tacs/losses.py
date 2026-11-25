from dataclasses import dataclass
import math

from torch import Tensor, nn
from torch.nn import functional as F

from .model import TACSOutput


@dataclass
class TACSLossOutput:
    total: Tensor
    task: Tensor
    policy: Tensor
    mean_reward: Tensor


class TACSLoss(nn.Module):
    """TACS hybrid objective with an optional supervised query-only term."""

    def __init__(
        self, policy_weight: float = 1.0, eps: float = 1e-6, *,
        aux_query_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(aux_query_weight) or aux_query_weight < 0:
            raise ValueError("aux_query_weight must be finite and non-negative")
        self.policy_weight = policy_weight
        self.eps = eps
        self.aux_query_weight = aux_query_weight

    @staticmethod
    def rewards(output: TACSOutput, targets: Tensor) -> Tensor:
        return (F.cross_entropy(output.no_context_logits, targets, reduction="none")
                - F.cross_entropy(output.context_logits, targets, reduction="none")).detach()

    def forward(self, output: TACSOutput, targets: Tensor,
                advantage: Tensor | None = None) -> TACSLossOutput:
        context_loss = F.cross_entropy(output.context_logits, targets, reduction="none")
        no_context_loss = F.cross_entropy(output.no_context_logits, targets, reduction="none")
        reward = (no_context_loss - context_loss).detach()
        if advantage is None:
            advantage = (reward - reward.mean()) / (reward.std(unbiased=False) + self.eps)
        elif (advantage.shape != reward.shape or advantage.requires_grad
              or not bool(advantage.isfinite().all())):
            raise ValueError("External advantage must be finite, detached and match targets")
        chosen_log_probability = (
            # Hard Gumbel-Max samples from softmax(scores); temperature only
            # controls the straight-through relaxation gradient.
            F.log_softmax(output.selector_scores, dim=-1)
            * output.selection_weights.detach()
        ).sum(dim=-1)
        policy_loss = -(chosen_log_probability * advantage).mean()
        task_loss = context_loss.mean()
        total = task_loss + self.policy_weight * policy_loss
        if self.aux_query_weight > 0:
            total = total + self.aux_query_weight * no_context_loss.mean()
        return TACSLossOutput(total, task_loss, policy_loss, reward.mean())
