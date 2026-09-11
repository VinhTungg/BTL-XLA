from dataclasses import dataclass

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
    """Hybrid objective from equations (8)-(10) of the TACS paper."""

    def __init__(self, policy_weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.policy_weight = policy_weight
        self.eps = eps

    def forward(self, output: TACSOutput, targets: Tensor) -> TACSLossOutput:
        context_loss = F.cross_entropy(output.context_logits, targets, reduction="none")
        no_context_loss = F.cross_entropy(output.no_context_logits, targets, reduction="none")
        reward = (no_context_loss - context_loss).detach()
        advantage = (reward - reward.mean()) / (reward.std(unbiased=False) + self.eps)
        chosen_log_probability = (
            F.log_softmax(output.selector_scores, dim=-1) * output.selection_weights.detach()
        ).sum(dim=-1)
        policy_loss = -(chosen_log_probability * advantage).mean()
        task_loss = context_loss.mean()
        total = task_loss + self.policy_weight * policy_loss
        return TACSLossOutput(total, task_loss, policy_loss, reward.mean())

