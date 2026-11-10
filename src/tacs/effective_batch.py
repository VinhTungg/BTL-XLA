"""Two-pass, sample-weighted policy normalization without retaining window graphs.

Inputs must already be materialized (including stochastic query augmentation).
No optimizer update is permitted between measurement and replay. Restore RNG
and registered buffers so replay doesn't consume a second set of random actions
or update BatchNorm statistics twice. Parameter gradients are NOT restored.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def capture_rng():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            random.getstate(), np.random.get_state())


def restore_rng(state):
    cpu, cuda, python, numpy = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(python)
    np.random.set_state(numpy)


def capture_buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()}


def restore_buffers(model, saved):
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(saved[name])


def backward_effective_window(model, batches, forward, criterion, autocast, scaler=None):
    """Backward one logical batch; caller owns zero_grad/step and AMP update.

`forward(batch)` returns (TACSOutput, targets). `autocast()` is a context
factory, called separately for each pass. CPU input buffering bounds VRAM.
The estimator uses detached window reward mean/std, not per-microbatch stats.
"""
    if not batches:
        raise ValueError("An effective batch cannot be empty")
    records, rewards = [], []
    with torch.no_grad():
        for batch in batches:
            rng, buffers = capture_rng(), capture_buffers(model)
            with autocast():
                output, targets = forward(batch)
                reward = criterion.rewards(output, targets).float()
            if not len(reward) or not bool(torch.isfinite(reward).all()):
                raise ValueError("Empty/non-finite reward in effective batch")
            records.append((rng, buffers, reward.detach().cpu(),
                            output.selection_weights.detach().argmax(-1).cpu()))
            rewards.append(reward.detach().cpu())
            del output, targets, reward
    end_rng, end_buffers = capture_rng(), capture_buffers(model)
    reward_window = torch.cat(rewards)
    advantage = (reward_window - reward_window.mean()) / (reward_window.std(unbiased=False) + criterion.eps)
    total = len(reward_window)
    stats = dict(loss_sum=0., task_loss_sum=0., policy_loss_sum=0., correct=0, samples=total,
                 reward_mean=float(reward_window.mean()), reward_std=float(reward_window.std(unbiased=False)),
                 replay_max_reward_delta=0.)
    offset = 0
    try:
        for batch, (rng, buffers, expected, actions) in zip(batches, records):
            restore_rng(rng)
            restore_buffers(model, buffers)
            with autocast():
                output, targets = forward(batch)
                current = criterion.rewards(output, targets).float().cpu()
                if (not torch.equal(output.selection_weights.detach().argmax(-1).cpu(), actions)
                        or not torch.allclose(current, expected, atol=1e-5, rtol=1e-4)):
                    raise RuntimeError("Replay changed actions/rewards; optimizer step refused")
                count = len(targets)
                loss = criterion(output, targets, advantage[offset:offset + count].to(targets.device))
                scaled_loss = loss.total * (count / total)
            if not bool(torch.isfinite(scaled_loss)):
                raise ValueError("Non-finite effective-batch loss")
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            stats['loss_sum'] += float(loss.total.detach()) * count
            stats['task_loss_sum'] += float(loss.task.detach()) * count
            stats['policy_loss_sum'] += float(loss.policy.detach()) * count
            stats['correct'] += int((output.context_logits.argmax(-1) == targets).sum())
            stats['replay_max_reward_delta'] = max(stats['replay_max_reward_delta'], float((current - expected).abs().max()))
            offset += count
            del output, targets, current, loss, scaled_loss
    finally:
        restore_rng(end_rng)
        restore_buffers(model, end_buffers)
    return stats
