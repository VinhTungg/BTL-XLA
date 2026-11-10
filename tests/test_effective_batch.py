import copy
from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tacs.effective_batch import backward_effective_window, capture_rng, restore_rng
from tacs.losses import TACSLoss
from tacs.model import TACSOutput, TACSModel
from tacs.train import ExperimentConfig, _validate_output_architecture, _restore_training_checkpoint, train_one_epoch


class Fixture(nn.Module):
    """Software fixture only, never used as experimental accuracy."""
    def __init__(self, stochastic=False):
        super().__init__()
        self.layer = nn.Linear(5, 3)
        self.norm = nn.BatchNorm1d(5) if stochastic else nn.Identity()
        self.drop = nn.Dropout(.3) if stochastic else nn.Identity()

    def forward(self, x):
        z = self.layer(self.drop(self.norm(x)))
        scores = z[:, :2]
        # A fixed hard action for full/microbatch equivalence; dropout is replay-tested separately.
        actions = F.one_hot((x[:, 0] > 0).long(), 2).float()
        return TACSOutput(z, z * .7, scores, actions, .1)


class EffectiveBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matches_full_batch_loss_and_gradients_including_short_microbatch(self):
        for sizes in [(4,) * 16, (4, 4), (3, 3, 1), (1,)]:
            with self.subTest(sizes=sizes):
                torch.manual_seed(12)
                model = Fixture()
                reference = copy.deepcopy(model)
                x, y = torch.randn(sum(sizes), 5), torch.arange(sum(sizes)) % 3
                loss = TACSLoss()(reference(x), y)
                loss.total.backward()
                batches = list(zip(x.split(sizes), y.split(sizes)))
                result = backward_effective_window(model, batches,
                    lambda b: (model(b[0]), b[1]), TACSLoss(), nullcontext)
                self.assertAlmostEqual(result['loss_sum'] / sum(sizes), float(loss.total.detach()), places=6)
                for p, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p.grad, expected.grad, atol=2e-7, rtol=2e-5)

    def test_dropout_rng_and_batchnorm_buffers_replayed_once(self):
        torch.manual_seed(2)
        model = Fixture(stochastic=True).train()
        ref = copy.deepcopy(model)
        batches = [(torch.randn(4, 5), torch.arange(4) % 3) for _ in range(3)]
        state = capture_rng()
        outputs = [(ref(x), y) for x, y in batches]
        expected_rng = torch.get_rng_state().clone()
        rewards = torch.cat([TACSLoss.rewards(o, y) for o, y in outputs])
        advantage = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        for i, (o, y) in enumerate(outputs):
            (TACSLoss()(o, y, advantage[i*4:(i+1)*4]).total / 3).backward()
        restore_rng(state)
        backward_effective_window(model, batches, lambda b: (model(b[0]), b[1]), TACSLoss(), nullcontext)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for p, expected in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(p.grad, expected.grad)
        for b, expected in zip(model.buffers(), ref.buffers()):
            torch.testing.assert_close(b, expected)

    def test_nonfinite_reward_and_empty_window_rejected(self):
        model = Fixture()
        with self.assertRaises(ValueError):
            backward_effective_window(model, [], None, TACSLoss(), nullcontext)
        with self.assertRaisesRegex(ValueError, 'reward'):
            backward_effective_window(model, [(torch.full((2, 5), float('nan')), torch.zeros(2, dtype=torch.long))],
                lambda b: (model(b[0]), b[1]), TACSLoss(), nullcontext)

    def test_replay_drift_fails_before_optimizer_step(self):
        model = Fixture()
        calls = 0
        def forward(b):
            nonlocal calls
            calls += 1
            output = model(b[0])
            if calls > 1:
                output.context_logits = output.context_logits * 2
            return output, b[1]
        with self.assertRaisesRegex(RuntimeError, 'Replay'):
            backward_effective_window(model, [(torch.randn(4, 5), torch.arange(4) % 3)],
                                      forward, TACSLoss(), nullcontext)

    def test_external_advantage_validated(self):
        output = Fixture()(torch.randn(3, 5))
        for adv in [torch.zeros(2), torch.full((3,), float('nan')), torch.zeros(3, requires_grad=True)]:
            with self.assertRaises(ValueError):
                TACSLoss()(output, torch.tensor([0, 1, 2]), adv)

    def test_real_training_loop_short_window_and_default_config(self):
        torch.manual_seed(5)
        model = TACSModel(3, embedding_dim=8)
        rows = [dict(query=torch.randn(3, 8, 8), candidates=torch.randn(2, 3, 8, 8), target=i % 3)
                for i in range(7)]
        result = train_one_epoch(model, DataLoader(rows, batch_size=3),
            torch.optim.SGD(model.parameters(), lr=.01), TACSLoss(), torch.device('cpu'),
            gradient_accumulation_steps=2, advantage_scope='effective_batch')
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['replay_max_reward_delta'], 0)
        self.assertEqual(ExperimentConfig().advantage_scope, 'microbatch')

    def test_output_scope_guard_and_invalid_baseline(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(advantage_scope='effective_batch')
        with tempfile.TemporaryDirectory() as temp:
            old = ExperimentConfig(baseline='tacs', output_dir=temp)
            old.save(Path(temp) / 'config.json')
            new = ExperimentConfig(baseline='tacs', output_dir=temp, advantage_scope='effective_batch')
            with self.assertRaisesRegex(RuntimeError, 'advantage_scope'):
                _validate_output_architecture(new)

    def test_resume_rejects_scope_change_before_mutating_model(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as temp:
            model = nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
            old = asdict(ExperimentConfig(baseline='tacs'))
            old.pop('advantage_scope')  # Old checkpoint schema is microbatch.
            path = Path(temp)/'checkpoint.pt'
            torch.save(dict(config=old), path)
            before = model.weight.detach().clone()
            with self.assertRaisesRegex(RuntimeError, 'advantage_scope'):
                _restore_training_checkpoint(path, model, optimizer, scheduler,
                    torch.amp.GradScaler('cuda', enabled=False),
                    ExperimentConfig(baseline='tacs', advantage_scope='effective_batch'), torch.device('cpu'))
            torch.testing.assert_close(before, model.weight)


if __name__ == '__main__':
    unittest.main()
