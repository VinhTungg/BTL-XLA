"""Small unit fixtures only, never used as experimental data."""
import copy
import unittest

import torch
from torch.nn import functional as F

from tacs.gradient_audit import accumulate, cpu_gradients, gradient_pair_summary, reward_window_summary, standardize_reward, state_digest
from tacs.losses import TACSLoss
from tacs.model import TACSModel, TACSOutput


class GradientAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_microbatch_normalization_is_not_whole_window(self):
        result = reward_window_summary([[-4., -3.], [3., 4.]])
        self.assertEqual(result["micro_vs_window_advantage_sign_flip_fraction"], .5)
        self.assertEqual(result["negative_reward_positive_micro_advantage_count"], 1)
        self.assertEqual(result["positive_reward_negative_micro_advantage_count"], 1)

    def test_singleton_and_constant_reward_zero_advantage(self):
        self.assertTrue(torch.equal(standardize_reward(torch.tensor([3.])), torch.zeros(1)))
        result = reward_window_summary([[2., 2.], [2., 2.]])
        self.assertEqual(result["zero_variance_microbatches"], 2)
        self.assertEqual(result["mean_abs_micro_vs_window_advantage_delta"], 0.)

    def test_nonfinite_rewards_rejected(self):
        with self.assertRaises(ValueError):
            reward_window_summary([[float('nan')]])

    def test_opposing_gradients_and_weighted_sum(self):
        result = gradient_pair_summary([torch.tensor([3., 4.])], [torch.tensor([-3., -4.])], 1.)
        self.assertEqual(result["task_l2"], 5.)
        self.assertEqual(result["task_policy_cosine"], -1.)
        self.assertEqual(result["combined_l2"], 0.)

    def test_disconnected_and_zero_gradients_are_distinct(self):
        result = gradient_pair_summary([None, torch.zeros(2)], [torch.ones(1), None])
        self.assertEqual(result["task_connected_tensors"], 1)
        self.assertEqual(result["task_nonzero_elements"], 0)
        self.assertIsNone(result["task_policy_cosine"])
        self.assertIsNone(result["weighted_policy_to_task_l2_ratio"])

    def test_nonfinite_gradients_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            gradient_pair_summary([torch.tensor([float('inf')])], [None])

    def test_accumulation_and_loss_scaling(self):
        x = torch.tensor([2., 3.], requires_grad=True)
        gradient = cpu_gradients(x.square().sum(), [x], 128., False)
        torch.testing.assert_close(gradient[0], torch.tensor([4., 6.]))
        accumulator = [None]
        accumulate(accumulator, gradient, 2)
        accumulate(accumulator, gradient, 2)
        torch.testing.assert_close(accumulator[0], gradient[0])
        self.assertIsNone(x.grad)

    def test_policy_score_gradient_matches_analytic_formula(self):
        scores = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
        weights = F.one_hot(torch.tensor([1, 0]), 3).float()
        context = torch.tensor([[2., 0.], [0., 2.]])
        no_context = torch.tensor([[0., 2.], [1., 0.]])
        targets = torch.tensor([0, 0])
        output = TACSOutput(context, no_context, scores, weights, .1)
        loss = TACSLoss()(output, targets)
        actual = torch.autograd.grad(loss.policy, scores)[0]
        reward = F.cross_entropy(no_context, targets, reduction='none') - F.cross_entropy(context, targets, reduction='none')
        expected = standardize_reward(reward)[:, None] * (scores.softmax(-1) - weights) / 2
        torch.testing.assert_close(actual, expected)

    def test_freezing_task_does_not_cut_selector_gradient_or_mutate_state(self):
        torch.manual_seed(19)
        model = TACSModel(3, 8, temperature=1., detach_candidate_embeddings=True).train()
        frozen = copy.deepcopy(model)
        frozen.task_network.requires_grad_(False)
        query, context = torch.randn(2, 2, 3, 32, 32)
        bank = torch.randn(5, 8)
        targets = torch.tensor([0, 2])
        before = state_digest(frozen)
        gradients = []
        for network in (model, frozen):
            torch.manual_seed(31)
            prediction = network.forward_embedding_bank(query, bank, context)
            loss = TACSLoss()(prediction, targets)
            params = list(network.selector.parameters())
            task = cpu_gradients(loss.task, params, 1., True)
            policy = cpu_gradients(loss.policy, params, 1., False)
            self.assertGreater(gradient_pair_summary(task, policy)["task_l2"], 0)
            gradients.append(task)
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
        self.assertEqual(before, state_digest(frozen))
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))


if __name__ == '__main__':
    unittest.main()
