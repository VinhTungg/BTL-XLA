import unittest

import torch

from tacs import TACSLoss, TACSModel


class TACSModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = TACSModel(num_classes=4, embedding_dim=16)
        self.query = torch.randn(3, 3, 24, 24)
        self.candidates = torch.randn(3, 6, 3, 24, 24)
        self.targets = torch.tensor([0, 1, 2])

    def test_forward_shapes_and_hard_selection(self) -> None:
        output = self.model(self.query, self.candidates)
        self.assertEqual(output.context_logits.shape, (3, 4))
        self.assertEqual(output.selector_scores.shape, (3, 6))
        self.assertTrue(torch.equal(output.selection_weights.sum(dim=1), torch.ones(3)))

    def test_hybrid_loss_backpropagates_to_selector(self) -> None:
        loss = TACSLoss()(self.model(self.query, self.candidates), self.targets)
        loss.total.backward()
        gradients = [p.grad for p in self.model.selector.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.0)

    def test_eval_selection_is_deterministic(self) -> None:
        self.model.eval()
        first = self.model(self.query, self.candidates).selection_weights
        second = self.model(self.query, self.candidates).selection_weights
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()

