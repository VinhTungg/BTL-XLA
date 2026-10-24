import unittest

import torch

from tacs import DINOv3VisionEncoder, TACSLoss, TACSModel


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

    def test_full_pool_embedding_bank_scores_every_candidate(self) -> None:
        self.model.eval()
        full_pool = torch.randn(11, 16)

        output = self.model.forward_embedding_bank(self.query, full_pool)

        self.assertEqual(output.selector_scores.shape, (3, 11))
        self.assertEqual(output.selection_weights.shape, (3, 11))
        self.assertTrue(torch.equal(output.selection_weights.sum(dim=1), torch.ones(3)))
        self.assertEqual(output.selector_temperature, 0.1)

    def test_detached_candidate_path_keeps_query_encoder_trainable(self) -> None:
        model = TACSModel(
            num_classes=4,
            embedding_dim=16,
            detach_candidate_embeddings=True,
        )
        candidates = self.candidates.clone().requires_grad_(True)

        TACSLoss()(model(self.query, candidates), self.targets).total.backward()

        self.assertIsNone(candidates.grad)
        encoder_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.selector.encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)

    def test_dinov3_vits16_encoder_contract_without_downloading_weights(self) -> None:
        encoder = DINOv3VisionEncoder(pretrained=False).eval()
        with torch.no_grad():
            embeddings = encoder(torch.randn(1, 3, 256, 256))

        self.assertEqual(embeddings.shape, (1, 384))
        self.assertGreater(sum(p.numel() for p in encoder.parameters()), 21_000_000)


if __name__ == "__main__":
    unittest.main()
