import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tacs.baselines import BaselineModel
from tacs.train import ExperimentConfig, evaluate, run_experiment, train_one_epoch


class BaselineModelTest(unittest.TestCase):
    """Kiểm tra kiến trúc mạng và forward pass của BaselineModel."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.query = torch.randn(4, 3, 32, 32)
        self.candidates_5d = torch.randn(4, 1, 3, 32, 32)
        self.candidates_4d = torch.randn(4, 3, 32, 32)

    def test_forward_shapes_all_baselines(self) -> None:
        for b_type in ("no_context", "random", "similarity"):
            model = BaselineModel(num_classes=10, embedding_dim=32, baseline_type=b_type)
            logits = model(self.query, self.candidates_5d)
            self.assertEqual(logits.shape, (4, 10))

    def test_no_context_ignores_candidates(self) -> None:
        model = BaselineModel(num_classes=10, embedding_dim=32, baseline_type="no_context")
        model.eval()
        with torch.no_grad():
            out_with_cand = model(self.query, self.candidates_5d)
            out_without_cand = model(self.query, None)
        self.assertTrue(torch.allclose(out_with_cand, out_without_cand, atol=1e-6))

    def test_extract_embedding_is_l2_normalized(self) -> None:
        model = BaselineModel(num_classes=10, embedding_dim=32)
        embs = model.extract_embedding(self.query)
        norms = torch.norm(embs, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(4), atol=1e-5))


class TrainingInfrastructureTest(unittest.TestCase):
    """Kiểm tra hạ tầng huấn luyện, gradient, checkpoint và tính tái lập."""

    def test_config_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "test_cfg.json"
            orig = ExperimentConfig(baseline="random", epochs=3, lr=5e-4, seed=123)
            orig.save(cfg_path)
            loaded = ExperimentConfig.load(cfg_path)
            self.assertEqual(orig, loaded)

    def test_validation_does_not_update_gradients(self) -> None:
        model = BaselineModel(num_classes=5, embedding_dim=16)
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 5, (8,))
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        criterion = nn.CrossEntropyLoss()

        evaluate(model, loader, criterion, torch.device("cpu"))

        for param in model.parameters():
            self.assertIsNone(param.grad)

    def test_checkpoint_roundtrip_preserves_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "ckpt.pt"
            model1 = BaselineModel(num_classes=5, embedding_dim=16)
            torch.save({"model_state_dict": model1.state_dict()}, ckpt_path)

            model2 = BaselineModel(num_classes=5, embedding_dim=16)
            data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model2.load_state_dict(data["model_state_dict"])

            x = torch.randn(2, 3, 32, 32)
            model1.eval()
            model2.eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model1(x), model2(x), atol=1e-6))

    def test_dry_run_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out1 = Path(tmp_dir) / "run1"
            out2 = Path(tmp_dir) / "run2"

            cfg1 = ExperimentConfig(
                baseline="no_context",
                seed=42,
                dry_run=True,
                output_dir=str(out1),
            )
            cfg2 = ExperimentConfig(
                baseline="no_context",
                seed=42,
                dry_run=True,
                output_dir=str(out2),
            )

            res1 = run_experiment(cfg1)
            res2 = run_experiment(cfg2)

            self.assertEqual(res1["best_val_acc"], res2["best_val_acc"])
            self.assertEqual(res1["history"][0]["train_loss"], res2["history"][0]["train_loss"])


if __name__ == "__main__":
    unittest.main()
