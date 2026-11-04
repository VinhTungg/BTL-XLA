"""Regression checks for DINOv3 query/context roles and checkpoint isolation."""

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from tacs.model import DINOv3PatchTokenTaskNetwork, create_task_network
from tacs.train import (
    ExperimentConfig,
    _restore_training_checkpoint,
    _validate_output_architecture,
    build_optimizer_and_scheduler,
    run_experiment,
    train_one_epoch,
)


class ContextRoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_roles_break_exchange_symmetry_and_preserve_query_only(self) -> None:
        torch.manual_seed(44)
        legacy = DINOv3PatchTokenTaskNetwork(4, False, "legacy_v1").eval()
        legacy_rng_state = torch.get_rng_state()
        # timm's untrained LayerScale=1e-5 nearly ignores all patch content.
        # Use non-degenerate shared fixture weights without downloading a model.
        with torch.no_grad():
            for block in legacy.backbone.blocks:
                block.gamma_1.fill_(0.1)
                block.gamma_2.fill_(0.1)
        torch.manual_seed(44)
        fixed = DINOv3PatchTokenTaskNetwork(4, False, "role_v2").eval()
        self.assertTrue(torch.equal(legacy_rng_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(legacy.head.weight, fixed.head.weight))
        # Diagnostic comparison only: equal legacy weights, one new role vector.
        fixed.load_state_dict({**fixed.state_dict(), **legacy.state_dict()}, strict=True)
        self.assertNotIn("context_role_embedding", legacy.state_dict())
        self.assertEqual(tuple(fixed.context_role_embedding.shape), (1, 1, 384))
        query, context = torch.randn(2, 1, 3, 256, 256)
        with torch.no_grad():
            old_delta = (legacy(query, context) - legacy(context, query)).abs().max().item()
            new_delta = (fixed(query, context) - fixed(context, query)).abs().max().item()
            self.assertLess(old_delta, 1e-5)
            self.assertGreater(new_delta, 1e-4)
            self.assertTrue(torch.equal(legacy(query), fixed(query)))
            self.assertTrue(torch.isfinite(fixed(query, query)).all())

        # The role is learnable and does not cut the existing straight-through proxy.
        proxy = torch.zeros(1, 384, requires_grad=True)
        loss = nn.functional.cross_entropy(fixed(query, context, proxy), torch.tensor([1]))
        loss.backward()
        role_gradient = fixed.context_role_embedding.grad
        self.assertIsNotNone(role_gradient)
        self.assertTrue(torch.isfinite(role_gradient).all())
        self.assertGreater(role_gradient.abs().sum().item(), 0)
        self.assertGreater(proxy.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None for p in fixed.backbone.parameters()))

    def test_invalid_role_version_and_context_shapes_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_network_version"):
            create_task_network("tiny", 4, task_network_version="typo")
        model = DINOv3PatchTokenTaskNetwork(4, False)
        query = torch.randn(1, 3, 256, 256)
        with self.assertRaisesRegex(ValueError, "same.*shape"):
            model(query, query.expand(2, -1, -1, -1))
        with self.assertRaisesRegex(ValueError, "same.*shape"):
            model(query, query[:, :, :128, :128])
        with self.assertRaisesRegex(ValueError, "requires a context"):
            model(query, context_proxy=torch.zeros(1, 384))
        with self.assertRaisesRegex(ValueError, "context_proxy must have shape"):
            model(query, query, torch.zeros(1, 4))

    def test_legacy_config_load_and_default_output_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"backbone": "dinov3_vits16", "dataset": "cifar100"}), encoding="utf-8")
            old = ExperimentConfig.load(path)
            new = ExperimentConfig(backbone="dinov3_vits16", dataset="cifar100")
            self.assertEqual(old.task_network_version, "legacy_v1")
            self.assertNotEqual(old.output_dir, new.output_dir)
            self.assertIn("role_v2", new.standard_encoder_path)
        with self.assertRaisesRegex(ValueError, "task_network_version"):
            ExperimentConfig(task_network_version="typo")

    def test_cross_version_output_rejected_before_data_setup_even_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            original = '{"backbone":"dinov3_vits16"}'
            path.write_text(original, encoding="utf-8")
            config = ExperimentConfig(backbone="dinov3_vits16", output_dir=temp, resume=False)
            with patch("tacs.train.build_dataloaders") as build:
                with self.assertRaisesRegex(RuntimeError, "legacy_v1"):
                    run_experiment(config)
                build.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            # Missing config.json does not let an old checkpoint bypass the guard.
            path.unlink()
            torch.save({"config": {}}, Path(temp) / "checkpoint_latest.pt")
            with self.assertRaisesRegex(RuntimeError, "legacy_v1"):
                _validate_output_architecture(config)

    def test_resume_missing_version_requires_legacy_and_does_not_mutate_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            legacy_config = ExperimentConfig(task_network_version="legacy_v1")
            config_dict = asdict(legacy_config)
            config_dict.pop("task_network_version")
            model = nn.Linear(2, 2)
            optimizer, scheduler = build_optimizer_and_scheduler(model, legacy_config)
            checkpoint = {
                "config": config_dict,
                "epoch": 2,
                "val_accuracy": 0.5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            path = Path(temp) / "checkpoint_latest.pt"
            torch.save(checkpoint, path)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            weights = model.weight.detach().clone()
            with self.assertRaisesRegex(RuntimeError, "task_network_version"):
                _restore_training_checkpoint(path, model, optimizer, scheduler, scaler,
                                             ExperimentConfig(), torch.device("cpu"))
            self.assertTrue(torch.equal(weights, model.weight))
            epoch, acc, _, _ = _restore_training_checkpoint(
                path, model, optimizer, scheduler, scaler, legacy_config, torch.device("cpu")
            )
            self.assertEqual((epoch, acc), (3, 0.5))


@unittest.skipUnless(
    os.environ.get("TACS_RUN_REAL_CIFAR_TESTS") == "1"
    and Path("data/cifar-100-python/train").is_file()
    and Path("data/splits/cifar100_seed42.json").is_file(),
    "Enable TACS_RUN_REAL_CIFAR_TESTS=1 with local CIFAR-100 to test DINOv3 resume.",
)
class RealDINOv3ResumeTest(unittest.TestCase):
    def test_role_v2_trains_and_resumes_after_interruption_on_real_images(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = ExperimentConfig(
                    baseline="random", dataset="cifar100", backbone="dinov3_vits16",
                    pretrained_backbone=False, epochs=2, batch_size=2, seed=44,
                    device="cpu", output_dir=temp, lr=1e-4,
                    max_train_samples=2, max_val_samples=2, eval_test=False,
                )
                calls = 0

                def interrupt_second_epoch(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("Simulated interruption before epoch 2")
                    return train_one_epoch(*args, **kwargs)

                with patch("tacs.train.train_one_epoch", side_effect=interrupt_second_epoch):
                    with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                        run_experiment(config)
                path = Path(temp) / "checkpoint_latest.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertIn("task_network.context_role_embedding", checkpoint["model_state_dict"])
                del checkpoint
                result = run_experiment(config)
                self.assertEqual(result["training_recipe"]["resumed_from_epoch"], 1)
                self.assertEqual([row["epoch"] for row in result["history"]], [1, 2])
                self.assertEqual(result["model_architecture"]["task_network_version"], "role_v2")
                self.assertIsNone(result["test_result"])
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epoch"], 2)
                self.assertEqual(checkpoint["config"]["task_network_version"], "role_v2")
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
