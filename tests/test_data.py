import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from tacs import TACSLoss, TACSModel
from tacs.data import (
    ContextDataset,
    SplitIndices,
    stratified_split,
)


class StratifiedSplitTest(unittest.TestCase):
    """Kiểm tra việc chia dữ liệu theo lớp."""

    def setUp(self) -> None:
        # 10 lớp, mỗi lớp có 100 mẫu.
        # Tổng cộng có 1.000 mẫu.
        self.labels = [
            label
            for label in range(10)
            for _ in range(100)
        ]

    def test_split_is_balanced_and_reproducible(
        self,
    ) -> None:
        first = stratified_split(
            labels=self.labels,
            validation_size=100,
            candidate_ratio=0.20,
            seed=42,
        )

        second = stratified_split(
            labels=self.labels,
            validation_size=100,
            candidate_ratio=0.20,
            seed=42,
        )

        # 1.000 ảnh:
        # 100 validation
        # 20% của 900 = 180 candidate
        # Còn lại 720 query
        self.assertEqual(
            len(first.query_train),
            720,
        )

        self.assertEqual(
            len(first.candidate_pool),
            180,
        )

        self.assertEqual(
            len(first.validation),
            100,
        )

        # Hàm validate kiểm tra trùng, thiếu
        # và index ngoài phạm vi.
        first.validate(
            dataset_size=len(self.labels)
        )

        # Cùng seed phải cho cùng split.
        self.assertEqual(first, second)

        # Kiểm tra từng tập vẫn cân bằng lớp.
        for indices in (
            first.query_train,
            first.candidate_pool,
            first.validation,
        ):
            class_counts = [
                sum(
                    self.labels[index] == label
                    for index in indices
                )
                for label in range(10)
            ]

            self.assertEqual(
                len(set(class_counts)),
                1,
            )

    def test_split_save_and_load(self) -> None:
        original = stratified_split(
            labels=self.labels,
            validation_size=100,
            candidate_ratio=0.20,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as directory:
            split_path = (
                Path(directory)
                / "splits"
                / "test_split.json"
            )

            original.save(split_path)

            loaded = SplitIndices.load(split_path)

            self.assertTrue(split_path.exists())
            self.assertEqual(original, loaded)

            loaded.validate(
                dataset_size=len(self.labels)
            )


class ContextDatasetTest(unittest.TestCase):
    """Kiểm tra candidate sampling và tích hợp TACS."""

    def setUp(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(42)

        images = torch.randn(
            30,
            3,
            32,
            32,
            generator=generator,
        )

        labels = torch.arange(30) % 10

        self.source = TensorDataset(
            images,
            labels,
        )

        # 20 ảnh đầu làm query.
        self.query_dataset = Subset(
            self.source,
            list(range(20)),
        )

        # 10 ảnh cuối làm candidate pool.
        self.candidate_dataset = Subset(
            self.source,
            list(range(20, 30)),
        )

    def test_sampling_is_reproducible(
        self,
    ) -> None:
        dataset = ContextDataset(
            query_dataset=self.query_dataset,
            candidate_dataset=self.candidate_dataset,
            num_candidates=8,
            seed=42,
        )

        first = dataset[0]
        second = dataset[0]

        # Cùng query và cùng epoch phải có
        # cùng candidate indices.
        self.assertTrue(
            torch.equal(
                first["candidate_indices"],
                second["candidate_indices"],
            )
        )

        query_index = first[
            "query_index"
        ].item()

        self.assertNotIn(
            query_index,
            first["candidate_indices"].tolist(),
        )

        # Sang epoch mới, candidates phải đổi.
        dataset.set_epoch(1)

        third = dataset[0]

        self.assertFalse(
            torch.equal(
                first["candidate_indices"],
                third["candidate_indices"],
            )
        )

    def test_batch_integrates_with_tacs(
        self,
    ) -> None:
        dataset = ContextDataset(
            query_dataset=self.query_dataset,
            candidate_dataset=self.candidate_dataset,
            num_candidates=8,
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        self.assertEqual(
            batch["query"].shape,
            (4, 3, 32, 32),
        )

        self.assertEqual(
            batch["candidates"].shape,
            (4, 8, 3, 32, 32),
        )

        self.assertEqual(
            batch["target"].shape,
            (4,),
        )

        model = TACSModel(
            num_classes=10,
            embedding_dim=16,
        )

        output = model(
            batch["query"],
            batch["candidates"],
        )

        self.assertEqual(
            output.context_logits.shape,
            (4, 10),
        )

        loss = TACSLoss()(
            output,
            batch["target"],
        )

        loss.total.backward()

        selector_gradients = [
            parameter.grad
            for parameter
            in model.selector.parameters()
            if parameter.grad is not None
        ]

        self.assertTrue(selector_gradients)

        total_gradient = sum(
            gradient.abs().sum().item()
            for gradient in selector_gradients
        )

        self.assertGreater(
            total_gradient,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()