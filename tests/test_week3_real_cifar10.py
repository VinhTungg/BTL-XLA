import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from tacs.data import ContextDataset, build_cifar10
from tacs.losses import TACSLoss
from tacs.model import TACSModel, TinyVisionEncoder
from tacs.retrieval import (
    CosineSimilarityRetriever,
    TopKContextDataset,
    precompute_candidate_embeddings,
)
from tacs.train import evaluate, train_one_epoch


DATA_ROOT = Path(os.environ.get("TACS_DATA_ROOT", "data"))
SPLIT_FILE = Path(os.environ.get("TACS_SPLIT_FILE", "data/splits/cifar10_seed42.json"))
REAL_CIFAR_AVAILABLE = (
    (DATA_ROOT / "cifar-10-batches-py" / "data_batch_1").is_file()
    and (DATA_ROOT / "cifar-10-batches-py" / "test_batch").is_file()
    and SPLIT_FILE.is_file()
)
RUN_REAL_CIFAR_TESTS = os.environ.get("TACS_RUN_REAL_CIFAR_TESTS") == "1"


@unittest.skipUnless(
    RUN_REAL_CIFAR_TESTS and REAL_CIFAR_AVAILABLE,
    "Set TACS_RUN_REAL_CIFAR_TESTS=1 and provide local CIFAR-10 data plus split.",
)
class Week3RealCIFAR10IntegrationTest(unittest.TestCase):
    def test_tacs_and_two_stage_train_on_real_cifar10(self) -> None:
        datasets = build_cifar10(
            root=DATA_ROOT,
            split_file=SPLIT_FILE,
            download=False,
        )

        self.assertEqual(len(datasets.query_train), 36_000)
        self.assertEqual(len(datasets.candidate_pool), 9_000)
        self.assertEqual(len(datasets.validation), 5_000)
        self.assertEqual(len(datasets.test), 10_000)
        self.assertEqual(datasets.candidate_pool.dataset.__class__.__name__, "CIFAR10")

        query_subset = Subset(datasets.query_train, [0, 1])
        validation_subset = Subset(datasets.validation, [0, 1])
        encoder = TinyVisionEncoder(embedding_dim=8)
        criterion = TACSLoss()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "candidate_embeddings.pt"
            precompute_candidate_embeddings(
                candidate_dataset=datasets.candidate_pool,
                encoder=encoder,
                batch_size=128,
                device="cpu",
                output_path=cache_path,
            )
            retriever = CosineSimilarityRetriever.from_cache(
                cache_path,
                candidate_dataset=datasets.candidate_pool,
            )

            tacs_train = ContextDataset(
                query_dataset=query_subset,
                candidate_dataset=datasets.candidate_pool,
                num_candidates=4,
                seed=42,
            )
            tacs_validation = ContextDataset(
                query_dataset=validation_subset,
                candidate_dataset=datasets.candidate_pool,
                num_candidates=4,
                seed=42,
            )
            two_stage_train = TopKContextDataset(
                query_dataset=query_subset,
                candidate_dataset=datasets.candidate_pool,
                retriever=retriever,
                encoder=encoder,
                device="cpu",
                batch_size=2,
                top_k=4,
            )
            two_stage_validation = TopKContextDataset(
                query_dataset=validation_subset,
                candidate_dataset=datasets.candidate_pool,
                retriever=retriever,
                encoder=encoder,
                device="cpu",
                batch_size=2,
                top_k=4,
            )

            for name, train_dataset, val_dataset in (
                ("tacs", tacs_train, tacs_validation),
                ("two_stage", two_stage_train, two_stage_validation),
            ):
                with self.subTest(model=name):
                    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
                    model = TACSModel(num_classes=10, embedding_dim=8)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                    train_metrics = train_one_epoch(
                        model, train_loader, optimizer, criterion, torch.device("cpu")
                    )
                    val_metrics = evaluate(
                        model, val_loader, criterion, torch.device("cpu")
                    )

                    self.assertGreater(train_metrics["loss"], 0)
                    self.assertGreater(val_metrics["loss"], 0)
                    self.assertGreater(train_metrics["throughput"], 0)
                    self.assertGreater(val_metrics["throughput"], 0)

            two_stage_batch = next(iter(DataLoader(two_stage_train, batch_size=2)))
            scores = two_stage_batch["similarity_scores"]
            self.assertEqual(tuple(two_stage_batch["candidates"].shape), (2, 4, 3, 32, 32))
            self.assertTrue(torch.all(scores[:, :-1] >= scores[:, 1:]).item())


if __name__ == "__main__":
    unittest.main()
