import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from tacs.model import TinyVisionEncoder
from tacs.retrieval import (
    CosineSimilarityRetriever,
    SimilarityContextDataset,
    extract_normalized_features,
    precompute_candidate_embeddings,
)


class RetrievalTest(unittest.TestCase):
    """Kiểm tra module trích xuất embedding và Cosine Similarity Retrieval (Gói 4)."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.num_candidates = 20
        self.embedding_dim = 16
        self.encoder = TinyVisionEncoder(embedding_dim=self.embedding_dim)
        self.encoder.eval()

        images = torch.randn(self.num_candidates, 3, 32, 32)
        targets = torch.randint(0, 10, (self.num_candidates,))
        self.candidate_dataset = TensorDataset(images, targets)

    def test_precompute_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "candidate_embeddings.pt"
            res = precompute_candidate_embeddings(
                candidate_dataset=self.candidate_dataset,
                encoder=self.encoder,
                batch_size=8,
                device="cpu",
                output_path=cache_file,
            )

            self.assertTrue(cache_file.exists())
            self.assertEqual(res["embeddings"].shape, (self.num_candidates, self.embedding_dim))
            self.assertEqual(len(res["indices"]), self.num_candidates)

            # Kiểm tra các vector đã được chuẩn hóa L2
            norms = torch.norm(res["embeddings"], p=2, dim=-1)
            self.assertTrue(torch.allclose(norms, torch.ones(self.num_candidates), atol=1e-5))

    def test_cosine_similarity_retrieval_matches_manual_dot_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cand.pt"
            res = precompute_candidate_embeddings(
                candidate_dataset=self.candidate_dataset,
                encoder=self.encoder,
                output_path=cache_file,
            )
            retriever = CosineSimilarityRetriever.from_cache(cache_file)

            # Tạo 3 query embeddings ngẫu nhiên đã chuẩn hóa L2
            q_emb = F.normalize(torch.randn(3, self.embedding_dim), dim=-1)
            scores, positions, indices = retriever.retrieve_top1(q_emb)

            # Tính toán thủ công bằng dot product
            cand_embs = res["embeddings"]  # [20, 16]
            manual_sim = torch.matmul(q_emb, cand_embs.T)  # [3, 20]
            manual_best_scores, manual_best_pos = torch.max(manual_sim, dim=-1)

            self.assertTrue(torch.allclose(scores, manual_best_scores, atol=1e-5))
            self.assertTrue(torch.equal(positions, manual_best_pos))

    def test_similarity_context_dataset_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cand.pt"
            precompute_candidate_embeddings(
                candidate_dataset=self.candidate_dataset,
                encoder=self.encoder,
                output_path=cache_file,
            )
            retriever = CosineSimilarityRetriever.from_cache(cache_file, candidate_dataset=self.candidate_dataset)

            query_images = torch.randn(10, 3, 32, 32)
            query_targets = torch.randint(0, 10, (10,))
            query_ds = TensorDataset(query_images, query_targets)

            sim_dataset = SimilarityContextDataset(
                query_dataset=query_ds,
                candidate_dataset=self.candidate_dataset,
                retriever=retriever,
                encoder=self.encoder,
            )

            self.assertEqual(len(sim_dataset), 10)
            sample = sim_dataset[0]

            self.assertIn("query", sample)
            self.assertIn("candidates", sample)
            self.assertIn("target", sample)
            self.assertIn("similarity_scores", sample)
            self.assertEqual(sample["candidates"].shape, (1, 3, 32, 32))
            self.assertTrue(0.0 <= sample["similarity_scores"].item() <= 1.0 or -1.0 <= sample["similarity_scores"].item() <= 1.0)


if __name__ == "__main__":
    unittest.main()
