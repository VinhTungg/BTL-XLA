import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from tacs.visualize_batch import make_context_grid


class VisualizeBatchTest(unittest.TestCase):
    """Kiểm tra hiển thị query và candidate pool."""

    def test_make_context_grid_returns_query_and_candidate_axes(self) -> None:
        images = torch.randn(12, 3, 32, 32)
        labels = torch.arange(12) % 10
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(loader))
        query_images = batch[0][:2]
        candidate_images = torch.stack(
            [torch.randn(8, 3, 32, 32) for _ in range(2)],
            dim=0,
        )
        fake_batch = {
            "query": query_images,
            "candidates": candidate_images,
        }

        fig = make_context_grid(fake_batch, num_queries=2)

        self.assertEqual(len(fig.axes), 2 * (8 + 1))

        for axis in fig.axes:
            self.assertFalse(axis.has_data() is False and axis.get_visible() is False)

        self.assertIn("query", fig.axes[0].get_title().lower())


if __name__ == "__main__":
    unittest.main()
