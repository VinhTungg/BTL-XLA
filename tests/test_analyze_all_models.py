import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analyze_all_models.py"
SPEC = importlib.util.spec_from_file_location("analyze_all_models", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def metric_payload(baseline: str) -> dict:
    sizes = {
        "query_train": 36_000,
        "candidate_pool": 9_000,
        "validation": 5_000,
        "test": 10_000,
    }
    selection = {
        "strategy": {
            "no_context": "no_context",
            "random": "random_context",
            "similarity": "cosine_top1",
            "tacs": "full_pool_tacs",
            "two_stage": "cosine_topk_then_tacs",
        }[baseline],
        "retrieval_candidates_scanned": (
            9_000 if baseline in {"similarity", "two_stage"} else 0
        ),
        "selector_candidates": {
            "no_context": 0,
            "random": 1,
            "similarity": 1,
            "tacs": 9_000,
            "two_stage": 32,
        }[baseline],
    }
    return {
        "metrics_schema_version": 3,
        "baseline": baseline,
        "seed": 42,
        "epochs": 5,
        "config": {
            "baseline": baseline,
            "batch_size": 64,
            "embedding_dim": 64,
            "num_classes": 10,
            "lr": 1e-3,
            "min_lr": 0.0,
            "weight_decay": 1e-4,
        },
        "runtime": {
            "device": "cpu",
            "torch_version": "test",
            "torch_num_threads": 4,
            "gpu_name": None,
        },
        "data": {
            "source": "cifar10",
            "split_sha256": "same-split",
            "full_sizes": sizes,
            "effective_sizes": sizes,
        },
        "selection": selection,
        "best_val_acc": 0.75,
        "selector_temperature": 0.1 if baseline in {"tacs", "two_stage"} else None,
        "policy_weight": 1.0 if baseline in {"tacs", "two_stage"} else None,
        "training_recipe": {
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "initial_lr": 1e-3,
            "minimum_lr": 0.0,
            "weight_decay": 1e-4,
            "scheduler_t_max": 5,
            "configured_epochs": 5,
            "executed_epochs": 5,
        },
        "data_setup_time_s": 1.0,
        "total_time_s": 20.0,
        "end_to_end_time_s": 21.0,
        "memory": {
            "peak_rss_mb": 512.0,
            "peak_vram_allocated_mb": None,
            "peak_vram_reserved_mb": None,
        },
        "test_result": {"accuracy": 0.72},
        "history": [
            {
                "val_acc": 0.75,
                "learning_rate": 1e-3,
                "next_learning_rate": 9e-4,
                "throughput_qps": 100.0,
                "val_throughput_qps": 120.0,
                "val_latency_ms_per_query": 8.3333,
                "train_peak_rss_mb": 510.0,
                "val_peak_rss_mb": 512.0,
            }
        ],
    }


class AnalyzeAllModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = {
            baseline: metric_payload(baseline) for baseline in ANALYZE.BASELINES
        }

    def test_validates_and_summarizes_all_five_models(self) -> None:
        ANALYZE.validate_comparison(self.metrics)
        rows = ANALYZE.build_rows(self.metrics)

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[3]["model"], "TACS Full")
        self.assertEqual(rows[3]["selector_candidates"], 9_000)
        self.assertEqual(rows[4]["retrieval_candidates_scanned"], 9_000)

    def test_rejects_mismatched_split_and_incomplete_runs(self) -> None:
        self.metrics["random"]["data"]["split_sha256"] = "other-split"
        with self.assertRaisesRegex(ANALYZE.BenchmarkValidationError, "split_sha256"):
            ANALYZE.validate_comparison(self.metrics)

        incomplete = metric_payload("tacs")
        incomplete["test_result"] = None
        with self.assertRaisesRegex(ANALYZE.BenchmarkValidationError, "test_result"):
            ANALYZE.validate_metric(incomplete, "tacs", require_complete=True)
        ANALYZE.validate_metric(incomplete, "tacs", require_complete=False)

    def test_writes_csv_json_and_markdown(self) -> None:
        rows = ANALYZE.build_rows(self.metrics)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ANALYZE.write_csv(rows, root / "benchmark.csv")
            ANALYZE.write_json(rows, root / "benchmark.json")
            ANALYZE.write_markdown(rows, root / "benchmark.md")

            self.assertIn("TACS Full", (root / "benchmark.csv").read_text("utf-8-sig"))
            self.assertIn("two_stage", (root / "benchmark.json").read_text("utf-8"))
            self.assertIn("Latency", (root / "benchmark.md").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
