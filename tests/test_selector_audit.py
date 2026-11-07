"""Unit fixtures only: these are NOT experiment data or reported accuracy."""

import copy
import unittest
from dataclasses import asdict

import torch

from tacs.model import TACSModel
from tacs.selector_audit import cosine_rescore, norm_diagnostics, summarize, validate_provenance
from tacs.train import ExperimentConfig, _selected_pool_positions


class SelectorAuditTest(unittest.TestCase):
    def provenance(self):
        cfg = asdict(ExperimentConfig(dataset="cifar100", baseline="tacs"))
        sizes = dict(query_train=36000, candidate_pool=9000, validation=5000)
        payload = {"config": cfg, "model_state_dict": {}, "val_accuracy": 0.8}
        metrics = {"config": copy.deepcopy(cfg), "data": {
            "source": "cifar100", "split_sha256": "hash", "full_sizes": sizes,
            "effective_sizes": sizes.copy()}}
        return payload, metrics

    def test_valid_provenance(self):
        p, m = self.provenance()
        self.assertEqual(validate_provenance(p, m, "hash").baseline, "tacs")

    def test_reject_wrong_split(self):
        p, m = self.provenance()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_provenance(p, m, "wrong")

    def test_reject_different_run_config(self):
        p, m = self.provenance()
        m["config"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "match exactly"):
            validate_provenance(p, m, "hash")

    def test_reject_legacy_weights(self):
        p, m = self.provenance()
        p["config"]["task_network_version"] = "legacy_v1"
        m["config"] = p["config"].copy()
        with self.assertRaisesRegex(ValueError, "role_v2"):
            validate_provenance(p, m, "hash")

    def test_reject_subset_training(self):
        p, m = self.provenance()
        m["data"]["effective_sizes"]["validation"] = 8
        with self.assertRaisesRegex(ValueError, "size provenance"):
            validate_provenance(p, m, "hash")

    def test_reject_missing_accuracy(self):
        p, m = self.provenance()
        del p["val_accuracy"]
        with self.assertRaisesRegex(ValueError, "val_accuracy"):
            validate_provenance(p, m, "hash")

    def test_paired_summary_averages_trials_within_query(self):
        rows = []
        for i in range(2):
            rows.append(dict(selected_correct=True, selected_ce=0.2,
                             no_context_correct=(i == 0), no_context_ce=0.6,
                             random_0_correct=False, random_0_ce=0.8,
                             random_1_correct=True, random_1_ce=0.4,
                             selected_pool_position=7, selected_label=2, target=2,
                             policy_entropy=0.5, top1_probability=0.9))
        report = summarize(rows, 2)
        self.assertEqual(report["selected_vs_random"]["accuracy_delta_pp"], 50)
        self.assertAlmostEqual(report["selected_vs_random"]["mean_ce_reduction"], 0.4)
        self.assertEqual(report["selected_vs_no_context"]["queries_with_lower_ce"], 2)
        self.assertEqual(report["selection"]["unique_contexts"], 1)
        self.assertEqual(report["selection"]["top1_share"], 1)

    def test_empty_summary_fails(self):
        with self.assertRaises(ValueError):
            summarize([], 1)

    def test_cosine_removes_positive_candidate_scaling_bias(self):
        query = torch.tensor([[1., 0.]])
        bank = torch.tensor([[10., 10.], [1., 0.]])
        dot, cosine = cosine_rescore(query, bank)
        self.assertEqual(dot.tolist(), [0])
        self.assertEqual(cosine.tolist(), [1])
        _, scaled = cosine_rescore(query * 7, bank * torch.tensor([[.01], [50.]]))
        self.assertEqual(scaled.tolist(), cosine.tolist())

    def test_cosine_per_query_bank_and_single_candidate(self):
        query = torch.tensor([[1., 0.], [0., 1.]])
        bank = torch.tensor([[[1., 0.]], [[0., 2.]]])
        dot, cosine = cosine_rescore(query, bank)
        self.assertEqual(dot.tolist(), [0, 0])
        self.assertEqual(cosine.tolist(), [0, 0])

    def test_cosine_rejects_zero_and_nonfinite_features(self):
        for bank in (torch.zeros(1, 2), torch.tensor([[float('nan'), 1.]])):
            with self.assertRaises(ValueError):
                cosine_rescore(torch.ones(1, 2), bank)

    def test_norms_measure_eligible_percentile_and_exposure(self):
        report = norm_diagnostics([1., 2., 3., 4.], [1, 3], [[0, 1], [2, 3]])
        self.assertEqual(report["mean_selected_norm"], 3.)
        self.assertEqual(report["mean_uniform_eligible_norm"], 2.5)
        self.assertEqual(report["mean_selected_norm_percentile_within_eligible"], 1.)
        self.assertEqual(report["exposed_candidates"], 4)

    def test_norm_constant_correlation_is_none(self):
        report = norm_diagnostics([1., 1.], [0, 0])
        self.assertIsNone(report["norm_vs_selection_count_pearson"])
        self.assertEqual(report["mean_selected_norm_percentile_within_eligible"], 1.)

    def test_topk_exposure_changes_rate_correlation(self):
        report = norm_diagnostics([1., 2., 3.], [0, 2], [[0, 1], [0, 2]])
        self.assertAlmostEqual(report["norm_vs_selection_count_pearson"], 0.)
        self.assertAlmostEqual(report["norm_vs_selection_rate_given_exposure_pearson"], .5)

    def test_norm_rejects_selection_outside_topk(self):
        with self.assertRaisesRegex(ValueError, "outside eligible"):
            norm_diagnostics([1., 2.], [1], [[0]])

    def test_extra_strategies_do_not_contaminate_random_average(self):
        row = dict(selected_correct=True, selected_ce=.2, no_context_correct=False,
                   no_context_ce=.6, random_0_correct=False, random_0_ce=.8,
                   selected_pool_position=0, selected_label=2, target=2,
                   policy_entropy=.5, top1_probability=.9,
                   selector_cosine_correct=True, selector_cosine_ce=.1,
                   selector_cosine_pool_position=1, selector_cosine_label=2)
        report = summarize([row], 1)
        self.assertEqual(report["selected_vs_random"]["accuracy_delta_pp"], 100.)
        self.assertAlmostEqual(report["selected_vs_random"]["mean_ce_reduction"], .6)
        self.assertEqual(report["selector_cosine"]["selection"]["same_as_original_selector_fraction"], 0.)

    def test_topk_local_positions_map_to_global_pool(self):
        batch = {"candidate_positions": torch.tensor([[91, 13], [2, 44]])}
        weights = torch.tensor([[0., 1.], [1., 0.]])
        self.assertEqual(_selected_pool_positions(batch, weights).tolist(), [13, 2])

    def test_single_candidate_topk(self):
        batch = {"candidate_positions": torch.tensor([[91], [44]])}
        self.assertEqual(_selected_pool_positions(batch, torch.ones(2, 1)).tolist(), [91, 44])

    def test_reject_malformed_payload(self):
        with self.assertRaisesRegex(ValueError, "dictionaries"):
            validate_provenance([], {}, "hash")

    def test_direct_task_inference_matches_production_zero_proxy(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(5)
            model = TACSModel(num_classes=3, embedding_dim=8).eval()
            query, context = torch.randn(2, 2, 3, 32, 32)
            bank = torch.randn(4, 8)
            with torch.no_grad():
                actual = model.forward_embedding_bank(query, bank, context)
                direct = model.task_network(query, context)
            torch.testing.assert_close(actual.context_logits, direct, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
