"""Unit fixtures only; runtime showcase never substitutes artificial data."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tacs.showcase.__main__ import (
    PRESETS, ROOT, external_input, make_report, settings_for, validation_input, write_report,
)
from tacs.showcase.explain import (
    HOW_IT_WORKS_TEXT,
    SCORING_MODES,
    SCORING_MODE_DETAILS,
    candidate_badges,
    candidate_rank_text,
    pipeline_card_texts,
    step_narration,
    strategy_group,
    summarize_outcome,
    top_stage_candidates,
)


class ShowcaseTest(unittest.TestCase):
    def fixture(self, root, scope='microbatch'):
        path = Path(root)/'artifacts/effective_batch_pilot'/PRESETS[scope][0]
        path.mkdir(parents=True)
        manifest = dict(kind='advantage_normalization_pilot', test_evaluated=False,
                        config=dict(advantage_scope=scope, seed=44, epochs=20, eval_test=False))
        result = dict(status='complete', advantage_scope=scope, epochs=20, training_seed=44,
                      test_evaluated=False, checkpoint_sha256=PRESETS[scope][1])
        (path/'experiment_manifest.json').write_text(json.dumps(manifest))
        (path/'result.json').write_text(json.dumps(result))
        return path, manifest, result

    def test_root_and_unknown_preset(self):
        self.assertTrue((ROOT/'pyproject.toml').is_file())
        with self.assertRaisesRegex(ValueError, 'Preset'):
            settings_for('unknown')

    def test_presets_have_separate_checkpoints_and_same_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture(temp)
            self.fixture(temp, 'effective_batch')
            a, b = settings_for('microbatch', temp), settings_for('effective_batch', temp)
            self.assertNotEqual(a.pop('checkpoint'), b.pop('checkpoint'))
            self.assertEqual(a, b)

    def test_wrong_scope_seed_or_test_manifest_refused(self):
        for key, value in [('advantage_scope', 'effective_batch'), ('seed', 42),
                           ('epochs', 100), ('eval_test', True)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                path, manifest, _ = self.fixture(temp)
                manifest['config'][key] = value
                (path/'experiment_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'manifest'):
                    settings_for('microbatch', temp)

    def test_wrong_hash_or_incomplete_result_refused(self):
        for key, value in [('checkpoint_sha256', 'changed'), ('status', 'partial'),
                           ('test_evaluated', True), ('training_seed', 43)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                path, _, result = self.fixture(temp)
                result[key] = value
                (path/'result.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, 'manifest'):
                    settings_for('microbatch', temp)

    def test_external_rgba_and_unknown_label(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'input.png'
            Image.new('RGBA', (40, 50)).save(path)
            image, query = external_input(path)
            self.assertEqual(image.mode, 'RGB')
            self.assertEqual(query['original_size'], [40, 50])
            self.assertIsNone(query['target'])
            self.assertEqual(query['filename'], 'input.png')

    def test_external_corrupt_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.png'
            path.write_text('not an image')
            with self.assertRaises(OSError):
                external_input(path)

    def test_external_oversized_image_rejected_before_pixel_conversion(self):
        with patch('PIL.Image.open') as opened:
            opened.return_value.__enter__.return_value = SimpleNamespace(width=4001, height=4000)
            with self.assertRaisesRegex(ValueError, '16 megapixel'):
                external_input('unit-fixture-only.png')

    def test_report_carries_input_truth_and_does_not_mutate_metadata(self):
        engine = SimpleNamespace(metadata={'preset': 'microbatch'}, split=SimpleNamespace(validation=[7]),
                                 validation_image=lambda p: (Image.new('RGB', (32, 32)), 'apple'))
        engine.predict = lambda image: dict(metadata=engine.metadata, results={
            'selected': dict(prediction='apple'), 'no_context': dict(prediction='pear')})
        image, query = validation_input(engine, 0)
        result = make_report(engine, image, query)
        self.assertEqual(result['query']['cifar_index'], 7)
        self.assertEqual(len(result['query']['rgb_sha256']), 64)
        self.assertTrue(result['results']['selected']['correct'])
        self.assertFalse(result['results']['no_context']['correct'])
        result['metadata']['preset'] = 'different'
        self.assertEqual(engine.metadata['preset'], 'microbatch')
        query['target'] = None
        self.assertIsNone(make_report(engine, image, query)['results']['selected']['correct'])

    def test_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'new/result.json'
            write_report(path, {'status': 'first'})
            with self.assertRaises(FileExistsError):
                write_report(path, {'status': 'second'})
            self.assertEqual(json.loads(path.read_text())['status'], 'first')

    def test_headless_invalid_position_refused_before_loading_model(self):
        import sys
        from tacs.showcase.__main__ import main
        with patch.object(sys, 'argv', ['showcase', '--headless', '--validation-position', '5000']), \
                patch('tacs.showcase.__main__.create_engine') as create:
            with self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 1)
            create.assert_not_called()

    def test_explain_helpers_cover_help_hurt_external_and_stage_ranking(self):
        self.assertEqual(strategy_group('selected'), 'Cùng weights Two-Stage')
        self.assertEqual(strategy_group('trained_no_context'), 'Checkpoint riêng (100 ep)')
        self.assertEqual(candidate_rank_text({'stage1_rank': 14, 'stage2_rank': 1}), 'GĐ1 #14 -> GĐ2 #1')
        self.assertEqual(candidate_rank_text({'stage1_rank': None, 'stage2_rank': None}), '-')

        candidates = [
            dict(pool_position=10, dataset_index=100, label='guitar', cosine_score=0.91,
                 stage1_rank=1, selector_score=4.2, stage2_rank=2,
                 is_selected=False, is_frozen_top1=True, is_random=False),
            dict(pool_position=20, dataset_index=200, label='accordion', cosine_score=0.72,
                 stage1_rank=2, selector_score=8.5, stage2_rank=1,
                 is_selected=True, is_frozen_top1=False, is_random=True),
        ]
        report = dict(
            metadata=dict(candidate_count=606, num_classes=101),
            query=dict(target='electric_guitar', original_size=[300, 103]),
            retrieval_ms=12.0,
            selector_ms=8.0,
            pipeline=dict(pool_size=606, topk_count=2, selected_stage1_rank=2, random_stage1_rank=2, candidates=candidates),
            results=dict(
                selected=dict(prediction='electric_guitar', correct=True, context_label='accordion',
                              stage1_rank=2, stage2_rank=1, cosine_score=0.72, selector_score=8.5,
                              top5=[dict(label='electric_guitar', probability=0.55)]),
                frozen_cosine=dict(prediction='electric_guitar', correct=True, context_label='guitar',
                                   stage1_rank=1, stage2_rank=2, cosine_score=0.91, selector_score=4.2,
                                   top5=[dict(label='electric_guitar', probability=0.65)]),
                no_context=dict(prediction='mandolin', correct=False, context_label=None,
                                top5=[dict(label='mandolin', probability=0.52)]),
                trained_no_context=dict(prediction='electric_guitar', correct=True, context_label=None,
                                        top5=[dict(label='electric_guitar', probability=0.66)]),
            ),
        )
        tone, text = summarize_outcome(report)
        self.assertEqual(tone, 'help')
        self.assertIn('CÓ LỢI', text)
        self.assertIn('No-context train riêng', text)
        self.assertEqual(list(SCORING_MODES)[0], 'hybrid')
        for mode_key in ('hybrid', 'cosine'):
            report['scoring_mode'] = mode_key
            summary_txt = summarize_outcome(report)[1]
            self.assertNotIn('Chẩn đoán', summary_txt)
            self.assertNotIn('không train lại', summary_txt.lower())
        self.assertNotIn('chẩn đoán', HOW_IT_WORKS_TEXT.lower())
        self.assertNotIn('không train lại', HOW_IT_WORKS_TEXT.lower())
        for detail in SCORING_MODE_DETAILS.values():
            self.assertNotIn('chẩn đoán', detail.lower())
            self.assertNotIn('không train lại', detail.lower())

        s1, s2 = top_stage_candidates(report, limit=8)
        self.assertEqual([c['label'] for c in s1], ['guitar', 'accordion'])
        self.assertEqual([c['label'] for c in s2], ['accordion', 'guitar'])
        self.assertEqual(candidate_badges(s2[0])[0][0], 'GĐ2 CHỌN')

        cards = pipeline_card_texts(report, report['query'], report['metadata'], 'Caltech-101')
        self.assertEqual(len(cards), 4)
        self.assertIn('GĐ1 #2 -> #1', cards[2]['detail'])
        self.assertIn('ĐẢO HẠNG', step_narration(3, report, report['query'], 'Caltech-101'))

        report['results']['selected']['correct'] = False
        report['results']['no_context']['correct'] = True
        self.assertEqual(summarize_outcome(report)[0], 'hurt')

        report['query']['target'] = None
        report['results']['selected']['correct'] = None
        report['results']['no_context']['correct'] = None
        self.assertEqual(summarize_outcome(report)[0], 'warn')

    def test_demo_engine_predict_includes_stage_pipeline_without_changing_outputs(self):
        import torch
        from tacs.inference import DemoEngine

        engine = DemoEngine.__new__(DemoEngine)
        engine.device = torch.device('cpu')
        engine.amp = False
        engine.cfg = SimpleNamespace(topk_candidates=3)
        engine.transform = lambda img: torch.zeros(3, 4, 4)
        engine.dataset = SimpleNamespace(classes=['cat', 'dog', 'bird', 'fish', 'frog'], targets=[0, 1, 2, 0])
        engine.split = SimpleNamespace(candidate_pool=[0, 1, 2, 3])
        engine.pool = [(torch.zeros(3, 4, 4), i) for i in range(4)]
        engine.metadata = {'dataset': 'unit'}
        engine.standard = lambda q: torch.tensor([[1.0, 0.0]])
        engine.frozen = torch.tensor([[0.9, 0.1], [0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])
        engine.selector = torch.tensor([
            [1.0, 0.0],    # pool 0 (Stage1 #1, cos=0.99): aligned direction, norm=1
            [0.6, 0.8],    # pool 1 (Stage1 #3): partial direction, norm=1
            [5.0, 8.66],   # pool 2 (Stage1 #2): huge norm=10, lower cosine=0.5 with query [1,0]
            [0.0, 1.0],    # pool 3 (not in top-3)
        ])

        def fake_selector(q, bank):
            q_emb = torch.tensor([[1.0, 0.0]])
            scores = torch.einsum('bd,bnd->bn', q_emb, bank)
            idx = scores.argmax(dim=-1)
            weights = torch.nn.functional.one_hot(idx, num_classes=bank.shape[1]).float()
            return scores, weights, q_emb, torch.zeros(1, 2)

        engine.model = SimpleNamespace(
            selector=SimpleNamespace(forward_embedding_bank=fake_selector),
            task_network=lambda q, ctx=None: torch.tensor([[0.2, 2.0, 0.1, -0.5, 0.0]]),
        )
        # Under raw dot-product, pool 2 ('bird') wins because of its huge norm (10.0) despite lower cosine (0.5).
        out_dot = engine.predict(Image.new('RGB', (4, 4)), random_seed=44, scoring_mode='dot')
        self.assertEqual(out_dot['scoring_mode'], 'dot')
        self.assertEqual(out_dot['pipeline']['pool_size'], 4)
        self.assertEqual(out_dot['pipeline']['topk_count'], 3)
        self.assertEqual(out_dot['results']['selected']['context_label'], 'bird')
        self.assertEqual(out_dot['results']['selected']['stage1_rank'], 2)
        self.assertEqual(out_dot['results']['selected']['stage2_rank'], 1)
        self.assertAlmostEqual(out_dot['results']['selected']['norm_percentile'], 1.0)
        self.assertEqual(out_dot['results']['frozen_cosine']['stage1_rank'], 1)

        # Under selector cosine (L2-normalized), pool 0 ('cat') wins because norm bias is removed!
        out_cos = engine.predict(Image.new('RGB', (4, 4)), random_seed=44, scoring_mode='cosine')
        self.assertEqual(out_cos['scoring_mode'], 'cosine')
        self.assertEqual(out_cos['results']['selected']['context_label'], 'cat')
        self.assertEqual(out_cos['results']['selected']['stage1_rank'], 1)
        self.assertEqual(out_cos['results']['selected']['stage2_rank'], 1)
        self.assertAlmostEqual(out_cos['results']['selected']['selector_cosine_score'], 1.0, places=4)

        # Hybrid also picks 'cat' cleanly and rejects unknown scoring modes.
        out_hyb = engine.predict(Image.new('RGB', (4, 4)), random_seed=44, scoring_mode='hybrid')
        self.assertEqual(out_hyb['results']['selected']['context_label'], 'cat')
        with self.assertRaisesRegex(ValueError, 'scoring_mode'):
            engine.predict(Image.new('RGB', (4, 4)), scoring_mode='invalid')


if __name__ == '__main__':
    unittest.main()
