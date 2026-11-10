import ast
from dataclasses import asdict
import importlib.util
import io
import json
import tempfile
from pathlib import Path
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from advantage_pilot import make_config
from build_advantage_notebooks import bundle_source, make_notebook
from compare_advantage_pilot import compare
from policy_weight_pilot import sha256


class AdvantagePilotTest(unittest.TestCase):
    def make_completed_fixture(self, path, scope):
        path.mkdir()
        (path/'run').mkdir()
        cfg = asdict(make_config('project', path, scope))
        manifest = dict(kind='advantage_normalization_pilot', config=cfg, source_tree_sha256='source',
                        source_bundle_sha256='bundle', split_sha256='split', encoder_sha256='encoder',
                        environment={}, initialization='fresh', test_evaluated=False)
        metrics = dict(config=cfg, history=[dict(epoch=i) for i in range(1, 21)], test_result=None, total_time_s=10)
        (path/'run/metrics.json').write_text(json.dumps(metrics))
        (path/'run/checkpoint_best.pt').write_bytes(b'unit fixture only')
        report = dict(status='complete', advantage_scope=scope, test_evaluated=False,
                      best_val_accuracy=.5, checkpoint_sha256=sha256(path/'run/checkpoint_best.pt'),
                      metrics_sha256=sha256(path/'run/metrics.json'))
        (path/'experiment_manifest.json').write_text(json.dumps(manifest))
        (path/'result.json').write_text(json.dumps(report))

    def test_comparison_rejects_uncontrolled_source_change(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            self.make_completed_fixture(a, 'microbatch')
            self.make_completed_fixture(b, 'effective_batch')
            self.assertEqual(compare(a, b)['best_validation_delta_pp'], 0)
            manifest = json.loads((b/'experiment_manifest.json').read_text())
            manifest['source_tree_sha256'] = 'other'
            (b/'experiment_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'source'):
                compare(a, b)

    def test_comparison_rejects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            self.make_completed_fixture(a, 'microbatch')
            self.make_completed_fixture(b, 'effective_batch')
            (b/'run/checkpoint_best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                compare(a, b)

    def test_only_scope_differs_in_controlled_recipe(self):
        a = asdict(make_config('p', 'o', 'microbatch'))
        b = asdict(make_config('p', 'o', 'effective_batch'))
        a.pop('advantage_scope')
        b.pop('advantage_scope')
        self.assertEqual(a, b)
        self.assertEqual(a['batch_size']*a['gradient_accumulation_steps'], 64)
        self.assertFalse(a['eval_test'])
        self.assertEqual((a['policy_weight'], a['epochs'], a['seed']), (1., 20, 44))

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            make_config('p', 'o', 'wrong')

    def test_bundle_is_deterministic_and_has_new_algorithm(self):
        data = bundle_source(ROOT)
        self.assertEqual(data, bundle_source(ROOT))
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertIn('src/tacs/effective_batch.py', z.namelist())
            self.assertIn('scripts/advantage_pilot.py', z.namelist())
            self.assertTrue(all(not name.startswith(('/', '..')) for name in z.namelist()))

    def test_notebook_cells_parse_and_control_outputs_are_isolated(self):
        raw = bundle_source(ROOT)
        for scope in ['microbatch', 'effective_batch']:
            nb = make_notebook(raw, scope)
            code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
            for c in nb['cells']:
                if c['cell_type']=='code':
                    ast.parse(''.join(c['source']))
                    self.assertEqual(c['outputs'], [])
            self.assertIn(f'SCOPE = {scope!r}', code)
            self.assertIn('SOURCE_SHA256', code)
            self.assertIn('RESUME_FROM = None', code)
            self.assertNotIn('--eval-test', code)
            self.assertIn('report[\'checkpoint_sha256\']', code)


if __name__ == '__main__':
    unittest.main()
