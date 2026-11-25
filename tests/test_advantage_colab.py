import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_advantage_colab import CONTROL_NOTEBOOK, EXPECTED_BUNDLE, make_colab, read_control_source
from build_advantage_notebooks import make_notebook


class AdvantageColabTest(unittest.TestCase):
    @unittest.skipUnless(CONTROL_NOTEBOOK.is_file(), 'Original microbatch notebook is a local artifact')
    def test_archived_source_is_same_as_completed_control(self):
        raw = read_control_source(CONTROL_NOTEBOOK)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_BUNDLE)

    def test_builder_reads_embedded_source_and_generates_colab(self):
        import json
        raw = b'embedded source fixture'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'control.ipynb'
            path.write_text(json.dumps(make_notebook(raw, 'microbatch')), encoding='utf-8')
            self.assertEqual(read_control_source(path), raw)
        nb = self.make_test_colab(raw)
        self.assertEqual(nb['metadata']['accelerator'], 'GPU')
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
        for required in ("SCOPE = 'effective_batch'", "gpu != 'Tesla T4'", 'drive.mount',
                         "torch==2.11.0", "torchvision==0.26.0", 'https://download.pytorch.org/whl/cu130',
                         "source_bundle_sha256", 'symlink_to', 'OUTPUT_ROOT.is_symlink()',
                         'best_encoder.pt', "control['environment'] != expected_env"):
            self.assertIn(required, code)
        self.assertNotIn('checkpoint_best.pt\'\n', code)
        self.assertNotIn('eval_test=True', code)
        self.assertNotIn('rmtree', code)

    def test_changed_source_refused(self):
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            make_colab(b'not the matched control source')

    def make_test_colab(self, raw):
        with mock.patch('build_advantage_colab.EXPECTED_BUNDLE', hashlib.sha256(raw).hexdigest()):
            return make_colab(raw)

    def preflight_code(self):
        code = ''.join(self.make_test_colab(b'embedded source fixture')['cells'][2]['source'])
        return code[:code.index('def install_live')]

    def test_cpu_runtime_refused_before_install(self):
        with mock.patch('shutil.which', return_value=None), mock.patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'No NVIDIA runtime'):
                exec(self.preflight_code(), {})
            run.assert_not_called()

    def test_wrong_gpu_refused_before_install(self):
        with mock.patch('shutil.which', return_value='/usr/bin/nvidia-smi'), mock.patch(
                'subprocess.run', return_value=SimpleNamespace(stdout='NVIDIA A100\n')):
            with self.assertRaisesRegex(RuntimeError, 'Expected one Tesla T4'):
                exec(self.preflight_code(), {})

    def test_t4_preflight_passes(self):
        with mock.patch('shutil.which', return_value='/usr/bin/nvidia-smi'), mock.patch(
                'subprocess.run', return_value=SimpleNamespace(stdout='Tesla T4\n')) as run:
            exec(self.preflight_code(), {})
            self.assertEqual(run.call_args.args[0], ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])


if __name__ == '__main__':
    unittest.main()
