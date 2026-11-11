import ast
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_advantage_colab import EXPECTED_BUNDLE, make_colab
from build_advantage_notebooks import bundle_source


class AdvantageColabTest(unittest.TestCase):
    def test_bundle_is_same_as_completed_control(self):
        raw = bundle_source(ROOT)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_BUNDLE)
        nb = make_colab(raw)
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


if __name__ == '__main__':
    unittest.main()
