import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from tacs.inference import DemoEngine, load_verified


class DemoTest(unittest.TestCase):
    def test_unverified_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Missing result'):
                load_verified(root/'checkpoint_best.pt', root/'encoder.pt', root/'split.json')

    def test_tampered_checkpoint_refused_before_loading_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'metrics.json').write_text('{}')
            (root/'result.json').write_text(json.dumps(dict(status='complete', checkpoint_sha256='wrong')))
            (root/'experiment_manifest.json').write_text('{}')
            checkpoint = root/'checkpoint_best.pt'
            checkpoint.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_verified(checkpoint, root/'encoder.pt', root/'split.json')

    def test_validation_bounds_and_raw_image(self):
        import numpy as np
        # Unit fixture, not runtime demo data.
        engine = DemoEngine.__new__(DemoEngine)
        engine.split = SimpleNamespace(validation=[2])
        engine.dataset = SimpleNamespace(data=np.zeros((3, 32, 32, 3), dtype=np.uint8), targets=[0, 0, 1], classes=['a', 'b'])
        image, target = engine.validation_image(0)
        self.assertEqual((image.size, target), ((32, 32), 'b'))
        for position in [-1, 1]:
            with self.assertRaises(ValueError):
                engine.validation_image(position)

    def test_cli_help_does_not_load_model_or_data(self):
        import sys
        from tacs.demo import main
        with patch.object(sys, 'argv', ['demo', '--help']):
            with self.assertRaises(SystemExit) as exit:
                main()
            self.assertEqual(exit.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
