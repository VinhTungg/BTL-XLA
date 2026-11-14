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


if __name__ == '__main__':
    unittest.main()
