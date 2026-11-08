import ast
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PolicyWeightPilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script('policy_weight_pilot')
        cls.builder = load_script('build_policy_weight_notebooks')

    def test_recipe_changes_only_weight_within_baseline(self):
        from dataclasses import asdict
        for baseline in ('tacs', 'two_stage'):
            configs = [asdict(self.runner.make_config(Path('project'), Path('output'), baseline, w))
                       for w in self.runner.WEIGHTS]
            for config in configs:
                self.assertFalse(config['eval_test'])
                self.assertEqual(config['epochs'], 20)
                self.assertEqual(config['batch_size'] * config['gradient_accumulation_steps'], 64)
                self.assertEqual(config['seeds'], (44,))
                config.pop('policy_weight')
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(configs[1], configs[2])

    def test_invalid_experiment_rejected(self):
        for baseline, weight in [('random', 1.0), ('tacs', -1), ('two_stage', float('nan'))]:
            with self.assertRaises(ValueError):
                self.runner.make_config(Path('p'), Path('o'), baseline, weight)

    def test_matching_manifest_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            self.runner.guard_manifest(target, {'weight': 0.1})
            self.runner.guard_manifest(target, {'weight': 0.1})
            self.assertEqual(json.loads((target / 'experiment_manifest.json').read_text()), {'weight': 0.1})

    def test_mismatch_never_overwrites_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            self.runner.guard_manifest(target, {'weight': 1.0})
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                self.runner.guard_manifest(target, {'weight': 0.1})
            self.assertEqual(json.loads((target / 'experiment_manifest.json').read_text()), {'weight': 1.0})

    def test_unowned_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            sentinel = target / 'checkpoint.pt'
            sentinel.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'Nonempty'):
                self.runner.guard_manifest(target, {})
            self.assertEqual(sentinel.read_bytes(), b'preserve')

    def test_corrupt_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'experiment_manifest.json').write_text('{')
            with self.assertRaises(ValueError):
                self.runner.guard_manifest(Path(tmp), {})

    def test_source_bundle_is_current_and_has_no_weights(self):
        raw = self.builder.bundle_source(ROOT)
        self.assertEqual(raw, self.builder.bundle_source(ROOT))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read('src/tacs/train.py'), (ROOT / 'src/tacs/train.py').read_bytes())
            self.assertEqual(archive.read('scripts/policy_weight_pilot.py'), (ROOT / 'scripts/policy_weight_pilot.py').read_bytes())
            self.assertFalse(any(n.endswith('.pt') for n in archive.namelist()))
            for name in archive.namelist():
                if name.endswith('.py'):
                    ast.parse(archive.read(name), filename=name)

    def test_wrong_encoder_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'best_encoder.pt'
            encoder.write_bytes(b'not the original encoder')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.runner.run(Path(tmp), Path(tmp) / 'output', 'two_stage', 0.1, 'source', encoder)
            self.assertFalse((Path(tmp) / 'output').exists())

    def test_generated_commands_do_not_evaluate_test(self):
        from tacs.multiseed import build_train_command
        for baseline in ('tacs', 'two_stage'):
            config = self.runner.make_config(Path('p'), Path('o'), baseline, 0.03)
            command = build_train_command(config, 44, baseline)
            self.assertNotIn('--eval-test', command)
            self.assertEqual(command[command.index('--policy-weight') + 1], '0.03')
            self.assertEqual(command[command.index('--epochs') + 1], '20')

    def test_all_notebooks_are_valid_and_isolated(self):
        raw = self.builder.bundle_source(ROOT)
        outputs = set()
        for baseline in ('tacs', 'two_stage'):
            for weight in self.runner.WEIGHTS:
                nb = self.builder.make_notebook(raw, baseline, weight)
                cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
                self.assertEqual(len(cells), 5)
                for cell in cells:
                    ast.parse(cell)
                self.assertIn(f'POLICY_WEIGHT = {weight!r}', cells[1])
                self.assertIn(f'BASELINE = {baseline!r}', cells[1])
                output_line = next(line for line in cells[1].splitlines() if line.startswith('OUTPUT_ROOT ='))
                outputs.add(output_line)
                self.assertIn('ENCODER_SHA256', cells[1])
                self.assertIn('RESUME_FROM = None', cells[1])
                self.assertIn('checkpoint_sha256', cells[-1])
        self.assertEqual(len(outputs), 6)


if __name__ == '__main__':
    unittest.main()
