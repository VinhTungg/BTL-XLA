import ast
import base64
import hashlib
import importlib.util
import io
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


class KagglePilotNotebookTest(unittest.TestCase):
    def test_selector_pilot_has_isolated_outputs_and_ordered_dependency(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'artifacts/role_v2/TACS_Colab_Bundle.zip').is_file():
            self.skipTest('Build the local source bundle first')
        spec = importlib.util.spec_from_file_location(
            'selector_pilot_builder', root / 'scripts/build_kaggle_role_pilot.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = root / 'artifacts/role_v2/TACS_Kaggle_Selector_Pilot.ipynb'
        with patch.object(Path, 'write_text', autospec=True, return_value=0) as write:
            module.main(selector_pilot=True)
        write.assert_called_once()
        self.assertEqual(write.call_args.args[0], target)
        notebook = json.loads(write.call_args.args[1])
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        for cell in cells:
            ast.parse(cell)
        train = cells[-1]
        self.assertIn("task_network_version='role_v2'", train)
        self.assertIn("baselines=('no_context', 'tacs', 'two_stage')", train)
        self.assertIn("run_baseline('no_context', gpu), run_baseline('two_stage', gpu)", train)
        self.assertIn("is_complete_run(CONFIG, 44, 'no_context')", train)
        self.assertIn('epochs=20, batch_size=4, gradient_accumulation_steps=16', train)
        self.assertIn('lr=1e-4, warmup_epochs=5, num_workers=2, eval_test=False', train)
        self.assertIn("/kaggle/working/TACS_selector_role_v2_pilot", train)
        self.assertIn("torch.cuda.device_count() >= 2", train)
        self.assertIn("for name in CONFIG.baselines", train)
        self.assertNotIn('checkpoint_best.pt', train)  # No warm-start from Random task weights.

    def test_generator_embeds_verified_source_and_controlled_recipe(self):
        root = Path(__file__).resolve().parents[1]
        bundle_path = root / 'artifacts/role_v2/TACS_Colab_Bundle.zip'
        if not bundle_path.is_file():
            self.skipTest('Build the local source bundle first')
        spec = importlib.util.spec_from_file_location(
            'kaggle_pilot_builder', root / 'scripts/build_kaggle_role_pilot.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = root / 'artifacts/role_v2/TACS_Kaggle_Role_Pilot.ipynb'
        with patch.object(Path, 'write_text', autospec=True, return_value=0) as write:
            module.main()
        write.assert_called_once()
        self.assertEqual(write.call_args.args[0], target)
        notebook = json.loads(write.call_args.args[1])
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        for code in cells:
            ast.parse(code)
        setup_ast = ast.parse(cells[0])
        encoded = next(node.value.value for node in setup_ast.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'encoded_bundle' for t in node.targets))
        raw = base64.b64decode(encoded)
        self.assertEqual(raw, bundle_path.read_bytes())
        self.assertIn(hashlib.sha256(raw).hexdigest(), cells[0])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn(b'context_role_embedding', archive.read('src/tacs/model.py'))
            self.assertFalse(any(name.endswith('.pt') for name in archive.namelist()))
        self.assertIn("VERSIONS = ('legacy_v1', 'role_v2')", cells[2])
        self.assertIn('seeds=(44,)', cells[2])
        self.assertIn('epochs=20, batch_size=4, gradient_accumulation_steps=16', cells[2])
        self.assertIn('eval_test=False', cells[2])
        self.assertIn("metrics['history'][-1]['val_acc']", cells[2])


if __name__ == '__main__':
    unittest.main()
