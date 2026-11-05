import ast
import base64
import hashlib
import importlib.util
import io
import json
import unittest
import zipfile
from pathlib import Path


class KagglePilotNotebookTest(unittest.TestCase):
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
        module.main()
        notebook = json.loads((root / 'artifacts/role_v2/TACS_Kaggle_Role_Pilot.ipynb').read_text(encoding='utf-8'))
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
