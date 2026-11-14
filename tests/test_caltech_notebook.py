"""Validate notebook contents without starting any cloud job."""
import ast
import base64
import hashlib
import importlib.util
import io
from pathlib import Path
import unittest
import zipfile


class CaltechNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/'scripts/build_caltech_colab.py'
        spec = importlib.util.spec_from_file_location('caltech_notebook',path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.build = staticmethod(mod.build)

    def test_separate_setup_train_verify_cells(self):
        nb = self.build()
        cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),7)
        for cell in cells:
            ast.parse(cell)
        for cell in cells[:5]:
            self.assertNotIn('subprocess.Popen',cell)
            self.assertNotIn("'-m','tacs.caltech.run'",cell)
        self.assertIn("EPOCHS=100",cells[4])
        self.assertIn('seed44_full100_v1',cells[1])
        self.assertNotIn('seed44_demo_v1',cells[1])
        self.assertIn("BASELINE='both'",cells[4])
        self.assertIn('STOP_AFTER=None',cells[4])
        self.assertIn('train.log',cells[5])
        self.assertIn("checkpoint_best.pt",cells[6])
        self.assertIn("not report['test_evaluated']",cells[6])
        self.assertTrue(all(c['execution_count'] is None and not c['outputs']
                            for c in nb['cells'] if c['cell_type']=='code'))

    def test_embedded_source_deterministic_hashed_complete(self):
        a,b=self.build(),self.build()
        self.assertEqual(a,b)
        tree=ast.parse(''.join(a['cells'][3]['source']))
        assignments={n.targets[0].id:n.value for n in tree.body
                     if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        sha=ast.literal_eval(assignments['SOURCE_SHA256'])
        raw=base64.b64decode(ast.literal_eval(assignments['raw'].args[0]))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),sha)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names=archive.namelist()
            self.assertIn('src/tacs/caltech/run.py',names)
            self.assertIn('src/tacs/model.py',names)
            self.assertIn('pyproject.toml',names)
            self.assertFalse(any(n.startswith('/') or '..' in Path(n).parts for n in names))
            self.assertFalse(any(n.endswith(('.pt','.jpg')) for n in names))


if __name__=='__main__':
    unittest.main()
