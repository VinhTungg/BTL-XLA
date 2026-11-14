"""Unit fixtures, not substitute Caltech data or reported model accuracy."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from tacs.caltech.data import (CaltechImages, make_split, validate_split, scan, digest)
from tacs.caltech.run import Config, fit, require_manifest, restore
from tacs.baselines import BaselineModel
from tacs.model import TinyVisionEncoder, TinyPatchTokenTaskNetwork


class CaltechTest(unittest.TestCase):
    def inventory(self, classes=3, count=34):
        return dict(classes=[f'class{i}' for i in range(classes)], inventory_sha256='unit-only',
            items=[dict(file=f'c{c}/{i}.jpg',target=c,sha256=f'{c}-{i}',width=300,height=200)
                   for c in range(classes) for i in range(count)])

    def test_split_is_reproducible_balanced_and_disjoint(self):
        inv = self.inventory()
        a, b = make_split(inv), make_split(inv)
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ('query_train','candidate_pool','validation','test')], [57,18,15,12])
        self.assertNotEqual(a['query_train'], make_split(inv, 43)['query_train'])

    def test_overlap_and_wrong_inventory_refused(self):
        inv = self.inventory()
        for mode in ('overlap','inventory','classmap'):
            s = make_split(inv)
            if mode=='overlap': s['candidate_pool'][0] = s['validation'][0]
            if mode=='inventory': s['inventory_sha256'] = 'wrong'
            if mode=='classmap': s['classes'] = ['wrong']
            with self.assertRaises(ValueError): validate_split(s, inv)

    def test_duplicate_content_crossing_roles_refused(self):
        inv = self.inventory()
        s = make_split(inv)
        inv['items'][s['test'][0]]['sha256'] = inv['items'][s['candidate_pool'][0]]['sha256']
        with self.assertRaisesRegex(ValueError, 'Byte-identical'):
            validate_split(s, inv)

    def test_grayscale_rgb_and_background_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'caltech101/101_ObjectCategories'
            (root/'object').mkdir(parents=True)
            (root/'BACKGROUND_Google').mkdir()
            for i in range(31):
                Image.new('L', (40,30), i).save(root/'object'/f'image_{i:04d}.jpg')
            inv = scan(temp, expected_classes=1)
            self.assertEqual(inv['classes'], ['object'])
            self.assertEqual(len(inv['items']), 31)
            image, target = CaltechImages(temp, inv)[0]
            self.assertEqual((image.mode, image.size, target), ('RGB',(40,30),0))
            with self.assertRaises(ValueError): scan(temp)

    def test_manifest_refuses_changed_recipe_and_unowned_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'
            require_manifest(path, {'schema':1})
            require_manifest(path, {'schema':1})
            with self.assertRaises(ValueError): require_manifest(path, {'schema':2})
            other = Path(temp)/'unowned'
            other.mkdir()
            (other/'user.txt').write_text('keep')
            with self.assertRaises(ValueError): require_manifest(other, {'schema':1})
            self.assertEqual((other/'user.txt').read_text(), 'keep')

    def test_config_rejects_wrong_dataset_head_and_schedule(self):
        for kw in [dict(dataset='cifar100'),dict(num_classes=100),dict(epochs=5),dict(batch_size=0),
                   dict(selector_temperature=0),dict(policy_weight=float('nan')),dict(topk_candidates=0)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Config(baseline='two_stage', **kw)

    def test_full_training_defaults_to_100_epochs(self):
        for name in ('no_context','two_stage'):
            self.assertEqual(Config(baseline=name).epochs,100)

    def test_real_epoch_engine_resume_roundtrip_on_unit_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            torch.set_num_threads(2)
            cfg = Config(baseline='no_context', epochs=2, warmup_epochs=0,
                         gradient_accumulation_steps=1)
            inventory = dict(classes=[str(i) for i in range(101)], inventory_sha256='unit')
            loader = DataLoader(TensorDataset(torch.rand(8,3,32,32),torch.arange(8)),batch_size=4)
            def model():
                return BaselineModel(101,8,'no_context',TinyVisionEncoder(8),TinyPatchTokenTaskNetwork(101,8))
            output = Path(temp)
            result = fit(cfg,model(),loader,loader,output,inventory,torch.device('cpu'),stop_after=1)
            self.assertEqual(result['status'],'partial')
            self.assertFalse((output/'result.json').exists())
            first = torch.load(output/'checkpoint_latest.pt',weights_only=True)
            self.assertEqual(first['epoch'],1)
            with self.assertRaisesRegex(ValueError,'config/history mismatch'):
                # Reject changed total schedule before touching model/optimizer.
                restore(output/'checkpoint_latest.pt',None,None,None,None,
                        Config(baseline='no_context',epochs=100))
            result = fit(cfg,model(),loader,loader,output,inventory,torch.device('cpu'))
            self.assertEqual(result['status'],'complete')
            latest = torch.load(output/'checkpoint_latest.pt',weights_only=True)
            self.assertEqual([h['epoch'] for h in latest['history']],[1,2])
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(result['reference_encoder_sha256'],digest(output/'best_encoder.pt'))
            self.assertEqual(latest['config'], asdict(cfg))


if __name__ == '__main__':
    unittest.main()
