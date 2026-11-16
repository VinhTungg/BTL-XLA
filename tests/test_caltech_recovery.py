"""Archive safety and artifact identity checks use explicit unit-only fixtures."""
import importlib.util
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from tacs.caltech.data import make_split, PROTOCOL
from tacs.caltech.run import Config
from tacs.showcase.caltech_artifacts import CHECKPOINTS, SOURCE_SHA256, verify_artifacts


class CaltechRecoveryTest(unittest.TestCase):
    def fixture(self):
        inventory=dict(classes=[f'class{i}' for i in range(101)],inventory_sha256='inventory',
            items=[dict(target=c,sha256=f'{c}-{i}') for c in range(101) for i in range(86 if c<92 else 85)])
        split=make_split(inventory)
        files={'inventory.json':inventory,'split.json':split}
        payloads={}
        for name in CHECKPOINTS:
            cfg=asdict(Config(baseline=name))
            history=[dict(epoch=e,val_acc=.9,seconds=1) for e in range(1,101)]
            files[f'{name}/experiment_manifest.json']=dict(kind='caltech101_demo',config=cfg,
                source_sha256=SOURCE_SHA256,inventory_sha256='inventory',split_sha256='split',
                reference_sha256='ref' if name=='two_stage' else None,environment={'gpu':'unit'},
                test_evaluated=False,protocol=PROTOCOL)
            files[f'{name}/result.json']=dict(status='complete',config=cfg,test_evaluated=False,
                protocol=PROTOCOL,checkpoint_sha256=CHECKPOINTS[name],metrics_sha256='metrics',
                reference_encoder_sha256='ref' if name=='no_context' else None,best_val_accuracy=.9)
            files[f'{name}/metrics.json']=dict(config=cfg,classes=inventory['classes'],protocol=PROTOCOL,
                history=history,best_val_acc=.9,test_result=None)
            payloads[f'{name}/checkpoint_best.pt']=dict(config=cfg,classes=inventory['classes'],
                protocol=PROTOCOL,epoch=1,val_accuracy=.9,model_state_dict={'fixture':torch.tensor([1])})
            payloads[f'{name}/checkpoint_latest.pt']=dict(config=cfg,epoch=100,history=history,best_val_accuracy=.9,
                **{k:{} for k in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict',
                                 'torch_rng_state','python_rng_state','numpy_rng_state')})
        payloads['no_context/best_encoder.pt']=dict(encoder_state_dict={'fixture':torch.tensor([1])},
            metadata=dict(dataset='caltech101',protocol=PROTOCOL,inventory_sha256='inventory'))
        return files,payloads

    def verify(self,files,payloads):
        def relative(p): return Path(p).relative_to(Path('unit')).as_posix()
        def fake_digest(p):
            rel=relative(p)
            if rel.endswith('checkpoint_best.pt'): return CHECKPOINTS[rel.split('/')[0]]
            if rel.endswith('best_encoder.pt'): return 'ref'
            if rel=='split.json': return 'split'
            return 'metrics'
        with patch('tacs.showcase.caltech_artifacts.read_json',side_effect=lambda p:files[relative(p)]), \
             patch('tacs.showcase.caltech_artifacts.digest',side_effect=fake_digest), \
             patch('torch.load',side_effect=lambda p,**kw:payloads[relative(p)]):
            return verify_artifacts('unit',verify_latest=True)

    def test_complete_recipe_and_full_checkpoint(self):
        files,payloads=self.fixture()
        result=self.verify(files,payloads)
        self.assertEqual(result['status'],'verified')
        self.assertEqual(result['best_val_delta_pp'],0)
        self.assertEqual(result['sizes']['validation'],505)

    def test_wrong_dataset_source_hash_partial_or_test_refused(self):
        for kind in ('dataset','source','hash','partial','test','latest','classes','history'):
            with self.subTest(kind=kind):
                files,payloads=self.fixture()
                if kind=='dataset': files['two_stage/experiment_manifest.json']['config']['dataset']='cifar100'
                if kind=='source': files['two_stage/experiment_manifest.json']['source_sha256']='wrong'
                if kind=='hash': files['two_stage/result.json']['checkpoint_sha256']='wrong'
                if kind=='partial': files['two_stage/result.json']['status']='partial'
                if kind=='test': files['two_stage/metrics.json']['test_result']={'accuracy':.99}
                if kind=='latest': payloads['two_stage/checkpoint_latest.pt']['epoch']=99
                if kind=='classes': payloads['two_stage/checkpoint_best.pt']['classes']=['wrong']
                if kind=='history': files['two_stage/metrics.json']['history'][50]['epoch']=49
                with self.assertRaises(ValueError): self.verify(files,payloads)

    def recover_module(self):
        path=Path(__file__).resolve().parents[1]/'scripts/recover_caltech_result.py'
        spec=importlib.util.spec_from_file_location('recover_caltech',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_unsafe_or_unexpected_archive_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('../escape.txt','not allowed')
            with self.assertRaises(ValueError): self.recover_module().extract_safely(archive,root)
            self.assertEqual([p.name for p in root.iterdir()],['bad.zip'])

    def test_recovery_idempotent_and_preserves_existing_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/'unit.zip'
            names=['inventory.json','split.json','train.log','no_context/best_encoder.pt']
            names += [f'{n}/{f}' for n in CHECKPOINTS for f in ('experiment_manifest.json','result.json',
                      'metrics.json','checkpoint_best.pt','checkpoint_latest.pt')]
            with zipfile.ZipFile(archive,'w') as z:
                for n in names: z.writestr('seed44_full100_v1/'+n,'unit fixture only')
            module=self.recover_module()
            path=module.extract_safely(archive,root)
            module.extract_safely(archive,root)
            (path/'inventory.json').write_text('user data')
            with self.assertRaises(FileExistsError): module.extract_safely(archive,root)
            self.assertEqual((path/'inventory.json').read_text(),'user data')
            self.assertTrue(archive.exists())


if __name__=='__main__': unittest.main()
