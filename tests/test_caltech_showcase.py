"""Dataset switching, provenance and model separation without real GPU work."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from tacs.inference import DemoEngine
from tacs.showcase.caltech import CaltechDemoEngine
from tacs.showcase.caltech_artifacts import CHECKPOINTS
from tacs.showcase.__main__ import external_input,validation_input,make_report,create_engine,main


class CaltechShowcaseTest(unittest.TestCase):
    def test_caltech_query_carries_native_size_and_file_not_cifar_index(self):
        engine=SimpleNamespace(dataset_name='caltech101',split=SimpleNamespace(validation=[8]),
            dataset=SimpleNamespace(inventory={'items':[{}]*8+[{'file':'object/image_1.jpg'}]}),
            validation_image=lambda i:(Image.new('RGB',(350,240)),'object'))
        image,query=validation_input(engine,0)
        self.assertEqual(query['original_size'],[350,240])
        self.assertEqual(query['dataset_index'],8)
        self.assertNotIn('cifar_index',query)
        self.assertEqual(image.size,(350,240))

    def test_external_caltech_uses_101_class_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'gray.png'
            Image.new('L',(400,300)).save(path)
            image,query=external_input(path,'caltech101')
            self.assertEqual(image.mode,'RGB')
            self.assertIn('101 lớp Caltech-101',query['warning'])
            self.assertIsNone(query['target'])

    def test_dataset_mismatch_refused_before_prediction(self):
        engine=SimpleNamespace(dataset_name='caltech101',predict=lambda x:self.fail('must not predict'))
        with self.assertRaisesRegex(ValueError,'khác dataset'):
            make_report(engine,Image.new('RGB',(2,2)),{'dataset':'cifar100'})

    def test_wrong_dataset_or_preset_refused_without_model_loading(self):
        for dataset,preset in [('caltech101','microbatch'),('cifar100','full100'),('other','full100')]:
            with self.subTest(dataset=dataset),self.assertRaises((ValueError,KeyError)):
                create_engine(preset,dataset=dataset)

    def test_caltech_cli_rejects_out_of_range_before_loading(self):
        import sys
        with patch.object(sys,'argv',['showcase','--dataset','caltech101','--headless','--validation-position','505']), \
                patch('tacs.showcase.__main__.create_engine') as create:
            with self.assertRaises(SystemExit) as exc: main()
            self.assertEqual(exc.exception.code,1)
            create.assert_not_called()

    def test_separate_no_context_control_has_own_hash_and_no_context_image(self):
        engine=CaltechDemoEngine.__new__(CaltechDemoEngine)
        engine.device=torch.device('cpu')
        engine.amp=False
        engine.transform=lambda image:torch.zeros(3,16,16)
        engine.dataset=SimpleNamespace(classes=[str(i) for i in range(101)])
        engine.control=lambda q:torch.arange(101,dtype=torch.float32).unsqueeze(0)
        report=dict(metadata={},retrieval_ms=1.,selector_ms=2.,comparison_ms=3.,results={
            name:dict(prediction='0',context_index=None) for name in ('selected','random','frozen_cosine','no_context')})
        with patch.object(DemoEngine,'predict',return_value=report):
            result=engine.predict(Image.new('RGB',(300,200)))
        a=result['results']['no_context']
        b=result['results']['trained_no_context']
        self.assertEqual(a['checkpoint_sha256'],CHECKPOINTS['two_stage'])
        self.assertEqual(b['checkpoint_sha256'],CHECKPOINTS['no_context'])
        self.assertEqual(b['prediction'],'100')
        self.assertIsNone(b['context_index'])
        self.assertEqual(len(b['top5']),5)


if __name__=='__main__': unittest.main()
