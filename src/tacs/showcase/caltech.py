"""Native-resolution Caltech previews with independently verified model weights."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from torch.utils.data import Subset

from .caltech_artifacts import CHECKPOINTS, read_json, verify_artifacts
from ..caltech.data import CaltechImages, scan, transforms
from ..caltech.run import Config, build_model
from ..device import resolve_device
from ..inference import DemoEngine
from ..model import create_task_network, create_vision_encoder
from ..selector_audit import encode_bank
from ..train import _autocast_context


class CaltechDemoEngine(DemoEngine):
    dataset_name='caltech101'

    def __init__(self, root, data_root, cache_dir, device='auto', progress=print):
        began=time.perf_counter()
        root=Path(root)
        torch.set_num_threads(2)
        progress('Xác minh checkpoint, class map, split và dữ liệu Caltech thật…')
        verified=verify_artifacts(root)
        inventory=scan(data_root)
        if inventory['inventory_sha256']!=verified['inventory_sha256']:
            raise ValueError('Ảnh Caltech local khác inventory đã train; không thay bằng CIFAR')
        self.split=SimpleNamespace(**read_json(root/'split.json'))
        self.cfg=Config(baseline='two_stage')
        self.device=resolve_device(device)
        if self.device.type=='cuda': torch.set_float32_matmul_precision('high')
        self.amp=self.device.type=='cuda' and self.cfg.amp
        self.transform=transforms(False)
        self.dataset=CaltechImages(data_root,inventory,self.transform)
        self.pool=Subset(self.dataset,self.split.candidate_pool)
        self.model=build_model(self.cfg,torch.device('cpu'),pretrained=False)
        payload=torch.load(root/'two_stage/checkpoint_best.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(payload['model_state_dict'],strict=True)
        self.model.to(self.device).eval()
        self.metadata=dict(dataset='caltech101',num_classes=101,preset='full100',
            protocol=verified['protocol'],checkpoint_sha256=CHECKPOINTS['two_stage'],
            checkpoint_epoch=payload['epoch'],checkpoint_val_accuracy=payload['val_accuracy'],
            no_context_checkpoint_sha256=CHECKPOINTS['no_context'],
            no_context_checkpoint_epoch=verified['runs']['no_context']['best_epoch'],
            no_context_val_accuracy=verified['runs']['no_context']['best_val_accuracy'],
            split_sha256=verified['split_sha256'],inventory_sha256=verified['inventory_sha256'],
            standard_encoder_sha256=verified['reference_sha256'],training_source_sha256=verified['source_sha256'],
            task_network_version='role_v2',policy_weight=self.cfg.policy_weight,training_seed=44,
            torch_version=str(torch.__version__),weights_updated=False,test_evaluated=False,
            device=str(self.device),amp=self.amp,candidate_count=len(self.pool))
        del payload
        self.standard=create_vision_encoder(self.cfg.backbone,self.cfg.embedding_dim,pretrained=False)
        payload=torch.load(root/'no_context/best_encoder.pt',map_location='cpu',weights_only=True)
        self.standard.load_state_dict(payload['encoder_state_dict'],strict=True)
        self.standard.to(self.device).eval()
        del payload
        identity=dict(schema=1,dataset='caltech101',split=verified['split_sha256'],
            inventory=verified['inventory_sha256'],checkpoint=CHECKPOINTS['two_stage'],
            reference=verified['reference_sha256'],torch=str(torch.__version__),
            device=str(self.device),amp=self.amp,precision=torch.get_float32_matmul_precision(),
            transform='RGB/Resize256x256/ImageNetNorm',batch_size=4)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cache=Path(cache_dir)/(key+'.pt')
        if cache.exists():
            saved=torch.load(cache,map_location='cpu',weights_only=True)
            if saved.get('identity')!=identity: raise ValueError('Caltech cache identity mismatch')
            frozen,selector=saved['frozen'],saved['selector']
        else:
            progress('Dựng kho đặc trưng 606 ảnh pool Caltech (lần đầu)…')
            frozen=encode_bank(self.standard,self.pool,self.device,4,False,'Caltech retrieval',normalize=True)
            selector=encode_bank(self.model.selector.encoder,self.pool,self.device,4,self.amp,'Caltech selector')
            cache.parent.mkdir(parents=True,exist_ok=True)
            temp=cache.with_suffix('.tmp')
            torch.save(dict(identity=identity,frozen=frozen,selector=selector),temp)
            temp.replace(cache)
        for bank in (frozen,selector):
            if tuple(bank.shape)!=(606,384) or not torch.isfinite(bank).all() or (bank.float().norm(dim=-1)==0).any():
                raise ValueError('Invalid Caltech embedding cache')
        self.frozen,self.selector=frozen.to(self.device),selector.to(self.device)
        self.control=create_task_network(self.cfg.backbone,101,384,pretrained=False,task_network_version='role_v2')
        payload=torch.load(root/'no_context/checkpoint_best.pt',map_location='cpu',weights_only=True)
        self.control.load_state_dict({k.removeprefix('task_network.'):v for k,v in
            payload['model_state_dict'].items() if k.startswith('task_network.')},strict=True)
        self.control.to(self.device).eval()
        self.metadata.update(cache=str(cache),setup_seconds=time.perf_counter()-began)
        progress('Sẵn sàng: Caltech-101 thật, hai checkpoint riêng, không train/test.')

    def raw_image(self,index):
        return self.dataset.raw_image(index)

    def validation_image(self,position):
        if not 0<=position<len(self.split.validation):
            raise ValueError('Validation position Caltech phải trong 0..504')
        index=self.split.validation[position]
        return self.raw_image(index),self.dataset.classes[self.dataset.targets[index]]

    @torch.no_grad()
    def predict(self,image,random_seed=44):
        report=super().predict(image,random_seed)
        for item in report['results'].values():
            item['checkpoint_sha256']=CHECKPOINTS['two_stage']
            item['model_group']='two_stage_intervention'
        query=self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        self._sync()
        start=time.perf_counter()
        with _autocast_context(self.device,self.amp): logits=self.control(query)
        if logits.shape!=(1,101) or not torch.isfinite(logits).all():
            raise ValueError('Invalid No-context control logits')
        probabilities,labels=logits.float().softmax(-1).topk(5)
        self._sync()
        elapsed=1000*(time.perf_counter()-start)
        report['results']['trained_no_context']=dict(
            prediction=self.dataset.classes[int(labels[0,0])],
            top5=[dict(label=self.dataset.classes[int(i)],probability=float(p))
                  for i,p in zip(labels[0],probabilities[0])],
            task_ms=elapsed,context_index=None,context_label=None,
            stage1_rank=None,stage2_rank=None,cosine_score=None,selector_score=None,
            checkpoint_sha256=CHECKPOINTS['no_context'],model_group='independent_no_context')
        report['comparison_ms']+=elapsed
        report['warning']=('First four rows intervene on the same Two-Stage task weights; '
            'trained_no_context is a separate 100-epoch model. Validation-only examples; '
            'not a test benchmark. Probabilities are not calibrated.')
        return report
