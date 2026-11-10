"""Offline, validation-only CIFAR-100 inference from a verified real checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from torch.nn import functional as F
from torch.utils.data import Subset

from .data import SplitIndices, cifar_transforms
from .device import resolve_device
from .model import TACSModel, create_task_network, create_vision_encoder
from .selector_audit import encode_bank, validate_provenance
from .train import _autocast_context, _load_standard_encoder, _sha256_file


def default_artifacts():
    root = Path(__file__).resolve().parents[2]
    pilot = root / 'artifacts/policy_weight_pilot'
    reference = pilot / 'kaggle_w0p03_seed44/policy_source_e427116799fe'
    return dict(checkpoint=pilot / 'kaggle_w1p0_seed44/policy_weight_two_stage_1p0_seed44/cifar100/dinov3_vits16/role_v2/seed_44/two_stage/checkpoint_best.pt',
                standard_encoder=reference / 'reference/no_context/best_encoder.pt',
                split_file=reference / 'data/splits/cifar100_seed42.json',
                data_root=root / 'data', cache_dir=root / 'artifacts/demo_cache')


def load_verified(checkpoint, standard_encoder, split_file, device='auto'):
    checkpoint, standard_encoder, split_file = map(Path, (checkpoint, standard_encoder, split_file))
    metrics = json.loads((checkpoint.parent / 'metrics.json').read_text(encoding='utf-8'))
    owner = next((p for p in checkpoint.parents if (p / 'result.json').is_file()
                  and (p / 'experiment_manifest.json').is_file()), None)
    if owner is None:
        raise ValueError('Missing result.json/experiment_manifest.json; use a verified experiment root')
    report = json.loads((owner / 'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((owner / 'experiment_manifest.json').read_text(encoding='utf-8'))
    digest = _sha256_file(checkpoint)
    if report.get('status') != 'complete' or digest != report.get('checkpoint_sha256'):
        raise ValueError('Checkpoint hash/status mismatch')
    if _sha256_file(checkpoint.parent / 'metrics.json') != report.get('metrics_sha256'):
        raise ValueError('Metrics hash mismatch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    split_hash = _sha256_file(split_file)
    cfg = validate_provenance(payload, metrics, split_hash)
    if cfg.baseline != 'two_stage' or cfg.backbone != 'dinov3_vits16':
        raise ValueError('This demo currently supports verified Two-Stage CIFAR-100 only')
    split = SplitIndices.load(split_file)
    split.validate(50_000, 'cifar100')
    cfg.standard_encoder_path = str(standard_encoder)
    standard, _, encoder_hash = _load_standard_encoder(cfg, torch.device('cpu'), split_hash)
    if encoder_hash != manifest.get('encoder_sha256') or split_hash != manifest.get('split_sha256'):
        raise ValueError('Reference encoder/split differs from training manifest')
    model = TACSModel(cfg.num_classes, cfg.embedding_dim, cfg.selector_temperature, True,
        encoder=create_vision_encoder(cfg.backbone, cfg.embedding_dim, pretrained=False),
        task_network=create_task_network(cfg.backbone, cfg.num_classes, cfg.embedding_dim,
            pretrained=False, task_network_version=cfg.task_network_version))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    metadata = dict(checkpoint_sha256=digest, checkpoint_epoch=payload['epoch'],
                    checkpoint_val_accuracy=payload['val_accuracy'], split_sha256=split_hash,
                    standard_encoder_sha256=encoder_hash, task_network_version=cfg.task_network_version,
                    policy_weight=cfg.policy_weight, training_seed=cfg.seed,
                    torch_version=str(torch.__version__), weights_updated=False, test_evaluated=False)
    return SimpleNamespace(model=model, standard=standard, config=cfg, split=split,
                           device=resolve_device(device), metadata=metadata)


class DemoEngine:
    def __init__(self, checkpoint, standard_encoder, split_file, data_root, cache_dir,
                 device='auto', progress=print):
        started = time.perf_counter()
        torch.set_num_threads(2)
        loaded = load_verified(checkpoint, standard_encoder, split_file, device)
        self.model, self.standard, self.cfg, self.split = loaded.model, loaded.standard, loaded.config, loaded.split
        self.device, self.metadata = loaded.device, loaded.metadata
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')
        self.amp = self.device.type == 'cuda' and self.cfg.amp
        self.transform = cifar_transforms('cifar100', self.cfg.image_size, 'dinov3')[1]
        from torchvision.datasets import CIFAR100
        self.dataset = CIFAR100(str(data_root), train=True, transform=self.transform, download=False)
        self.pool = Subset(self.dataset, self.split.candidate_pool)
        self.model.to(self.device).eval()
        self.standard.to(self.device).eval()
        identity = {**self.metadata, 'schema': 1, 'device': str(self.device),
                    'matmul_precision': torch.get_float32_matmul_precision(), 'amp': self.amp, 'bank_batch_size': 4,
                    'pool_indices': self.split.candidate_pool, 'transform': 'cifar100/dinov3/eval256'}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(cache_dir) / (key + '.pt')
        if cache.is_file():
            progress('Loading verified embedding cache')
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            if saved.get('identity') != identity:
                raise ValueError('Embedding cache identity mismatch')
            frozen, selector = saved['frozen'], saved['selector']
        else:
            progress('Encoding 9,000 frozen retrieval candidates (first launch only)')
            frozen = encode_bank(self.standard, self.pool, self.device, 4, False, 'demo retrieval', normalize=True)
            progress('Encoding 9,000 selector candidates (first launch only)')
            selector = encode_bank(self.model.selector.encoder, self.pool, self.device, 4, self.amp, 'demo selector')
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix('.tmp')
            torch.save(dict(identity=identity, frozen=frozen, selector=selector), temp)
            temp.replace(cache)
        expected = (len(self.pool), self.cfg.embedding_dim)
        for bank in (frozen, selector):
            if tuple(bank.shape) != expected or not bool(torch.isfinite(bank).all()):
                raise ValueError('Invalid/nonfinite candidate bank')
        self.frozen, self.selector = frozen.to(self.device), selector.to(self.device)
        self.metadata.update(device=str(self.device), amp=self.amp, setup_seconds=time.perf_counter()-started,
                             candidate_count=len(self.pool), cache=str(cache))
        progress('Ready: real checkpoint, real CIFAR-100, validation only')

    def validation_image(self, position):
        if not 0 <= position < len(self.split.validation):
            raise ValueError('Validation position must be in 0..4999')
        index = self.split.validation[position]
        return self.raw_image(index), self.dataset.classes[self.dataset.targets[index]]

    def raw_image(self, index):
        from PIL import Image
        return Image.fromarray(self.dataset.data[index])

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict(self, image, random_seed=44):
        query = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        self._sync()
        start = time.perf_counter()
        # Frozen retrieval is FP32, same as training/audit.
        q = F.normalize(self.standard(query).float(), dim=-1)
        if not bool(torch.isfinite(q).all()) or bool((q.norm(dim=-1) == 0).any()):
            raise ValueError('Non-finite/zero query embedding')
        topk = (q @ self.frozen.float().T).topk(min(self.cfg.topk_candidates, len(self.pool)), dim=-1).indices
        self._sync()
        retrieved_at = time.perf_counter()
        with _autocast_context(self.device, self.amp):
            choice = self.model.selector.forward_embedding_bank(query, self.selector[topk])
        if not bool(torch.isfinite(choice[0]).all()):
            raise ValueError('Non-finite selector scores')
        local = int(choice[1].argmax(-1))
        selected = int(topk[0, local])
        self._sync()
        selected_at = time.perf_counter()
        generator = torch.Generator().manual_seed(random_seed)
        random_position = int(topk[0, int(torch.randint(topk.shape[1], (1,), generator=generator))])
        interventions = dict(selected=selected, random=random_position,
                             frozen_cosine=int(topk[0, 0]), no_context=None)
        results = {}
        for name, position in interventions.items():
            context = None if position is None else self.pool[position][0].unsqueeze(0).to(self.device)
            self._sync()
            t = time.perf_counter()
            with _autocast_context(self.device, self.amp):
                logits = self.model.task_network(query, context)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Non-finite demo logits')
            probabilities, labels = logits.float().softmax(-1).topk(5)
            self._sync()
            item = dict(prediction=self.dataset.classes[int(labels[0, 0])],
                        top5=[dict(label=self.dataset.classes[int(i)], probability=float(p))
                              for i, p in zip(labels[0], probabilities[0])],
                        task_ms=1000*(time.perf_counter()-t), context_index=None,
                        context_label=None)
            if position is not None:
                index = self.split.candidate_pool[position]
                item.update(context_index=index, context_label=self.dataset.classes[self.dataset.targets[index]])
            results[name] = item
        return dict(metadata=self.metadata, results=results,
                    retrieval_ms=1000*(retrieved_at-start), selector_ms=1000*(selected_at-retrieved_at),
                    comparison_ms=1000*(time.perf_counter()-start),
                    warning='Same task weights, not independently trained baselines. Confidence is not calibrated. Batch=1 demo timing is not a formal benchmark.')
