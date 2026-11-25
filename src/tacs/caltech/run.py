"""Caltech-101 demo trainer using the existing task/selector/loss/epoch engine.

This is a separate demo protocol, NOT the original paper's complete evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .cosine import CosineTop1Model, PairedClassificationLoss
from .data import PROTOCOL, TRAIN_PER_CLASS, atomic_json, datasets, digest, prepare, prepare_train_subset
from ..baselines import BaselineModel
from ..device import seed_everything
from ..losses import TACSLoss
from ..model import TACSModel, create_task_network, create_vision_encoder
from ..retrieval import CosineSimilarityRetriever, SimilarityContextDataset, TopKContextDataset, precompute_candidate_embeddings
from ..train import train_one_epoch, evaluate, build_optimizer_and_scheduler, _atomic_torch_save


@dataclass
class Config:
    baseline: str
    dataset: str = 'caltech101'
    backbone: str = 'dinov3_vits16'
    task_network_version: str = 'role_v2'
    num_classes: int = 101
    embedding_dim: int = 384
    image_size: int = 256
    seed: int = 44
    epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    advantage_scope: str = 'microbatch'
    lr: float = 1e-4
    min_lr: float = 0.
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    selector_temperature: float = .1
    policy_weight: float = 1.
    topk_candidates: int = 32
    num_workers: int = 0
    amp: bool = True
    aux_query_weight: ClassVar[float] = 0.0

    def __post_init__(self):
        if self.baseline not in ('no_context', 'two_stage', 'cosine_top1'):
            raise ValueError('Unknown Caltech classifier')
        if (self.dataset, self.backbone, self.num_classes, self.image_size, self.embedding_dim,
                self.task_network_version, self.advantage_scope) != (
                'caltech101', 'dinov3_vits16', 101, 256, 384, 'role_v2', 'microbatch'):
            raise ValueError('Wrong Caltech-101 demo architecture/protocol')
        if min(self.epochs, self.batch_size, self.gradient_accumulation_steps) < 1 or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError('Invalid epochs/batch/accumulation/warmup')
        if not 0 <= self.min_lr < self.lr or self.weight_decay < 0 or self.num_workers < 0:
            raise ValueError('Invalid optimizer/workers')
        if (not all(math.isfinite(x) for x in (self.lr, self.min_lr, self.weight_decay,
                self.selector_temperature, self.policy_weight))
                or self.selector_temperature <= 0 or self.policy_weight < 0
                or self.topk_candidates < 1):
            raise ValueError('Invalid selector/optimizer values')
        if self.baseline == 'cosine_top1' and (self.topk_candidates != 1 or self.policy_weight != 0):
            raise ValueError('Cosine top-1 requires one context and no selector policy')


@dataclass
class AnchoredConfig(Config):
    # Keep the original Config payload identical for existing full100 artifacts.
    aux_query_weight: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if (self.baseline not in ('two_stage', 'cosine_top1') or not math.isfinite(self.aux_query_weight)
                or self.aux_query_weight <= 0):
            raise ValueError('Anchored classification requires a context arm and a positive finite query-only loss weight')


def source_digest():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' not in path.parts:
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def require_manifest(output, expected):
    path = Path(output)/'experiment_manifest.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Manifest differs: preserve outputs and use the exact original source/config/environment')
    else:
        if Path(output).exists() and any(Path(output).iterdir()):
            raise ValueError('Nonempty unowned output: refusing to overwrite')
        atomic_json(path, expected)


def build_model(cfg, device, pretrained=True):
    seed_everything(cfg.seed)
    encoder = create_vision_encoder(cfg.backbone, cfg.embedding_dim, pretrained=pretrained)
    task = create_task_network(cfg.backbone, cfg.num_classes, cfg.embedding_dim,
                               pretrained=pretrained, task_network_version=cfg.task_network_version)
    if cfg.baseline == 'no_context':
        model = BaselineModel(cfg.num_classes, cfg.embedding_dim, 'no_context', encoder, task)
    elif cfg.baseline == 'cosine_top1':
        model = CosineTop1Model(task)
    else:
        model = TACSModel(cfg.num_classes, cfg.embedding_dim, cfg.selector_temperature,
                          True, encoder=encoder, task_network=task)
    return model.to(device)


def build_loaders(cfg, data_root, inventory, split, reference, device, *, train_indices=None):
    query, pool, val = datasets(data_root, inventory, split, train_indices=train_indices)
    if cfg.baseline != 'no_context':
        frozen = create_vision_encoder(cfg.backbone, cfg.embedding_dim, pretrained=False).to(device)
        payload = torch.load(reference, map_location='cpu', weights_only=True)
        if (payload['metadata']['dataset'] != cfg.dataset
                or payload['metadata']['inventory_sha256'] != inventory['inventory_sha256']
                or payload['metadata']['protocol'] != PROTOCOL):
            raise ValueError('Wrong Caltech reference encoder')
        frozen.load_state_dict(payload['encoder_state_dict'], strict=True)
        bank = precompute_candidate_embeddings(pool, frozen, cfg.batch_size, device)
        retriever = CosineSimilarityRetriever(bank['embeddings'], bank['indices'], bank['targets'], pool)
        print(f'Preparing fixed top-{cfg.topk_candidates} assignments for train/validation...', flush=True)
        if cfg.baseline == 'cosine_top1':
            # Retrieve from the same resized view used at inference. Keep normal
            # random crops for the classifier's training input.
            retrieval_queries = Subset(pool.dataset, query.indices)
            query = SimilarityContextDataset(query, pool, retriever, frozen, device, cfg.batch_size,
                                             retrieval_query_dataset=retrieval_queries)
            val = SimilarityContextDataset(val, pool, retriever, frozen, device, cfg.batch_size)
        else:
            query = TopKContextDataset(query, pool, retriever, frozen, device, cfg.batch_size, cfg.topk_candidates)
            val = TopKContextDataset(val, pool, retriever, frozen, device, cfg.batch_size, cfg.topk_candidates)
        del frozen, bank, payload
    return (DataLoader(query, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
            DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers))


def restore(path, model, optimizer, scheduler, scaler, cfg, *, train_subset_sha256=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('config') != asdict(cfg) or payload.get('epoch', 0) != len(payload.get('history', [])):
        raise ValueError('Checkpoint config/history mismatch')
    if payload.get('train_subset_sha256') != train_subset_sha256:
        raise ValueError('Checkpoint training subset mismatch')
    if [row['epoch'] for row in payload['history']] != list(range(1, payload['epoch']+1)):
        raise ValueError('Checkpoint history has gaps')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scheduler.load_state_dict(payload['scheduler_state_dict'])
    scaler.load_state_dict(payload['scaler_state_dict'])
    torch.set_rng_state(payload['torch_rng_state'].cpu())
    if payload['cuda_rng_state_all'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng_state_all']])
    random.setstate(payload['python_rng_state'])
    ns = payload['numpy_rng_state']
    np.random.set_state((ns[0], ns[1].numpy(), ns[2], ns[3], ns[4]))
    return payload['epoch']+1, payload['best_val_accuracy'], payload['history']


def fit(cfg, model, train_loader, val_loader, output, inventory, device, stop_after=None, *,
        train_subset_sha256=None):
    """Same epoch/optimizer primitives as CIFAR; separate identity and checkpoints."""
    output = Path(output)
    subset_identity = {} if train_subset_sha256 is None else dict(train_subset_sha256=train_subset_sha256)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    amp = cfg.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    compare_no_context = cfg.baseline != 'no_context'
    if cfg.baseline == 'two_stage':
        criterion = TACSLoss(cfg.policy_weight, aux_query_weight=cfg.aux_query_weight)
    elif cfg.baseline == 'cosine_top1':
        criterion = PairedClassificationLoss(cfg.aux_query_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    start, best, history = 1, -1., []
    latest = output/'checkpoint_latest.pt'
    if latest.is_file():
        start, best, history = restore(latest, model, optimizer, scheduler, scaler, cfg,
                                       train_subset_sha256=train_subset_sha256)
        print(f'RESUME {cfg.baseline}: next epoch {start}/{cfg.epochs}', flush=True)
    elif any((output/name).exists() for name in ('checkpoint_best.pt', 'metrics.json')):
        raise ValueError('Partial run has no latest checkpoint; do not overwrite')
    end = min(cfg.epochs, stop_after if stop_after is not None else cfg.epochs)
    for epoch in range(start, end+1):
        began = time.perf_counter()
        print(f'[{cfg.baseline}] Epoch {epoch}/{cfg.epochs} TRAIN START ({len(train_loader)} batches)', flush=True)
        lr = optimizer.param_groups[0]['lr']
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler,
            amp_enabled=amp, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            advantage_scope=cfg.advantage_scope)
        print(f'[{cfg.baseline}] VALIDATION START', flush=True)
        va = evaluate(model, val_loader, criterion, device, amp_enabled=amp,
                      compare_no_context=compare_no_context)
        scheduler.step()
        history.append(dict(epoch=epoch, lr=lr, train_acc=tr['accuracy'], train_loss=tr['loss'],
                            val_acc=va['accuracy'], val_loss=va['loss'],
                            val_macc=va['accuracy'], seconds=time.perf_counter()-began))
        if compare_no_context:
            history[-1].update(
                val_no_context_acc=va['no_context_accuracy'],
                val_context_helped=va['context_helped'],
                val_context_hurt=va['context_hurt'],
            )
        if not all(np.isfinite(x) for x in (tr['loss'], va['loss'], va['accuracy'])):
            raise ValueError('Nonfinite training/validation metrics')
        if va['accuracy'] > best:
            best = va['accuracy']
            _atomic_torch_save(dict(epoch=epoch, config=asdict(cfg), classes=inventory['classes'],
                model_state_dict=model.state_dict(), val_accuracy=best, protocol=PROTOCOL, **subset_identity),
                output/'checkpoint_best.pt')
            if cfg.baseline == 'no_context':
                # BaselineModel.encoder is the frozen pretrained retriever, not the task backbone.
                _atomic_torch_save(dict(encoder_state_dict=model.encoder.state_dict(), metadata=dict(
                    dataset=cfg.dataset, protocol=PROTOCOL, inventory_sha256=inventory['inventory_sha256'],
                    origin='frozen DINOv3 retriever stored with No-context run; not a fine-tuned task encoder')),
                    output/'best_encoder.pt')
        ns = np.random.get_state()
        _atomic_torch_save(dict(epoch=epoch, config=asdict(cfg), history=history, best_val_accuracy=best,
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
            torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            python_rng_state=random.getstate(), numpy_rng_state=(ns[0], torch.from_numpy(ns[1].copy()), ns[2], ns[3], ns[4]),
            **subset_identity), latest)
        atomic_json(output/'metrics.json', dict(config=asdict(cfg), history=history, best_val_acc=best,
            classes=inventory['classes'], protocol=PROTOCOL, test_result=None,
            note='Validation is balanced: 5 per class, so validation Acc equals mAcc. No test evaluation.',
            **subset_identity))
        print(f'[{cfg.baseline}] epoch {epoch} | train={tr["accuracy"]:.2%} | val={va["accuracy"]:.2%} | saved latest | {history[-1]["seconds"]:.1f}s', flush=True)
    if len(history) == cfg.epochs:
        report = dict(status='complete', config=asdict(cfg), protocol=PROTOCOL, test_evaluated=False,
            best_val_accuracy=best, checkpoint_sha256=digest(output/'checkpoint_best.pt'),
            metrics_sha256=digest(output/'metrics.json'),
            reference_encoder_sha256=digest(output/'best_encoder.pt') if cfg.baseline=='no_context' else None,
            **subset_identity)
        atomic_json(output/'result.json', report)
        print(f'COMPLETE {cfg.baseline}: {best:.2%} validation (not test)', flush=True)
        return report
    print(f'PAUSED at epoch {len(history)}/{cfg.epochs}; same config resumes automatically.', flush=True)
    return dict(status='partial', last_epoch=len(history))


def run(data_root, output_root, baseline, epochs=100, stop_after=None, *,
        aux_query_weight=0.5, batch_size=4, warmup_epochs=None, train_per_class=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no silent CPU fallback')
    if batch_size not in (2, 4):
        raise ValueError('Caltech microbatch size must be 2 or 4')
    schedule = dict(epochs=epochs, warmup_epochs=min(5, epochs-1) if warmup_epochs is None else warmup_epochs,
                    batch_size=batch_size, gradient_accumulation_steps=64 // batch_size)
    control_cfg = Config(baseline='no_context', **schedule)
    if baseline == 'two_stage_anchored':
        cfg = AnchoredConfig(baseline='two_stage', aux_query_weight=aux_query_weight, **schedule)
    elif baseline == 'cosine_top1':
        cfg = AnchoredConfig(baseline=baseline, aux_query_weight=aux_query_weight,
                             topk_candidates=1, policy_weight=0., **schedule)
    else:
        cfg = Config(baseline=baseline, **schedule)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    output_root = Path(output_root)
    inventory, split = prepare(data_root, output_root/'split.json')
    train_indices, subset_identity = prepare_train_subset(output_root, inventory, split, train_per_class)
    subset_sha256 = subset_identity.get('train_subset_sha256')
    reference = output_root/'no_context/best_encoder.pt'
    if cfg.baseline != 'no_context':
        control = json.loads((output_root/'no_context/result.json').read_text(encoding='utf-8'))
        if control.get('status') != 'complete' or control.get('config') != asdict(control_cfg):
            raise ValueError('Complete the matching Caltech No-context run first')
        if digest(output_root/'no_context/checkpoint_best.pt') != control['checkpoint_sha256']:
            raise ValueError('No-context checkpoint hash changed')
        if digest(reference) != control['reference_encoder_sha256']:
            raise ValueError('No-context reference encoder hash changed')
    output = output_root/baseline
    manifest = dict(kind='caltech101_demo', protocol=PROTOCOL, config=asdict(cfg),
        source_sha256=source_digest(), inventory_sha256=inventory['inventory_sha256'],
        split_sha256=digest(output_root/'split.json'),
        reference_sha256=digest(reference) if cfg.baseline!='no_context' else None,
        environment=dict(torch=str(torch.__version__), torchvision=version('torchvision'),
                         timm=version('timm'), numpy=version('numpy'), gpu=torch.cuda.get_device_name(0)),
        test_evaluated=False, **subset_identity)
    if cfg.baseline != 'no_context' and (cfg.baseline == 'cosine_top1' or train_per_class is not None):
        control_manifest = json.loads((output_root/'no_context/experiment_manifest.json').read_text(encoding='utf-8'))
        for key in ('protocol', 'source_sha256', 'inventory_sha256', 'split_sha256', 'environment',
                    'test_evaluated', 'train_per_class', 'train_subset_sha256'):
            if control_manifest.get(key) != manifest.get(key):
                raise ValueError(f'Caltech comparison requires the same No-context {key}')
        if control.get('train_subset_sha256') != subset_sha256:
            raise ValueError('No-context checkpoint report belongs to a different training subset')
    require_manifest(output, manifest)
    if (output/'result.json').is_file():
        result = json.loads((output/'result.json').read_text(encoding='utf-8'))
        metrics = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
        if (result['status'] != 'complete' or result['config'] != asdict(cfg)
                or result['checkpoint_sha256'] != digest(output/'checkpoint_best.pt')
                or result['metrics_sha256'] != digest(output/'metrics.json')
                or (baseline=='no_context' and result['reference_encoder_sha256'] != digest(reference))
                or result.get('train_subset_sha256') != subset_sha256
                or metrics.get('train_subset_sha256') != subset_sha256
                or len(metrics['history']) != epochs or metrics['test_result'] is not None):
            raise ValueError('Invalid completion report')
        print('ALREADY COMPLETE', baseline, result['best_val_accuracy'], flush=True)
        return result
    seed_everything(cfg.seed)
    device = torch.device('cuda')
    loaders = build_loaders(cfg, data_root, inventory, split, reference, device, train_indices=train_indices)
    model = build_model(cfg, device)
    return fit(cfg, model, *loaders, output, inventory, device, stop_after, train_subset_sha256=subset_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--download', action='store_true')
    baselines = {
        'both': ('no_context', 'two_stage'),
        'pilot': ('no_context', 'two_stage', 'two_stage_anchored'),
        'cosine_pilot': ('no_context', 'cosine_top1'),
    }
    parser.add_argument('--baseline',
                        choices=['no_context', 'two_stage', 'two_stage_anchored', 'cosine_top1', *baselines],
                        default='both', help='cosine_pilot runs No-context and fixed cosine top-1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, choices=(2, 4), default=4,
                        help='microbatch size, with accumulation to 64 images')
    parser.add_argument('--warmup-epochs', type=int)
    parser.add_argument('--train-per-class', type=int,
                        help=f'restrict labeled training queries per class (default: {TRAIN_PER_CLASS})')
    parser.add_argument('--aux-query-weight', type=float, default=0.5,
                        help='query-only cross-entropy weight in the anchored arm')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.epochs < 1 or (args.stop_after is not None and not 1 <= args.stop_after <= args.epochs):
        parser.error('Invalid epochs/stop-after; stop-after is an absolute epoch, not extra epochs')
    if args.warmup_epochs is not None and not 0 <= args.warmup_epochs < args.epochs:
        parser.error('Warmup must be nonnegative and shorter than the training schedule')
    if args.train_per_class is not None and not 1 <= args.train_per_class <= TRAIN_PER_CLASS:
        parser.error(f'train-per-class must be between 1 and {TRAIN_PER_CLASS}')
    if args.baseline in ('pilot', 'two_stage_anchored', 'cosine_pilot', 'cosine_top1') and (
            not math.isfinite(args.aux_query_weight) or args.aux_query_weight <= 0):
        parser.error('Anchored classification requires a positive finite query-only loss weight')
    try:
        inventory, split = prepare(args.data_root, args.output_root/'split.json', args.download)
        train_indices, _ = prepare_train_subset(args.output_root, inventory, split, args.train_per_class)
        sizes = {k: len(split[k]) for k in ('candidate_pool', 'validation', 'test')}
        print('REAL DATA:', len(inventory['classes']), 'classes;',
              dict(query_train=len(train_indices), **sizes), flush=True)
        if not args.prepare_only:
            for baseline in baselines.get(args.baseline, (args.baseline,)):
                print(f'RUN {baseline} ({args.epochs} epochs)', flush=True)
                result = run(args.data_root, args.output_root, baseline, args.epochs, args.stop_after,
                             aux_query_weight=args.aux_query_weight, batch_size=args.batch_size,
                             warmup_epochs=args.warmup_epochs, train_per_class=args.train_per_class)
                if result['status'] != 'complete':
                    break
                gc.collect()
                torch.cuda.empty_cache()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'CALTECH DEMO FAILED: {exc}\nPreserve outputs; no fallback to CIFAR or artificial data.\n')


if __name__ == '__main__':
    main()
