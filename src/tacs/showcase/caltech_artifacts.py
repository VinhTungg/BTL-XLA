"""Read-only verification of the recovered Caltech full100 experiment."""
from dataclasses import asdict
import json
from pathlib import Path

from ..caltech.data import digest, PROTOCOL, validate_split
from ..caltech.run import Config

SOURCE_SHA256 = 'ad14eece024694224d7ae1eaa2544e532cd91038261ee2931eff4300406890a9'
CHECKPOINTS = {
    'no_context':'f2fa1b9f26c162eccf962e02679c08f76545349d69d99e8448b96c00faa9073d',
    'two_stage':'27091fd7ca2405ece49d2fd4ce8b2d08f98c6e1fedad52876cc06ab9b25e5dc5',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_artifacts(root, verify_latest=False):
    """Check reports, exact recipe, source identity, split and real checkpoint payloads."""
    import torch
    root=Path(root)
    inventory=read_json(root/'inventory.json')
    split=read_json(root/'split.json')
    validate_split(split,inventory)
    if len(inventory['classes'])!=101 or len(inventory['items'])!=8677:
        raise ValueError('Wrong Caltech class map/inventory size')
    reference=digest(root/'no_context/best_encoder.pt')
    summaries={}
    common=None
    for name in CHECKPOINTS:
        folder=root/name
        manifest=read_json(folder/'experiment_manifest.json')
        report=read_json(folder/'result.json')
        metrics=read_json(folder/'metrics.json')
        cfg=asdict(Config(baseline=name))
        if (manifest.get('kind')!='caltech101_demo' or manifest.get('config')!=cfg
                or manifest.get('source_sha256')!=SOURCE_SHA256
                or manifest.get('inventory_sha256')!=inventory['inventory_sha256']
                or manifest.get('split_sha256')!=digest(root/'split.json')
                or manifest.get('reference_sha256')!=(reference if name=='two_stage' else None)
                or manifest.get('test_evaluated') is not False or manifest.get('protocol')!=PROTOCOL
                or report.get('status')!='complete' or report.get('config')!=cfg
                or report.get('test_evaluated') is not False or report.get('protocol')!=PROTOCOL
                or metrics.get('config')!=cfg or metrics.get('protocol')!=PROTOCOL
                or metrics.get('classes')!=inventory['classes'] or metrics.get('test_result') is not None):
            raise ValueError(f'{name}: experiment identity/completion mismatch')
        if common is not None and common!=manifest['environment']:
            raise ValueError('Control/treatment environments differ')
        common=manifest['environment']
        if (report.get('checkpoint_sha256')!=CHECKPOINTS[name]
                or digest(folder/'checkpoint_best.pt')!=CHECKPOINTS[name]
                or digest(folder/'metrics.json')!=report.get('metrics_sha256')
                or (name=='no_context' and reference!=report.get('reference_encoder_sha256'))):
            raise ValueError(f'{name}: checkpoint/metrics/reference hash mismatch')
        history=metrics['history']
        if [row['epoch'] for row in history]!=list(range(1,101)):
            raise ValueError(f'{name}: missing/duplicate epochs')
        best=max(row['val_acc'] for row in history)
        if best!=report.get('best_val_accuracy') or best!=metrics.get('best_val_acc'):
            raise ValueError(f'{name}: best validation mismatch')
        payload=torch.load(folder/'checkpoint_best.pt',map_location='cpu',weights_only=True)
        best_epoch=next(row['epoch'] for row in history if row['val_acc']==best)
        if (payload.get('config')!=cfg or payload.get('classes')!=inventory['classes']
                or payload.get('protocol')!=PROTOCOL or payload.get('epoch')!=best_epoch
                or payload.get('val_accuracy')!=best or not payload.get('model_state_dict')):
            raise ValueError(f'{name}: best checkpoint payload mismatch')
        del payload
        if verify_latest:
            payload=torch.load(folder/'checkpoint_latest.pt',map_location='cpu',weights_only=True)
            if (payload.get('epoch')!=100 or payload.get('history')!=history
                    or payload.get('config')!=cfg or payload.get('best_val_accuracy')!=best
                    or not all(key in payload for key in ('optimizer_state_dict','scheduler_state_dict',
                        'scaler_state_dict','torch_rng_state','python_rng_state','numpy_rng_state'))):
                raise ValueError(f'{name}: latest checkpoint is not a full completed run')
            del payload
        summaries[name]=dict(epochs=100,best_epoch=best_epoch,best_val_accuracy=best,
            final_val_accuracy=history[-1]['val_acc'],train_seconds=sum(r['seconds'] for r in history),
            checkpoint_sha256=CHECKPOINTS[name],metrics_sha256=report['metrics_sha256'])
    reference_payload=torch.load(root/'no_context/best_encoder.pt',map_location='cpu',weights_only=True)
    meta=reference_payload.get('metadata',{})
    if (meta.get('dataset')!='caltech101' or meta.get('protocol')!=PROTOCOL
            or meta.get('inventory_sha256')!=inventory['inventory_sha256']
            or not reference_payload.get('encoder_state_dict')):
        raise ValueError('Wrong reference encoder metadata')
    return dict(status='verified',dataset='caltech101',protocol=PROTOCOL,
        source_sha256=SOURCE_SHA256,split_sha256=digest(root/'split.json'),
        inventory_sha256=inventory['inventory_sha256'],reference_sha256=reference,
        sizes={k:len(split[k]) for k in ('query_train','candidate_pool','validation','test')},
        classes=101,images=8677,test_evaluated=False,latest_checked=verify_latest,
        environment=common,runs=summaries,
        best_val_delta_pp=round(100*(summaries['two_stage']['best_val_accuracy']-
                                   summaries['no_context']['best_val_accuracy']),4))
