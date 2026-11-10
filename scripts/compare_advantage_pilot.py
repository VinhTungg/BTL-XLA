"""Validate two normalization pilot outputs before a screening comparison."""
import argparse
import json
from pathlib import Path

from policy_weight_pilot import sha256


def read_run(root, scope):
    root = Path(root)
    manifest = json.loads((root/'experiment_manifest.json').read_text())
    result = json.loads((root/'result.json').read_text())
    metrics_path = root/'run/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    if (manifest.get('kind') != 'advantage_normalization_pilot' or result.get('status') != 'complete'
            or result.get('advantage_scope') != scope or manifest['config'].get('advantage_scope') != scope
            or metrics['config'] != manifest['config'] or len(metrics['history']) != 20
            or metrics['history'][-1]['epoch'] != 20 or metrics.get('test_result') is not None
            or manifest.get('test_evaluated') is not False or result.get('test_evaluated') is not False):
        raise ValueError('Incomplete, wrong-scope, old-recipe or test-evaluated run')
    for path, key in [(root/'run/checkpoint_best.pt', 'checkpoint_sha256'), (metrics_path, 'metrics_sha256')]:
        if sha256(path) != result[key]:
            raise ValueError('Artifact hash mismatch')
    return manifest, result, metrics


def compare(microbatch, effective):
    a, ar, am = read_run(microbatch, 'microbatch')
    b, br, bm = read_run(effective, 'effective_batch')
    for key in ['source_tree_sha256', 'source_bundle_sha256', 'split_sha256', 'encoder_sha256', 'environment', 'initialization']:
        if a[key] != b[key]:
            raise ValueError(f'Uncontrolled experiment difference: {key}')
    ac, bc = dict(a['config']), dict(b['config'])
    for cfg in (ac, bc):
        for key in ['advantage_scope', 'output_dir']:
            cfg.pop(key)
    if ac != bc:
        raise ValueError('Recipe differs beyond advantage scope/output')
    return dict(status='validated_screening_comparison', microbatch=ar, effective_batch=br,
                best_validation_delta_pp=100*(br['best_val_accuracy']-ar['best_val_accuracy']),
                total_train_seconds=dict(microbatch=am['total_time_s'], effective_batch=bm['total_time_s']),
                warning='Single training seed. Accuracy alone does not prove selector utility. Audit both best checkpoints before multi-seed decision; no test-based selection.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--microbatch', required=True, type=Path)
    parser.add_argument('--effective-batch', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.microbatch, args.effective_batch), indent=2))
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'COMPARISON REFUSED: {error}\n')
