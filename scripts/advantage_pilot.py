"""Matched normalization pilot, one isolated run; validation only."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from contextlib import redirect_stdout

from tacs.train import ExperimentConfig, run_experiment
from policy_weight_pilot import ENCODER_SHA256, guard_manifest, sha256


def make_config(project, output, scope):
    if scope not in ('microbatch', 'effective_batch'):
        raise ValueError('Unknown advantage scope')
    return ExperimentConfig(baseline='two_stage', dataset='cifar100', backbone='dinov3_vits16',
        task_network_version='role_v2', seed=44, epochs=20, batch_size=4,
        gradient_accumulation_steps=16, advantage_scope=scope, lr=1e-4, min_lr=0.,
        warmup_epochs=5, weight_decay=1e-4, policy_weight=1., topk_candidates=32,
        selector_temperature=.1, num_workers=2, device='cuda', amp=True,
        data_root=str(Path(project)/'data'), split_file=str(Path(project)/'data/splits/cifar100_seed42.json'),
        output_dir=str(Path(output)/'run'), standard_encoder_path=str(Path(project)/'reference/no_context/best_encoder.pt'),
        eval_test=False, resume=True)


def source_digest(project):
    root = Path(project)
    digest = hashlib.sha256()
    paths = sorted((root/'src/tacs').glob('*.py')) + [root/'scripts/advantage_pilot.py', root/'scripts/policy_weight_pilot.py']
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Tee:
    def __init__(self, log, terminal):
        self.log, self.terminal = log, terminal
    def write(self, text):
        self.log.write(text)
        self.terminal.write(text)
        self.flush()
        return len(text)
    def flush(self):
        self.log.flush()
        self.terminal.flush()


def run(project, output, scope, encoder, bundle_sha256=None):
    import torch
    import timm
    import torchvision
    project, output, encoder = Path(project).resolve(), Path(output).resolve(), Path(encoder).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('Enable CUDA GPU; no automatic CPU training fallback')
    if sha256(encoder) != ENCODER_SHA256:
        raise ValueError('Reference encoder hash mismatch')
    cfg = make_config(project, output, scope)
    cfg.standard_encoder_path = str(encoder)
    manifest = dict(schema=1, kind='advantage_normalization_pilot', config=asdict(cfg),
        source_tree_sha256=source_digest(project), source_bundle_sha256=bundle_sha256,
        split_sha256=sha256(cfg.split_file), encoder_sha256=ENCODER_SHA256, test_evaluated=False,
        environment=dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                         timm=str(timm.__version__), gpu=torch.cuda.get_device_name(0)),
        initialization='fresh pretrained task/selector, no task warm-start; scope ablation only')
    guard_manifest(output, manifest)
    run_dir = Path(cfg.output_dir)
    if (output/'result.json').is_file():
        report = json.loads((output/'result.json').read_text())
        metrics = json.loads((run_dir/'metrics.json').read_text())
        if (report.get('status') != 'complete' or report.get('advantage_scope') != scope
                or report.get('epochs') != 20 or report.get('test_evaluated') is not False
                or metrics['config'] != asdict(cfg) or len(metrics['history']) != 20
                or metrics.get('test_result') is not None
                or sha256(run_dir/'checkpoint_best.pt') != report.get('checkpoint_sha256')
                or sha256(run_dir/'metrics.json') != report.get('metrics_sha256')):
            raise ValueError('Existing completion report is invalid')
        print('Already complete:', report, flush=True)
        return report
    if run_dir.exists() and any(run_dir.iterdir()) and not (run_dir/'checkpoint_latest.pt').is_file():
        raise ValueError('Partial output has no latest checkpoint; preserve and use new output root')
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise RuntimeError('Need at least 2 GiB free before pilot; no files deleted automatically')
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    with (output/'train.log').open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(log, sys.stdout)):
            metrics = run_experiment(cfg)
    if (len(metrics['history']) != 20 or metrics['history'][-1]['epoch'] != 20
            or metrics['test_result'] is not None or metrics['config'] != asdict(cfg)):
        raise RuntimeError('Incomplete or mismatched run; no completion marker written')
    report = dict(status='complete', kind='advantage_normalization_pilot', advantage_scope=scope,
                  training_seed=44, epochs=20, test_evaluated=False,
                  best_val_accuracy=metrics['best_val_acc'], final_val_accuracy=metrics['history'][-1]['val_acc'],
                  checkpoint_sha256=sha256(run_dir/'checkpoint_best.pt'), metrics_sha256=sha256(run_dir/'metrics.json'),
                  run_dir=str(run_dir), note='One-seed validation pilot, not proof of improvement. Run paired utility audit before multi-seed decision.')
    (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ADVANTAGE PILOT COMPLETE:', json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', required=True, choices=['microbatch', 'effective_batch'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output, args.scope, args.encoder)
