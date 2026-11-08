"""One isolated validation-only policy-weight experiment (also embedded in Kaggle)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tacs.multiseed import RunnerConfig, build_train_command, is_complete_run

ENCODER_SHA256 = 'ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64'
WEIGHTS = (1.0, 0.1, 0.03)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_config(project, output, baseline, weight):
    if baseline not in ('tacs', 'two_stage') or weight not in WEIGHTS:
        raise ValueError('Only tacs/two_stage and policy weights 1.0, 0.1, 0.03 are allowed')
    return RunnerConfig(
        dataset='cifar100', backbone='dinov3_vits16', task_network_version='role_v2',
        seeds=(44,), baselines=(baseline,), outputs_root=Path(output),
        data_root=Path(project) / 'data',
        split_file=Path(project) / 'data/splits/cifar100_seed42.json',
        device='cuda', epochs=20, batch_size=4, gradient_accumulation_steps=16,
        lr=1e-4, warmup_epochs=5, num_workers=2, eval_test=False,
        policy_weight=weight, resume=True,
    )


def guard_manifest(output, manifest):
    """Never adopt an old run without its matching experiment identity."""
    output = Path(output)
    target = output / 'experiment_manifest.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Experiment manifest mismatch; use a new output directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Nonempty output without manifest; refusing to overwrite/adopt it')
        output.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def run(project, output, baseline, weight, source_sha256, encoder=None):
    import torch
    import timm
    import torchvision

    project, output = Path(project), Path(output)
    config = make_config(project, output, baseline, weight)
    encoder_hash = None
    if baseline == 'two_stage':
        if encoder is None or not Path(encoder).is_file():
            raise ValueError('Two-Stage requires the original pilot No-context best_encoder.pt')
        encoder_hash = sha256(encoder)
        if encoder_hash != ENCODER_SHA256:
            raise ValueError('Wrong No-context encoder SHA256; do not use Random/legacy weights')
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a CUDA GPU before training')
    command = build_train_command(config, 44, baseline)
    if baseline == 'two_stage':
        command[command.index('--standard-encoder') + 1] = str(encoder)
    manifest = {
        'schema': 1, 'source_sha256': source_sha256,
        'split_sha256': sha256(config.resolved_split_file),
        'encoder_sha256': encoder_hash, 'baseline': baseline, 'policy_weight': weight,
        'command_args': command[1:], 'test_evaluated': False,
        'environment': {'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                        'timm': timm.__version__, 'gpu': torch.cuda.get_device_name(0)},
        'initialization': 'pretrained backbone plus seed-44 fresh task/selector; no warm-start',
    }
    guard_manifest(output, manifest)
    run_dir = config.run_dir(44, baseline)
    if not is_complete_run(config, 44, baseline):
        if run_dir.exists() and any(run_dir.iterdir()) and not any(
            (run_dir / name).is_file() for name in ('checkpoint_latest.pt', 'checkpoint_best.pt')
        ):
            raise RuntimeError('Existing partial run has no checkpoint; preserve it and use a fresh experiment directory')
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        print('TRAIN COMMAND:', command, flush=True)
        with (output / 'train.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=project, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                if process.wait() != 0:
                    raise RuntimeError('Training failed; retain output and inspect train.log')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    if not is_complete_run(config, 44, baseline):
        raise RuntimeError('Training did not produce a complete, matching run')
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    if len(metrics['history']) != 20 or metrics.get('test_result') is not None:
        raise RuntimeError('Unexpected epoch count or forbidden test evaluation')
    for name in ('checkpoint_best.pt', 'checkpoint_latest.pt', 'best_encoder.pt'):
        if not (run_dir / name).is_file():
            raise RuntimeError(f'Missing required artifact: {name}')
    report = {
        'status': 'complete', 'baseline': baseline, 'policy_weight': weight,
        'training_seed': 44, 'epochs': 20, 'test_evaluated': False,
        'best_val_accuracy': metrics['best_val_acc'],
        'final_val_accuracy': metrics['history'][-1]['val_acc'],
        'checkpoint_sha256': sha256(run_dir / 'checkpoint_best.pt'),
        'metrics_sha256': sha256(run_dir / 'metrics.json'),
        'run_dir': str(run_dir), 'source_sha256': source_sha256,
        'note': 'Single-seed validation pilot, not test or proof of improvement. Paired utility audit still required.',
    }
    (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('POLICY WEIGHT PILOT COMPLETE:', json.dumps(report, indent=2), flush=True)
    return report
