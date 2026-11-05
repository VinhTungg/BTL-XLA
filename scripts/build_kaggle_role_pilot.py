"""Build a self-contained Kaggle notebook from the verified role_v2 source bundle."""

import ast
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main(selector_pilot: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = root / "artifacts/role_v2/TACS_Colab_Bundle.zip"
    raw = bundle.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    setup = f'''from pathlib import Path
import base64, hashlib, io, os, subprocess, sys, zipfile
PROJECT_DIR = Path('/kaggle/working/TACS_role_source')
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
encoded_bundle = {base64.b64encode(raw).decode('ascii')!r}
raw = base64.b64decode(encoded_bundle)
assert hashlib.sha256(raw).hexdigest() == {digest!r}, 'Source bundle checksum mismatch'
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        assert (PROJECT_DIR / name).resolve().is_relative_to(PROJECT_DIR.resolve())
    archive.extractall(PROJECT_DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                'timm>=1.0.20', 'psutil>=5.9', 'setuptools>=68', 'wheel'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(PROJECT_DIR)], check=True)
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR / 'src'))
print('Verified source SHA256:', {digest!r}, flush=True)
'''
    data = '''import torch
from tacs.data import build_cifar100
assert torch.cuda.is_available(), 'Enable GPU accelerator before running'
print('GPU count:', torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print('GPU', i, torch.cuda.get_device_name(i), flush=True)
datasets = build_cifar100(PROJECT_DIR / 'data', PROJECT_DIR / 'data/splits/cifar100_seed42.json', download=True)
print('Real CIFAR-100:', len(datasets.query_train), len(datasets.candidate_pool), len(datasets.validation), flush=True)
del datasets
'''
    train = '''import concurrent.futures
import json
from tacs.multiseed import RunnerConfig, build_train_command, is_complete_run, write_summaries

OUTPUT_ROOT = Path('/kaggle/working/TACS_context_role_pilot')
VERSIONS = ('legacy_v1', 'role_v2')
def run_version(version, gpu):
    config = RunnerConfig(dataset='cifar100', backbone='dinov3_vits16',
        task_network_version=version, seeds=(44,), baselines=('random',),
        outputs_root=OUTPUT_ROOT, data_root=PROJECT_DIR / 'data',
        split_file=PROJECT_DIR / 'data/splits/cifar100_seed42.json',
        device='cuda', epochs=20, batch_size=4, gradient_accumulation_steps=16,
        lr=1e-4, warmup_epochs=5, num_workers=2, eval_test=False)
    if not is_complete_run(config, 44, 'random'):
        command = build_train_command(config, 44, 'random')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        log_dir = OUTPUT_ROOT / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{version}] GPU={gpu}; seed=44; epochs=20; physical batch=4; effective batch=64', flush=True)
        with (log_dir / f'{version}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f'[{version}] {line}', end='', flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f'{version} failed; see {log.name}')
            finally:
                if process.poll() is None:
                    process.terminate()
    assert is_complete_run(config, 44, 'random'), f'Incomplete run: {version}'
    write_summaries(config)
    path = config.run_dir(44, 'random') / 'metrics.json'
    metrics = json.loads(path.read_text())
    return {'version': version, 'best_val_accuracy': metrics['best_val_acc'],
            'final_val_accuracy': metrics['history'][-1]['val_acc'],
            'metrics_path': str(path), 'epochs': len(metrics['history'])}

# Two independent controlled runs, one on each T4. Fall back to serial on one GPU.
workers = min(2, torch.cuda.device_count())
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(run_version, version, i % workers) for i, version in enumerate(VERSIONS)]
    results = [future.result() for future in futures]
report = {'seed': 44, 'pilot_epochs': 20, 'test_evaluated': False, 'runs': results,
          'note': 'Pilot only; same recipe in both versions. Not a 100-epoch paper benchmark.'}
(OUTPUT_ROOT / 'pilot_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('PILOT COMPLETE:', json.dumps(report, indent=2), flush=True)
'''
    if selector_pilot:
        train = (root / 'scripts/kaggle_selector_pilot_train.py').read_text(encoding='utf-8')
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# TACS Random role pilot — seed 44\n",
        "20 epoch/version, pretrained DINOv3, physical batch 4 × accumulation 16. "
        "Real CIFAR-100, validation only. Outputs remain separate from historical runs.\n",
        "Use GPU T4 x2, Internet on, and Save & Run All to preserve outputs.\n"]}]
    if selector_pilot:
        cells[0]['source'] = [
            '# TACS selector pilot — role_v2, seed 44\n',
            '20 epochs each: TACS on GPU 0; No-context then Two-Stage on GPU 1. '
            'Physical batch 4 x accumulation 16, LR 1e-4, warm-up 5. '
            'Real CIFAR-100, validation only; start from pretrained, not Random task weights.\n',
            'Use GPU T4 x2, Internet on, Save & Run All. New output root; preserve Random Version 1.\n',
        ]
    for code in (setup, data, train):
        ast.parse(code)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": code.splitlines(keepends=True)})
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    filename = 'TACS_Kaggle_Selector_Pilot.ipynb' if selector_pilot else 'TACS_Kaggle_Role_Pilot.ipynb'
    target = root / 'artifacts/role_v2' / filename
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(target)
    print('Bundle SHA256:', digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selector-pilot', action='store_true')
    main(parser.parse_args().selector_pilot)
