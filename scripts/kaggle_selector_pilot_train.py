"""Kaggle training cell; executed after the verified setup and data cells."""

import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import torch
from tacs.multiseed import RunnerConfig, build_train_command, is_complete_run, write_summaries

PROJECT_DIR = Path('/kaggle/working/TACS_role_source')
OUTPUT_ROOT = Path('/kaggle/working/TACS_selector_role_v2_pilot')
CONFIG = RunnerConfig(
    dataset='cifar100', backbone='dinov3_vits16', task_network_version='role_v2',
    seeds=(44,), baselines=('no_context', 'tacs', 'two_stage'),
    outputs_root=OUTPUT_ROOT, data_root=PROJECT_DIR / 'data',
    split_file=PROJECT_DIR / 'data/splits/cifar100_seed42.json',
    device='cuda', epochs=20, batch_size=4, gradient_accumulation_steps=16,
    lr=1e-4, warmup_epochs=5, num_workers=2, eval_test=False,
)


def run_baseline(baseline, gpu):
    if baseline == 'two_stage':
        # Never relabel a Random encoder or silently borrow incompatible weights.
        if not is_complete_run(CONFIG, 44, 'no_context'):
            raise RuntimeError('Two-Stage requires this pilot No-context run to be complete')
    if not is_complete_run(CONFIG, 44, baseline):
        command = build_train_command(CONFIG, 44, baseline)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        log_dir = OUTPUT_ROOT / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{baseline}] GPU={gpu}; role_v2; seed=44; epochs=20; effective batch=64', flush=True)
        print(f'[{baseline}] Command: {command}', flush=True)
        with (log_dir / f'{baseline}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f'[{baseline}] {line}', end='', flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f'{baseline} failed; see {log.name}')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    if not is_complete_run(CONFIG, 44, baseline):
        raise RuntimeError(f'Incomplete run: {baseline}')
    path = CONFIG.run_dir(44, baseline) / 'metrics.json'
    metrics = json.loads(path.read_text(encoding='utf-8'))
    history = metrics['history']
    if len(history) != 20 or metrics.get('test_result') is not None:
        raise RuntimeError(f'Unexpected history/test evaluation: {baseline}')
    result = {'baseline': baseline, 'version': 'role_v2',
              'best_val_accuracy': metrics['best_val_acc'],
              'final_val_accuracy': history[-1]['val_acc'],
              'metrics_path': str(path), 'epochs': len(history)}
    # Separate per-run reports survive a later dependency failure.
    (OUTPUT_ROOT / f'{baseline}_result.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    return result


def run_retrieval_chain(gpu):
    return [run_baseline('no_context', gpu), run_baseline('two_stage', gpu)]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU before running this pilot')
    if torch.cuda.device_count() >= 2:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tacs_future = pool.submit(run_baseline, 'tacs', 0)
            retrieval_future = pool.submit(run_retrieval_chain, 1)
            results = [tacs_future.result(), *retrieval_future.result()]
    else:
        results = [run_baseline(name, 0) for name in CONFIG.baselines]
    # Aggregate only after workers finish: avoid concurrent summary writes.
    write_summaries(CONFIG)
    report = {'seed': 44, 'pilot_epochs': 20, 'test_evaluated': False,
              'runs': results,
              'note': 'Validation pilot only. Compare with Random role_v2 Version 1; not a paper benchmark.'}
    (OUTPUT_ROOT / 'selector_pilot_comparison.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('SELECTOR PILOT COMPLETE:', json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
