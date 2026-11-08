"""Build six standalone Kaggle notebooks from current source, not a stale ZIP."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def bundle_source(root):
    paths = sorted((root / 'src/tacs').rglob('*.py')) + [
        root / 'pyproject.toml', root / 'data/splits/cifar100_seed42.json',
        root / 'scripts/policy_weight_pilot.py',
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            # Stable metadata makes rebuilds byte-identical for unchanged source.
            entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    return buffer.getvalue()


def make_notebook(raw, baseline, weight):
    digest = hashlib.sha256(raw).hexdigest()
    tag = str(weight).replace('.', 'p')
    setup = f'''from pathlib import Path
import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # One GPU per independent run; stable resume RNG topology.
SOURCE_SHA256 = {digest!r}
PROJECT_DIR = Path('/kaggle/working/policy_source_' + SOURCE_SHA256[:12])
raw = base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        if not (PROJECT_DIR / name).resolve().is_relative_to(PROJECT_DIR.resolve()):
            raise RuntimeError('Unsafe source archive path')
    archive.extractall(PROJECT_DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                'timm==1.0.30', 'psutil>=5.9', 'setuptools>=68', 'wheel'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(PROJECT_DIR)], check=True)
sys.path.insert(0, str(PROJECT_DIR / 'src'))
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
os.chdir(PROJECT_DIR)
print('Verified source:', SOURCE_SHA256, flush=True)
'''
    config = f'''BASELINE = {baseline!r}
POLICY_WEIGHT = {weight!r}
OUTPUT_ROOT = Path('/kaggle/working/policy_weight_{baseline}_{tag}_seed44')
# Optional: attached PREVIOUS OUTPUT ROOT containing experiment_manifest.json.
# Copy the whole experiment folder, not just checkpoint_best.pt. Never use the old selector pilot here.
RESUME_FROM = None  # Example: '/kaggle/input/my-interrupted-run/policy_weight_{baseline}_{tag}_seed44'
# Two-Stage only: exact original No-context encoder from selector role_v2 pilot Version 2.
ENCODER_INPUT = None  # Optional explicit '/kaggle/input/.../no_context/best_encoder.pt'
from policy_weight_pilot import ENCODER_SHA256, sha256
ENCODER = None
if BASELINE == 'two_stage':
    candidates = [Path(ENCODER_INPUT)] if ENCODER_INPUT else sorted(Path('/kaggle/input').rglob('best_encoder.pt'))
    matches = [p for p in candidates if p.is_file() and sha256(p) == ENCODER_SHA256]
    if not matches:
        raise RuntimeError('Attach original No-context best_encoder.pt (selector pilot V2); no matching SHA256 found')
    ENCODER = PROJECT_DIR / 'reference/no_context/best_encoder.pt'
    ENCODER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], ENCODER)
    print('Verified frozen retrieval encoder:', ENCODER_SHA256, flush=True)
if RESUME_FROM is not None:
    previous = Path(RESUME_FROM)
    if not (previous / 'experiment_manifest.json').is_file():
        raise RuntimeError('Resume root is missing experiment_manifest.json')
    if OUTPUT_ROOT.exists():
        raise RuntimeError('Output already exists; set RESUME_FROM=None to resume this local copy')
    shutil.copytree(previous, OUTPUT_ROOT)
print('Run:', BASELINE, POLICY_WEIGHT, 'Output:', OUTPUT_ROOT, flush=True)
'''
    data = '''import torch
from tacs.data import build_cifar100
if not torch.cuda.is_available():
    raise RuntimeError('Enable GPU before running')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
datasets = build_cifar100(PROJECT_DIR / 'data', PROJECT_DIR / 'data/splits/cifar100_seed42.json', download=True)
print('Real data:', len(datasets.query_train), len(datasets.candidate_pool), len(datasets.validation), flush=True)
del datasets
'''
    train = '''from policy_weight_pilot import run
result = run(PROJECT_DIR, OUTPUT_ROOT, BASELINE, POLICY_WEIGHT, SOURCE_SHA256, ENCODER)
'''
    verify = '''report = json.loads((OUTPUT_ROOT / 'result.json').read_text())
assert report['status'] == 'complete' and report['epochs'] == 20 and not report['test_evaluated']
checkpoint = Path(report['run_dir']) / 'checkpoint_best.pt'
assert sha256(checkpoint) == report['checkpoint_sha256']
print(json.dumps(report, indent=2), flush=True)
print('Keep the ENTIRE output experiment directory, including latest checkpoint and manifest.')
print('No duplicate ZIP is created, to avoid doubling output storage.')
'''
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        f'# Policy weight pilot: {baseline}, lambda={weight}, seed 44\n',
        '20 epochs; role_v2; DINOv3 ViT-S/16; batch 4 x accumulation 16; LR 1e-4; warm-up 5.\n',
        'Validation only. ONE run per notebook. No old task weights are loaded.\n',
        'Enable GPU and Internet. Check remaining quota, then Save & Run All. '
        'Two-Stage requires the original No-context encoder as an attached dataset.\n',
        'Resume is allowed only from this exact experiment with matching source, recipe and environment. '
        'A new Kaggle session does not automatically retain /kaggle/working.\n',
    ]}]
    for i, code in enumerate((setup, config, data, train, verify)):
        ast.parse(code)
        cells.append({'cell_type': 'code', 'id': f'policy-cell-{i}', 'metadata': {},
                      'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)})
    cells[0]['id'] = 'policy-intro'
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}


def main():
    output = ROOT / 'artifacts/policy_weight_pilot'
    output.mkdir(parents=True, exist_ok=True)
    raw = bundle_source(ROOT)
    for baseline in ('tacs', 'two_stage'):
        for weight in (1.0, 0.1, 0.03):
            target = output / f'Policy_{baseline}_w{str(weight).replace(".", "p")}_seed44.ipynb'
            target.write_text(json.dumps(make_notebook(raw, baseline, weight), ensure_ascii=False, indent=1), encoding='utf-8')
            print(target)
    print('Source SHA256:', hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
