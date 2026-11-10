"""Build two isolated Kaggle notebooks with an embedded, hashed source bundle."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def bundle_source(root):
    paths = sorted((root/'src/tacs').glob('*.py')) + [root/'pyproject.toml',
        root/'data/splits/cifar100_seed42.json', root/'scripts/advantage_pilot.py', root/'scripts/policy_weight_pilot.py']
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    return data.getvalue()


def make_notebook(raw, scope):
    if scope not in ('microbatch', 'effective_batch'):
        raise ValueError('Invalid scope')
    digest = hashlib.sha256(raw).hexdigest()
    setup = f'''from pathlib import Path
import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
SOURCE_SHA256 = {digest!r}
PROJECT_DIR = Path('/kaggle/working/advantage_source_' + SOURCE_SHA256[:12])
raw = base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert all((PROJECT_DIR/n).resolve().is_relative_to(PROJECT_DIR.resolve()) for n in archive.namelist())
    archive.extractall(PROJECT_DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.30', 'psutil>=5.9', 'setuptools>=68', 'wheel'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(PROJECT_DIR)], check=True)
sys.path.insert(0, str(PROJECT_DIR/'src'))
sys.path.insert(0, str(PROJECT_DIR/'scripts'))
os.chdir(PROJECT_DIR)
print('Verified source bundle:', SOURCE_SHA256, flush=True)
'''
    config = f'''SCOPE = {scope!r}
OUTPUT_ROOT = Path('/kaggle/working/advantage_' + SCOPE + '_seed44')
# Attach original No-context best_encoder.pt from TACS_Role_Pilot_seed44 as an input.
ENCODER_INPUT = None  # Optional exact /kaggle/input/.../best_encoder.pt
# For interrupted SAME experiment only; attach entire output including manifest and latest.
RESUME_FROM = None
from policy_weight_pilot import ENCODER_SHA256, sha256
candidates = [Path(ENCODER_INPUT)] if ENCODER_INPUT else sorted(Path('/kaggle/input').rglob('best_encoder.pt'))
matches = [p for p in candidates if p.is_file() and sha256(p) == ENCODER_SHA256]
if not matches:
    raise RuntimeError('Attach original No-context encoder; no matching SHA256 found')
ENCODER = PROJECT_DIR/'reference/no_context/best_encoder.pt'
ENCODER.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(matches[0], ENCODER)
if RESUME_FROM is not None:
    previous = Path(RESUME_FROM)
    if OUTPUT_ROOT.exists() or not (previous/'experiment_manifest.json').is_file():
        raise RuntimeError('Unsafe resume: expected new output and previous matching experiment manifest')
    shutil.copytree(previous, OUTPUT_ROOT)
print('Scope:', SCOPE, 'Output:', OUTPUT_ROOT, flush=True)
'''
    data = '''import torch
from tacs.data import build_cifar100
if not torch.cuda.is_available():
    raise RuntimeError('Enable a CUDA GPU')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
datasets = build_cifar100(PROJECT_DIR/'data', PROJECT_DIR/'data/splits/cifar100_seed42.json', download=True)
print('Real data:', len(datasets.query_train), len(datasets.candidate_pool), len(datasets.validation), flush=True)
del datasets
'''
    train = '''from advantage_pilot import run
result = run(PROJECT_DIR, OUTPUT_ROOT, SCOPE, ENCODER, SOURCE_SHA256)
'''
    verify = '''report = json.loads((OUTPUT_ROOT/'result.json').read_text())
assert report['status'] == 'complete' and report['advantage_scope'] == SCOPE
assert report['epochs'] == 20 and not report['test_evaluated']
assert sha256(Path(report['run_dir'])/'checkpoint_best.pt') == report['checkpoint_sha256']
print(json.dumps(report, indent=2), flush=True)
print('Download entire experiment output; keep latest, best, metrics, config, manifest and log.')
print('Do not compare with older source pilot as if normalization were the only difference.')
'''
    intro = f'''# Advantage normalization pilot — {scope}, seed 44
Two-Stage role_v2; policy weight 1.0; 20 epochs; microbatch 4 × accumulation 16;
LR 1e-4, warm-up 5, K=32. Validation only. No warm-start of task/selector.
Enable GPU/Internet and attach the original No-context encoder. One notebook = one run.
Run BOTH branches from this same bundle before claiming a normalization effect.
Effective-batch uses two forward passes, so it may take longer; no accuracy promise.
Use Save & Run All after inspecting settings; retain output if interrupted.
'''
    cells = [dict(cell_type='markdown', id='intro', metadata={}, source=intro.splitlines(True))]
    for index, code in enumerate([setup, config, data, train, verify]):
        ast.parse(code)
        cells.append(dict(cell_type='code', id=f'advantage-{index}', metadata={}, execution_count=None,
                          outputs=[], source=code.splitlines(True)))
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')))


def main():
    raw = bundle_source(ROOT)
    output = ROOT/'artifacts/effective_batch_pilot'
    output.mkdir(parents=True, exist_ok=True)
    for scope in ('microbatch', 'effective_batch'):
        path = output/f'Advantage_{scope}_seed44.ipynb'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing notebook: {path}')
        path.write_text(json.dumps(make_notebook(raw, scope), ensure_ascii=False, indent=1), encoding='utf-8')
        print(path)
    print('Source bundle SHA256:', hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
