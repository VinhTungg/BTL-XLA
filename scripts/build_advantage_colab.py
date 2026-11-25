"""Package the unchanged matched pilot for Colab, with persistent Drive output."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import zipfile

from build_advantage_notebooks import ROOT, bundle_source, make_notebook

EXPECTED_BUNDLE = '1d85d0bbb17201c08e38b7428f720fccfe2859cc94cd0b256fac1103c7c9dce5'
EXPECTED_ENCODER = 'ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64'
CONTROL_NOTEBOOK = ROOT/'artifacts/effective_batch_pilot/Advantage_microbatch_seed44.ipynb'


def read_control_source(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        match = re.search(r"^raw = base64\.b64decode\('([A-Za-z0-9+/=]+)'\)$", ''.join(cell['source']), re.MULTILINE)
        if match:
            return base64.b64decode(match[1], validate=True)
    raise ValueError(f'No embedded source bundle in {path}')


def make_colab(raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED_BUNDLE:
        raise ValueError('Source changed from the completed microbatch control')
    nb = make_notebook(raw, 'effective_batch')
    setup = ''.join(nb['cells'][1]['source'])
    mount = '''from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/TACS_Advantage_Pilot')
INPUT_ZIP = DRIVE_ROOT/'TACS_Advantage_Colab_Inputs.zip'
PERSISTENT_OUTPUT = DRIVE_ROOT/'advantage_effective_batch_seed44'
if not INPUT_ZIP.is_file():
    from google.colab import files
    print('Choose TACS_Advantage_Colab_Inputs.zip from your computer.', flush=True)
    uploaded = files.upload()
    if set(uploaded) != {'TACS_Advantage_Colab_Inputs.zip'}:
        raise ValueError('Expected only TACS_Advantage_Colab_Inputs.zip; rerun with the correct file')
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_ZIP.write_bytes(uploaded['TACS_Advantage_Colab_Inputs.zip'])
    del uploaded
print('Persistent experiment directory:', PERSISTENT_OUTPUT, flush=True)
'''
    packages = '''import subprocess, sys, importlib.metadata, shutil
if not shutil.which('nvidia-smi'):
    raise RuntimeError('No NVIDIA runtime attached. Select T4 and reconnect to a GPU runtime before installing packages.')
gpu_probe = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True)
if gpu_probe.stdout.strip() != 'Tesla T4':
    raise RuntimeError(f'Expected one Tesla T4, got {gpu_probe.stdout.strip()!r}; no packages installed.')
print('GPU PREFLIGHT:', gpu_probe.stdout.strip(), flush=True)
def install_live(arguments):
    print('Installing:', ' '.join(arguments), flush=True)
    with subprocess.Popen([sys.executable, '-m', 'pip', 'install', *arguments],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait() != 0:
            raise RuntimeError(f'pip failed with exit code {process.returncode}; stop and inspect the log above')
required = {'torch':'2.11.0+cu130', 'torchvision':'0.26.0+cu130', 'timm':'1.0.30'}
def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
needs_torch = any(installed(k) != required[k] for k in ('torch','torchvision'))
if needs_torch:
    if 'torch' in sys.modules or 'torchvision' in sys.modules:
        raise RuntimeError('Restart session before installing the pinned torch environment')
    install_live(['--no-cache-dir',
        'torch==2.11.0', 'torchvision==0.26.0',
        '--index-url', 'https://download.pytorch.org/whl/cu130'])
install_live(['timm==1.0.30', 'psutil>=5.9', 'setuptools>=68', 'wheel'])
import torch, torchvision, timm
actual = {'torch':str(torch.__version__), 'torchvision':str(torchvision.__version__), 'timm':str(timm.__version__)}
if actual != required or not torch.cuda.is_available():
    raise RuntimeError(f'Cannot match Kaggle CUDA environment: {actual}; enable GPU and check driver compatibility')
gpu = torch.cuda.get_device_name(0)
if gpu != 'Tesla T4':
    raise RuntimeError(f'Matched control used Tesla T4, got {gpu}. Stop: choose T4 or plan a new paired control.')
print('MATCHED ENVIRONMENT:', actual, gpu, flush=True)
'''
    # Keep canonical Linux paths to preserve the exact config comparison and resume identity.
    config = '''SCOPE = 'effective_batch'
OUTPUT_ROOT = Path('/kaggle/working/advantage_effective_batch_seed44')
PERSISTENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
PERSISTENT_OUTPUT.mkdir(exist_ok=True)
OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
if OUTPUT_ROOT.is_symlink():
    if OUTPUT_ROOT.resolve() != PERSISTENT_OUTPUT.resolve():
        raise RuntimeError('Output link points at a different experiment; refusing to overwrite')
elif OUTPUT_ROOT.exists():
    raise RuntimeError('Canonical output already exists and is not the expected Drive link')
else:
    OUTPUT_ROOT.symlink_to(PERSISTENT_OUTPUT, target_is_directory=True)
from policy_weight_pilot import sha256, ENCODER_SHA256
with zipfile.ZipFile(INPUT_ZIP) as archive:
    contents = archive.read('reference/no_context/best_encoder.pt')
    if hashlib.sha256(contents).hexdigest() != ENCODER_SHA256:
        raise ValueError('Wrong reference encoder: no microbatch task/selector warm-start allowed')
    control = json.loads(archive.read('control/experiment_manifest.json'))
    expected_env = dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                        timm=str(timm.__version__), gpu=torch.cuda.get_device_name(0))
    if control['environment'] != expected_env or control['source_bundle_sha256'] != SOURCE_SHA256:
        raise ValueError('Input control identity/environment mismatch')
    ENCODER = PROJECT_DIR/'reference/no_context/best_encoder.pt'
    ENCODER.parent.mkdir(parents=True, exist_ok=True)
    ENCODER.write_bytes(contents)
print('Scope:', SCOPE, 'Drive output:', PERSISTENT_OUTPUT, flush=True)
print('Only SAME-scope completed epoch checkpoints can be resumed automatically.')
'''
    # run() resolves the Drive symlink; output_dir is already excluded by the paired
    # comparator. All other recipe paths remain identical to the Kaggle control.
    train = '''from advantage_pilot import run
print('Training effective_batch; fresh initialization unless its own matching latest exists.', flush=True)
print('Output is persisted on Drive; do not mount a microbatch output here.', flush=True)
result = run(PROJECT_DIR, OUTPUT_ROOT, SCOPE, ENCODER, SOURCE_SHA256)
'''
    intro = '''# TACS effective-batch — Colab, seed44
20 epochs, validation only; unchanged source bundle and training recipe.
Upload `TACS_Advantage_Colab_Inputs.zip` when cell1 asks (or place it in
`MyDrive/TACS_Advantage_Pilot/` before running).
Choose T4 GPU, then run cells top to bottom and approve Drive mount yourself.
Output/checkpoints are written directly to Drive. Same-scope resume only.
Colab may stop a session; no uninterrupted-runtime guarantee.
Only output_dir differs after resolving the Drive link; the existing comparator
already excludes output_dir. All source/environment/recipe checks stay enabled.
'''
    codes = [mount, packages, setup, config, ''.join(nb['cells'][3]['source']), train,
             ''.join(nb['cells'][5]['source'])]
    cells = [dict(cell_type='markdown', metadata={}, source=intro.splitlines(True))]
    for index, code in enumerate(codes):
        ast.parse(code)
        cells.append(dict(cell_type='code', id=f'colab-{index}', metadata={},
                          execution_count=None, outputs=[], source=code.splitlines(True)))
    nb['cells'] = cells
    nb['metadata']['accelerator'] = 'GPU'
    nb['metadata']['colab'] = {'name':'Advantage_effective_batch_seed44_Colab.ipynb', 'provenance':[]}
    return nb


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Verified Kaggle results (3).zip')
    parser.add_argument('--source-notebook', type=Path, help='Original notebook of the matched microbatch control')
    args = parser.parse_args()
    source = args.source_notebook
    if source is None and CONTROL_NOTEBOOK.is_file():
        source = CONTROL_NOTEBOOK
    raw = read_control_source(source) if source is not None else bundle_source(ROOT)
    notebook = make_colab(raw)
    output = ROOT/'artifacts/effective_batch_pilot'
    notebook_path = output/'Advantage_effective_batch_seed44_Colab.ipynb'
    input_path = output/'TACS_Advantage_Colab_Inputs.zip'
    if notebook_path.exists() or input_path.exists():
        raise FileExistsError('Refusing to overwrite existing Colab deliverables')
    prefix = 'advantage_source_' + EXPECTED_BUNDLE[:12]
    with zipfile.ZipFile(args.archive) as source:
        encoder = source.read(prefix+'/reference/no_context/best_encoder.pt')
        manifest = source.read('advantage_microbatch_seed44/experiment_manifest.json')
    if hashlib.sha256(encoder).hexdigest() != EXPECTED_ENCODER:
        raise ValueError('Reference encoder hash mismatch')
    if json.loads(manifest)['source_bundle_sha256'] != EXPECTED_BUNDLE:
        raise ValueError('Wrong control source')
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('reference/no_context/best_encoder.pt', encoder)
        archive.writestr('control/experiment_manifest.json', manifest)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(notebook_path)
    print(input_path)


if __name__ == '__main__':
    main()
