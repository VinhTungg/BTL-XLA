"""First Colab cell: verify GPU and persist Caltech demo outputs; no training."""
from pathlib import Path
import subprocess

import torch
from google.colab import drive

if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU attached. Select T4 and reconnect; do not train on CPU.')
probe = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True, check=True,
)
print('GPU:', probe.stdout.strip(), flush=True)
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, flush=True)
drive.mount('/content/drive')
OUTPUT_ROOT = Path('/content/drive/MyDrive/TACS_Caltech101_Demo/seed44_demo_v1')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print('Persistent output:', OUTPUT_ROOT, flush=True)
print('Existing entries:', sorted(p.name for p in OUTPUT_ROOT.iterdir()), flush=True)
print('PREFLIGHT OK — no training started, no previous outputs deleted.', flush=True)
