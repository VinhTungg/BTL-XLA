"""Build a self-contained, source-hashed Colab notebook for Caltech demo."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    data = io.BytesIO()
    paths = sorted(p for p in (ROOT/'src/tacs').rglob('*.py') if '__pycache__' not in p.parts)
    paths.append(ROOT/'pyproject.toml')
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),(2026,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    raw = data.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    cells = []
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))
    cells.append(dict(cell_type='markdown',metadata={},source=[
        '# TACS Caltech-101 — demo ảnh rõ, seed44\n',
        'Giữ CIFAR-100 riêng. Đây là protocol demo mới, không phải tái hiện đầy đủ paper.\n',
        'T4 GPU; chạy từng cell từ trên xuống. Cell TRAIN là thao tác bắt đầu huấn luyện.\n',
        'Không mở test để chọn model. Checkpoint lưu Drive sau mỗi epoch. Không bảo đảm Colab chạy liên tục.\n']))
    code('''# 1. GPU + THƯ VIỆN — chưa train.
import os, subprocess, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Chọn GPU T4 trước khi chạy; không tự train CPU.')
print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, flush=True)
subprocess.run([sys.executable,'-m','pip','install','timm==1.0.30','psutil>=5.9','setuptools>=68','wheel'],check=True)
''')
    code('''# 2. DRIVE — tự chọn đúng tài khoản và cấp quyền khi được hỏi.
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_ROOT=Path('/content/drive/MyDrive/TACS_Caltech101_Demo/seed44_demo_v1')
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
print('Persistent output:',OUTPUT_ROOT,flush=True)
print('Existing entries:',sorted(p.name for p in OUTPUT_ROOT.iterdir()),flush=True)
''')
    code(f'''# 3. SOURCE — nhúng sẵn, không cần upload ZIP hay clone repo.
import base64, hashlib, io, zipfile
SOURCE_SHA256={sha!r}
raw=base64.b64decode({base64.b64encode(raw).decode()!r})
if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:
    raise ValueError('Source bundle hash mismatch')
PROJECT_DIR=Path('/content/tacs_caltech101')
PROJECT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    if not all((PROJECT_DIR/n).resolve().is_relative_to(PROJECT_DIR.resolve()) for n in archive.namelist()):
        raise ValueError('Unsafe source path')
    archive.extractall(PROJECT_DIR)
subprocess.run([sys.executable,'-m','pip','install','--no-deps','-e',str(PROJECT_DIR)],check=True)
os.chdir(PROJECT_DIR)
sys.path.insert(0,str(PROJECT_DIR/'src'))
DATA_ROOT=PROJECT_DIR/'data'
print('Source verified:',SOURCE_SHA256,flush=True)
''')
    code('''# DOWNLOAD + PREFLIGHT, chưa train.
from tacs.caltech.data import prepare
inventory,split=prepare(DATA_ROOT,OUTPUT_ROOT/'split.json',download=True)
from tacs.caltech.data import atomic_json
atomic_json(OUTPUT_ROOT/'inventory.json', inventory)
print('Classes:',len(inventory['classes']), 'Images:',len(inventory['items']),flush=True)
print({k:len(split[k]) for k in ('query_train','candidate_pool','validation','test')},flush=True)
print('Native image sizes:',[(x['width'],x['height']) for x in inventory['items'][:3]],flush=True)
assert len(inventory['classes'])==101
''')
    code('''# KHÓA CONFIG trước lần train đầu. Không đổi EPOCHS khi resume.
EPOCHS=20
BASELINE='both'  # no_context rồi two_stage; không chạy CIFAR.
STOP_AFTER=None  # ví dụ 10 = dừng sau epoch10 với scheduler vẫn thiết kế20epoch.
print('Plan:',BASELINE, 'total epochs/run:',EPOCHS,'stop at:',STOP_AFTER,flush=True)
print('19 query + 6 pool + 5 validation / class; test giữ riêng. Demo protocol, không phải paper score.')
''')
    code('''# TRAIN — CHỈ BẤM CELL NÀY KHI MUỐN BẮT ĐẦU.
# GPU sẽ chạy trên Colab; có thể mở cùng notebook/tài khoản trên máy khác trước khi bấm.
import shlex
cmd=[sys.executable,'-u','-m','tacs.caltech.run','--data-root',str(DATA_ROOT),
     '--output-root',str(OUTPUT_ROOT),'--baseline',BASELINE,'--epochs',str(EPOCHS)]
if STOP_AFTER is not None:
    cmd+=['--stop-after',str(STOP_AFTER)]
print('Executing:',shlex.join(cmd),flush=True)
env=dict(os.environ,PYTHONUNBUFFERED='1')
with (OUTPUT_ROOT/'train.log').open('a',encoding='utf-8',buffering=1) as log:
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env) as process:
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line)
        status=process.wait()
if status:
    raise RuntimeError(f'Train stopped with code {status}; giữ output, kiểm tra log; không tự xóa/reset.')
''')
    code('''# VERIFY — không train/test, từ chối đánh dấu hoàn tất nếu còn thiếu run.
import json
from tacs.caltech.data import digest
for name in ('no_context','two_stage'):
    folder=OUTPUT_ROOT/name
    report=json.loads((folder/'result.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text())
    assert report['status']=='complete' and not report['test_evaluated']
    assert len(metrics['history'])==EPOCHS and metrics['history'][-1]['epoch']==EPOCHS
    assert metrics['test_result'] is None
    assert report['checkpoint_sha256']==digest(folder/'checkpoint_best.pt')
    assert report['metrics_sha256']==digest(folder/'metrics.json')
    if name=='no_context':
        assert report['reference_encoder_sha256']==digest(folder/'best_encoder.pt')
    print(name,'COMPLETE — best validation:',report['best_val_accuracy'],flush=True)
print('Tải nguyên thư mục output trên Drive, gồm split, manifests, configs/checkpoints/metrics và train.log.')
print('Đây là checkpoint để tích hợp demo Caltech; không đổi tên thành checkpoint CIFAR.')
''')
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        accelerator='GPU',colab=dict(name='TACS_Caltech101_Demo_seed44.ipynb')))


def main():
    path=ROOT/'artifacts/caltech_demo/TACS_Caltech101_Demo_seed44.ipynb'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    nb=build()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
