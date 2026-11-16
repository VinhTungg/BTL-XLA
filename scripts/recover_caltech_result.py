"""Safely recover a known Caltech ZIP; keep the ZIP and all existing files."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from tacs.caltech.data import digest, scan
from tacs.showcase.caltech_artifacts import SOURCE_SHA256, verify_artifacts


def embedded_source_hash(notebook):
    nb=json.loads(Path(notebook).read_text(encoding='utf-8'))
    source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'
                and 'SOURCE_SHA256=' in ''.join(c['source']))
    assignments={n.targets[0].id:n.value for n in ast.parse(source).body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    raw=base64.b64decode(ast.literal_eval(assignments['raw'].args[0]))
    if hashlib.sha256(raw).hexdigest()!=ast.literal_eval(assignments['SOURCE_SHA256']):
        raise ValueError('Notebook source ZIP hash mismatch')
    h=hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in sorted(n for n in z.namelist() if n.startswith('src/tacs/') and n.endswith('.py')):
            h.update(name.removeprefix('src/tacs/').encode())
            h.update(z.read(name))
    return h.hexdigest()


def extract_safely(archive, parent):
    parent=Path(parent).resolve()
    prefix='seed44_full100_v1'
    expected={f'{prefix}/{x}' for x in ('inventory.json','split.json','train.log')}
    for name in ('no_context','two_stage'):
        expected.update(f'{prefix}/{name}/{x}' for x in ('experiment_manifest.json','result.json',
            'metrics.json','checkpoint_best.pt','checkpoint_latest.pt'))
    expected.add(f'{prefix}/no_context/best_encoder.pt')
    with zipfile.ZipFile(archive) as z:
        entries=[i for i in z.infolist() if not i.is_dir()]
        names=[i.filename for i in entries]
        if len(names)!=len(set(names)) or set(names)!=expected:
            raise ValueError('Unexpected/missing/duplicate ZIP entries; preserve archive for review')
        for item in entries:
            rel=PurePosixPath(item.filename)
            dest=(parent/Path(*rel.parts)).resolve()
            if '\\' in item.filename or rel.is_absolute() or '..' in rel.parts or not dest.is_relative_to(parent):
                raise ValueError('Unsafe ZIP target')
        needed=sum(i.file_size for i in entries if not (parent/i.filename).exists())
        if shutil.disk_usage(parent).free < needed+512*1024**2:
            raise ValueError('Insufficient disk space; need extracted bytes plus 512 MiB reserve')
        for item in entries:
            dest=parent/item.filename
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as source:
                if dest.exists():
                    if hashlib.file_digest(source,'sha256').hexdigest()!=digest(dest):
                        raise FileExistsError(f'Existing file differs; never overwrite: {dest}')
                else:
                    with dest.open('xb') as target:
                        shutil.copyfileobj(source,target,1024*1024)
            print('RECOVERED',item.filename,flush=True)
    return parent/prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=Path('artifacts/caltech_demo'))
    parser.add_argument('--notebook',type=Path,default=Path('artifacts/caltech_demo/TACS_Caltech101_Full100_seed44.ipynb'))
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--report',type=Path,default=Path('artifacts/caltech_demo/audit_full100.json'))
    args=parser.parse_args()
    try:
        if args.report.exists(): raise FileExistsError('Audit report exists; choose a new report path')
        if embedded_source_hash(args.notebook)!=SOURCE_SHA256:
            raise ValueError('Notebook does not contain the trained source')
        root=extract_safely(args.archive,args.parent)
        report=verify_artifacts(root,verify_latest=True)
        local=scan(args.data_root)
        if local['inventory_sha256']!=report['inventory_sha256']:
            raise ValueError('Local image bytes/class map do not match cloud inventory')
        report.update(archive=args.archive.name,archive_sha256=digest(args.archive),
                      local_real_images_verified=True,training_source_notebook_verified=True,
                      warning='Validation-only demo protocol; no test score or improvement claim.')
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,RuntimeError,zipfile.BadZipFile) as exc:
        parser.exit(1,f'CALTECH RECOVERY FAILED: {exc}\nKeep ZIP and outputs; nothing was deleted.\n')


if __name__=='__main__':
    main()
