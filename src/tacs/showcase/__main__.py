"""Offline CIFAR-100/Caltech-101 demo: python -m tacs.showcase."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import queue
import threading

from ..inference import DemoEngine


ROOT = Path(__file__).resolve().parents[3]
PRESETS = {
    'microbatch': ('kaggle_microbatch_seed44/advantage_microbatch_seed44',
                  'fbc50a26867d321a32892f262a9d447f45e3be95dd0a3a345053cf04ee8425d8'),
    'effective_batch': ('colab_effective_batch_seed44/advantage_effective_batch_seed44',
                        '5e7406aa36bfd16e20628f27c61f3238db3926b1afe12cc552ddb2618b845fa4'),
}
NAMES = dict(selected='Selector đã học', frozen_cosine='Ảnh gần nhất (retrieval)',
             random='Random trong top-32', no_context='Bỏ context (cùng weights)',
             trained_no_context='No-context train riêng')
DATASETS = dict(cifar100=dict(title='CIFAR-100',classes=100,validation_size=5000,presets=list(PRESETS)),
                caltech101=dict(title='Caltech-101',classes=101,validation_size=505,presets=['full100']))
# Candidate examples from the full audit; confirmed again in batch-1 smoke.
EXAMPLES = [('Cùng đúng', 0), ('Ví dụ A', 120), ('Ví dụ B', 111)]
CALTECH_EXAMPLES = [('Cùng đúng',0),('Context giúp',178),('Context gây sai',169)]


def settings_for(preset, root=ROOT):
    if preset not in PRESETS:
        raise ValueError(f'Preset không hợp lệ: {preset}')
    root = Path(root)
    experiment = root / 'artifacts/effective_batch_pilot' / PRESETS[preset][0]
    manifest = json.loads((experiment / 'experiment_manifest.json').read_text(encoding='utf-8'))
    result = json.loads((experiment / 'result.json').read_text(encoding='utf-8'))
    cfg = manifest['config']
    if (manifest.get('kind') != 'advantage_normalization_pilot'
            or cfg.get('advantage_scope') != preset or cfg.get('seed') != 44
            or cfg.get('epochs') != 20 or cfg.get('eval_test') is not False
            or result.get('status') != 'complete' or result.get('advantage_scope') != preset
            or result.get('epochs') != 20 or result.get('training_seed') != 44
            or result.get('test_evaluated') is not False
            or manifest.get('test_evaluated') is not False
            or result.get('checkpoint_sha256') != PRESETS[preset][1]):
        raise ValueError('Preset/manifest không khớp pilot đã nghiệm thu; không tự thay checkpoint')
    return dict(checkpoint=experiment/'run/checkpoint_best.pt',
                standard_encoder=root/'artifacts/policy_weight_pilot/kaggle_w0p03_seed44/policy_source_e427116799fe/reference/no_context/best_encoder.pt',
                split_file=root/'data/splits/cifar100_seed42.json', data_root=root/'data',
                cache_dir=root/'artifacts/demo_cache')


def create_engine(preset, device='auto', progress=print, dataset='cifar100'):
    if dataset=='caltech101':
        if preset!='full100': raise ValueError('Caltech-101 yêu cầu preset full100')
        from .caltech import CaltechDemoEngine
        return CaltechDemoEngine(ROOT/'artifacts/caltech_demo/seed44_full100_v1',ROOT/'data',
                                 ROOT/'artifacts/demo_cache/caltech101',device,progress)
    if dataset!='cifar100': raise ValueError('Dataset không được hỗ trợ')
    engine = DemoEngine(**settings_for(preset), device=device, progress=progress)
    engine.dataset_name='cifar100'
    engine.metadata.update(dataset='cifar100',num_classes=100,preset=preset,advantage_scope=engine.cfg.advantage_scope)
    return engine


def validation_input(engine, position):
    image, label = engine.validation_image(position)
    if getattr(engine,'dataset_name','cifar100')=='caltech101':
        index=engine.split.validation[position]
        return image,dict(kind='validation',dataset='caltech101',validation_position=position,
            dataset_index=index,relative_file=engine.dataset.inventory['items'][index]['file'],
            target=label,original_size=list(image.size))
    return image, dict(kind='validation', validation_position=position,
                       cifar_index=engine.split.validation[position], target=label,
                       original_size=list(image.size))


def external_input(path,dataset='cifar100'):
    if dataset not in DATASETS: raise ValueError('Dataset không được hỗ trợ')
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        if source.width * source.height > 16_000_000:
            raise ValueError('Ảnh quá lớn: tối đa 16 megapixel')
        image = ImageOps.exif_transpose(source).convert('RGB')
    return image, dict(kind='external', dataset=dataset, filename=Path(path).name, target=None,
                       original_size=list(image.size),
                       warning=f"Ảnh ngoài phân phối; mô hình buộc chọn một trong {DATASETS[dataset]['classes']} lớp {DATASETS[dataset]['title']}, không phát hiện unknown.")


def make_report(engine, image, query):
    if query.get('dataset',getattr(engine,'dataset_name','cifar100'))!=getattr(engine,'dataset_name','cifar100'):
        raise ValueError('Ảnh và model khác dataset; chọn ảnh lại sau khi đổi dataset')
    report = engine.predict(image)
    report['metadata'] = dict(report['metadata'])
    report['query'] = dict(query)
    report['query']['rgb_sha256'] = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    report['random_seed'] = 44
    for item in report['results'].values():
        item['correct'] = None if query['target'] is None else item['prediction'] == query['target']
    return report


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


class ShowcaseApp:
    def __init__(self, root, preset='microbatch', device='auto', initialize=True, dataset='cifar100'):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.device = root, device
        self.engine = self.image = self.query = self.report = None
        self.events = queue.Queue()
        self.busy = False
        root.title('TACS — Demo CIFAR-100 / Caltech-101 | dữ liệu thật')
        root.geometry('1180x940')
        root.minsize(1050, 850)
        root.configure(bg='#f0f3f7')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f3f7')
        style.configure('TLabel', background='#f0f3f7', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), padding=6)
        style.configure('Treeview', font=('Segoe UI', 10), rowheight=30)
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))
        self.status = tk.StringVar(value='Chưa nạp mô hình')
        self.model_info = tk.StringVar(value='Hai preset là hai checkpoint riêng; không có kết quả train giả.')
        self.truth = tk.StringVar(value='Chọn ảnh validation hoặc mở ảnh riêng')
        self.preset = tk.StringVar(value=preset)
        self.dataset_choice = tk.StringVar(value=dataset)
        self.subtitle = tk.StringVar(value='Chọn dataset và checkpoint; không train trong demo.')
        self.preview_note = tk.StringVar()
        self.footer_note = tk.StringVar()
        self.position = tk.StringVar(value='0')
        ttk.Label(root, text='TACS | Ảnh nào thực sự giúp dự đoán?',
                  font=('Segoe UI', 19, 'bold')).pack(pady=(12, 3))
        ttk.Label(root, textvariable=self.subtitle).pack()
        modelbar = ttk.Frame(root)
        modelbar.pack(pady=8)
        ttk.Label(modelbar, text='Dataset:').pack(side='left',padx=5)
        self.dataset_box=ttk.Combobox(modelbar,values=list(DATASETS),textvariable=self.dataset_choice,
                                    state='readonly',width=14)
        self.dataset_box.pack(side='left',padx=5)
        self.dataset_box.bind('<<ComboboxSelected>>',self.dataset_changed)
        ttk.Label(modelbar, text='Checkpoint:').pack(side='left', padx=5)
        self.preset_box = ttk.Combobox(modelbar, values=DATASETS[dataset]['presets'], textvariable=self.preset,
                                      state='readonly', width=22)
        self.preset_box.pack(side='left', padx=5)
        self.preset_box.bind('<<ComboboxSelected>>',lambda event:self.load_model())
        self.reload_button = ttk.Button(modelbar, text='Nạp checkpoint', command=self.load_model)
        self.reload_button.pack(side='left', padx=5)
        ttk.Label(root, textvariable=self.model_info).pack()
        bar = ttk.Frame(root)
        bar.pack(pady=8)
        ttk.Label(bar, text='Validation position:').pack(side='left')
        self.position_box = ttk.Spinbox(bar, from_=0, to=DATASETS[dataset]['validation_size']-1, textvariable=self.position, width=7)
        self.position_box.pack(side='left', padx=5)
        self.validation_button = ttk.Button(bar, text='Chọn ảnh', command=self.load_validation)
        self.validation_button.pack(side='left', padx=5)
        self.file_button = ttk.Button(bar, text='Mở ảnh riêng…', command=self.open_image)
        self.file_button.pack(side='left', padx=5)
        self.run_button = ttk.Button(bar, text='So sánh dự đoán', command=self.predict)
        self.run_button.pack(side='left', padx=5)
        examples = ttk.Frame(root)
        examples.pack()
        ttk.Label(examples, text='Ví dụ minh họa (không thay số liệu toàn tập):').pack(side='left')
        self.example_buttons = []
        for name, position in EXAMPLES:
            button = ttk.Button(examples, text=f'{name} · {position}', command=lambda p=position: self.choose_example(p))
            button.pack(side='left', padx=4)
            self.example_buttons.append(button)
        pictures = ttk.Frame(root)
        pictures.pack(pady=7)
        for column, title, attr in [(0, 'QUERY — ảnh cần phân loại', 'query_view'),
                                     (1, 'CONTEXT — chọn hàng kết quả để xem', 'context_view')]:
            frame = ttk.Frame(pictures)
            frame.grid(row=0, column=column, padx=32)
            ttk.Label(frame, text=title, font=('Segoe UI', 10, 'bold')).pack()
            label = ttk.Label(frame, text='Chưa chọn', anchor='center')
            label.pack()
            setattr(self, attr, label)
        ttk.Label(root, textvariable=self.truth).pack(pady=4)
        ttk.Label(root, textvariable=self.preview_note).pack()
        columns = ('strategy', 'prediction', 'correct', 'probability', 'context', 'time')
        self.table = ttk.Treeview(root, columns=columns, show='headings', height=5)
        for key, label, width in zip(columns, ['Mô hình / context', 'Dự đoán', 'Đúng / sai', 'Top-1 probability', 'Context / Dataset ID', 'Task (ms)'],
                                     [205, 135, 85, 140, 240, 95]):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=70)
        self.table.tag_configure('correct', foreground='#126333')
        self.table.tag_configure('wrong', foreground='#a32424')
        self.table.pack(fill='x', padx=16, pady=5)
        self.table.bind('<<TreeviewSelect>>', self.show_context)
        self.details = tk.Text(root, height=5, wrap='word', font=('Consolas', 10), bg='#ffffff', relief='flat')
        self.details.pack(fill='both', expand=True, padx=16, pady=5)
        ttk.Label(root, textvariable=self.footer_note).pack()
        self.save_button = ttk.Button(root, text='Lưu bằng chứng JSON…', command=self.save)
        self.save_button.pack(pady=4)
        ttk.Label(root, textvariable=self.status, wraplength=1120).pack(pady=(2, 8))
        self.set_controls()
        self.update_dataset_labels()
        root.after(100, self.poll)
        if initialize:
            self.load_model()

    def set_controls(self):
        ready = self.engine is not None and not self.busy
        self.reload_button.configure(state='disabled' if self.busy else 'normal')
        self.preset_box.configure(state='disabled' if self.busy else 'readonly')
        self.dataset_box.configure(state='disabled' if self.busy else 'readonly')
        for button in [self.validation_button, self.file_button, *self.example_buttons]:
            button.configure(state='normal' if ready else 'disabled')
        self.position_box.configure(state='normal' if ready else 'disabled')
        self.run_button.configure(state='normal' if ready and self.image is not None else 'disabled')
        self.save_button.configure(state='normal' if self.report is not None and not self.busy else 'disabled')

    def update_dataset_labels(self):
        name=self.dataset_choice.get()
        meta=DATASETS[name]
        self.subtitle.set(f"{meta['title']} • DINOv3 ViT-S/16 • role_v2 • seed44 • Offline / không train")
        self.position_box.configure(to=meta['validation_size']-1)
        self.preview_note.set('Caltech: preview ảnh nguồn, không thu về 32×32; model nhận 256×256. Ảnh riêng có thể ngoài phân phối.'
            if name=='caltech101' else 'CIFAR gốc 32×32 được phóng to, không tạo thêm chi tiết. Ảnh riêng có thể ngoài phân phối.')
        self.footer_note.set('4 hàng đầu: cùng Two-Stage weights. Hàng No-context train riêng: checkpoint độc lập 100 epoch. Probability chưa hiệu chỉnh.'
            if name=='caltech101' else 'Các hàng dùng CÙNG task weights, không phải 4 baseline train độc lập. Probability chưa hiệu chỉnh.')
        examples=CALTECH_EXAMPLES if name=='caltech101' else EXAMPLES
        for button,(label,position) in zip(self.example_buttons,examples):
            button.configure(text=f'{label} · {position}',command=lambda p=position:self.choose_example(p))

    def dataset_changed(self,event=None):
        if self.busy: return
        self.preset_box.configure(values=DATASETS[self.dataset_choice.get()]['presets'])
        self.preset.set(DATASETS[self.dataset_choice.get()]['presets'][0])
        self.position.set('0')
        self.update_dataset_labels()
        self.load_model()

    def clear_results(self):
        self.report = None
        for iid in self.table.get_children():
            self.table.delete(iid)
        self.context_view.configure(image='', text='Chưa chạy')
        self.details.delete('1.0', 'end')
        self.set_controls()

    def load_model(self):
        if self.busy:
            return
        self.busy = True
        self.image = self.query = None
        self.query_view.configure(image='', text='Chưa chọn')
        self.truth.set('Đang đổi checkpoint; cần chọn ảnh lại')
        self.clear_results()
        preset = self.preset.get()
        dataset = self.dataset_choice.get()
        # Drop old model before allocating a replacement on a 4 GB GPU.
        self.engine = None
        self.model_info.set('Đang xác minh và nạp ' + preset)
        self.status.set('Đang xác minh artifact và dựng/nạp kho đặc trưng của đúng dataset; không train.')
        def worker():
            try:
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.events.put(('ready', create_engine(preset, self.device,
                    lambda text: self.events.put(('status', text)),dataset=dataset)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def show_image(self, widget, image):
        from PIL import Image, ImageTk
        scale = min(250/image.width, 220/image.height)
        preview = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                               Image.Resampling.NEAREST if image.size == (32, 32) else Image.Resampling.LANCZOS)
        widget.image = ImageTk.PhotoImage(preview)
        widget.configure(image=widget.image, text='')

    def load_validation(self):
        if self.busy or self.engine is None:
            return
        try:
            image, query = validation_input(self.engine, int(self.position.get()))
            self.image, self.query = image, query
            self.show_image(self.query_view, image)
            index=query.get('dataset_index',query.get('cifar_index'))
            self.truth.set(f"Validation #{query['validation_position']} | Dataset ID={index} | Nhãn thật: {query['target']} | Gốc {image.width}×{image.height}")
            self.clear_results()
        except (ValueError, IndexError) as exc:
            self.error_dialog(str(exc))

    def choose_example(self, position):
        self.position.set(str(position))
        self.load_validation()

    def open_image(self):
        from tkinter import filedialog
        if self.busy or self.engine is None:
            return
        path = filedialog.askopenfilename(filetypes=[('Ảnh', '*.png *.jpg *.jpeg *.webp *.bmp')])
        if not path:
            return
        try:
            from PIL import Image
            image, query = external_input(path,self.dataset_choice.get())
            self.image, self.query = image, query
            self.show_image(self.query_view, image)
            meta=DATASETS[self.dataset_choice.get()]
            self.truth.set(f"Ảnh riêng — không biết nhãn thật; chỉ dự đoán trong {meta['classes']} lớp {meta['title']}")
            self.clear_results()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            self.error_dialog(str(exc))

    def predict(self):
        if self.busy or self.engine is None or self.image is None:
            return
        self.busy = True
        self.clear_results()
        self.status.set('Đang so sánh, không cập nhật trọng số…')
        engine, image, query = self.engine, self.image.copy(), dict(self.query)
        def worker():
            try:
                self.events.put(('result', make_report(engine, image, query)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'status':
                    self.status.set(value)
                elif kind == 'ready':
                    self.engine, self.busy = value, False
                    self.model_info.set(f"{value.metadata['preset']} | best epoch {value.metadata['checkpoint_epoch']} | "
                        f"Validation lúc train {value.metadata['checkpoint_val_accuracy']:.2%} | {value.metadata['device']}")
                    if value.metadata.get('dataset')=='caltech101':
                        self.model_info.set(f"Two-Stage: best epoch {value.metadata['checkpoint_epoch']}, val {value.metadata['checkpoint_val_accuracy']:.2%} | "
                            f"No-context riêng: epoch {value.metadata['no_context_checkpoint_epoch']}, val {value.metadata['no_context_val_accuracy']:.2%} | {value.metadata['device']}")
                    self.status.set('Sẵn sàng. Chọn ảnh rồi bấm So sánh dự đoán.')
                    self.load_validation()
                elif kind == 'result':
                    self.report, self.busy = value, False
                    for name,item in value['results'].items():
                        correct = item['correct']
                        context = '—' if item['context_index'] is None else f"{item['context_label']} / {item['context_index']}"
                        self.table.insert('', 'end', iid=name, tags=('correct' if correct else 'wrong',) if correct is not None else (),
                            values=(NAMES[name], item['prediction'], '—' if correct is None else ('Đúng' if correct else 'Sai'),
                                    f"{item['top5'][0]['probability']:.2%}", context, f"{item['task_ms']:.1f}"))
                    self.table.selection_set('selected')
                    self.show_context()
                    self.status.set(f"Xong | retrieval {value['retrieval_ms']:.1f} ms | selector {value['selector_ms']:.1f} ms | "
                                    f"so sánh {value['comparison_ms']:.1f} ms (không phải benchmark)")
                elif kind == 'error':
                    self.busy = False
                    self.status.set('Lỗi: ' + value)
                    self.error_dialog(value)
                self.set_controls()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def show_context(self, event=None):
        if not self.report or not self.table.selection():
            return
        name = self.table.selection()[0]
        item = self.report['results'][name]
        if item['context_index'] is None:
            self.context_view.configure(image='', text='Không dùng context')
        else:
            self.show_image(self.context_view, self.engine.raw_image(item['context_index']))
        self.details.delete('1.0', 'end')
        self.details.insert('end', NAMES[name] + ' | Top-5: ' + ' · '.join(
            f"{x['label']} {x['probability']:.2%}" for x in item['top5']))
        self.details.insert('end', '\nCheckpoint SHA256: ' + item.get('checkpoint_sha256',self.report['metadata']['checkpoint_sha256']))
        self.details.insert('end', '\nBằng chứng: validation-only; random seed 44; không cập nhật trọng số.')

    def save(self):
        from tkinter import filedialog
        if self.report is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                write_report(path, self.report)
                self.status.set('Đã lưu JSON: ' + path)
            except (OSError, ValueError) as exc:
                self.error_dialog(str(exc) + '\nChọn tên file mới; không ghi đè bằng chứng cũ.')

    @staticmethod
    def error_dialog(text):
        from tkinter import messagebox
        messagebox.showerror('Demo không chạy được', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=list(DATASETS),default='cifar100')
    parser.add_argument('--preset', choices=[*PRESETS,'full100'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--validation-position', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-ui', action='store_true')
    args = parser.parse_args()
    args.preset=args.preset or DATASETS[args.dataset]['presets'][0]
    try:
        if args.preset not in DATASETS[args.dataset]['presets']:
            raise ValueError('Preset không thuộc dataset đã chọn')
        if args.headless:
            if args.output and args.output.exists():
                raise FileExistsError('Output đã tồn tại; chọn tên mới')
            size=DATASETS[args.dataset]['validation_size']
            if not 0 <= args.validation_position < size:
                raise ValueError(f'Validation position phải trong 0..{size-1}')
            engine = create_engine(args.preset, args.device,dataset=args.dataset)
            image, query = validation_input(engine, args.validation_position)
            result = make_report(engine, image, query)
            if args.output:
                write_report(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            import tkinter as tk
            root = tk.Tk()
            if args.check_ui:
                root.withdraw()
            ShowcaseApp(root, args.preset, args.device, initialize=not args.check_ui,dataset=args.dataset)
            if args.check_ui:
                root.update_idletasks()
                root.destroy()
                print('SHOWCASE_WIDGETS_OK (not a visual or inference check)')
            else:
                root.mainloop()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'DEMO FAILED: {exc}\nKhông thay đổi checkpoint.\n')


if __name__ == '__main__':
    main()
