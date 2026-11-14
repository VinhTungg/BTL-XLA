"""Offline demo of verified advantage pilots: python -m tacs.showcase."""
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
             random='Random trong top-32', no_context='Không context')
# Candidate examples from the full audit; confirmed again in batch-1 smoke.
EXAMPLES = [('Cùng đúng', 0), ('Ví dụ A', 120), ('Ví dụ B', 111)]


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


def create_engine(preset, device='auto', progress=print):
    engine = DemoEngine(**settings_for(preset), device=device, progress=progress)
    engine.metadata.update(preset=preset, advantage_scope=engine.cfg.advantage_scope)
    return engine


def validation_input(engine, position):
    image, label = engine.validation_image(position)
    return image, dict(kind='validation', validation_position=position,
                       cifar_index=engine.split.validation[position], target=label,
                       original_size=list(image.size))


def external_input(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        if source.width * source.height > 16_000_000:
            raise ValueError('Ảnh quá lớn: tối đa 16 megapixel')
        image = ImageOps.exif_transpose(source).convert('RGB')
    return image, dict(kind='external', filename=Path(path).name, target=None,
                       original_size=list(image.size),
                       warning='Ảnh ngoài phân phối; mô hình buộc chọn một trong 100 lớp, không phát hiện unknown.')


def make_report(engine, image, query):
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
    def __init__(self, root, preset='microbatch', device='auto', initialize=True):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.device = root, device
        self.engine = self.image = self.query = self.report = None
        self.events = queue.Queue()
        self.busy = False
        root.title('TACS — Demo CIFAR-100 | dữ liệu thật')
        root.geometry('1180x860')
        root.minsize(1050, 760)
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
        self.position = tk.StringVar(value='0')
        ttk.Label(root, text='TACS | Ảnh nào thực sự giúp dự đoán?',
                  font=('Segoe UI', 19, 'bold')).pack(pady=(12, 3))
        ttk.Label(root, text='CIFAR-100 • DINOv3 ViT-S/16 • Two-Stage role_v2 • seed 44 • Offline / không train').pack()
        modelbar = ttk.Frame(root)
        modelbar.pack(pady=8)
        ttk.Label(modelbar, text='Checkpoint:').pack(side='left', padx=5)
        self.preset_box = ttk.Combobox(modelbar, values=list(PRESETS), textvariable=self.preset,
                                      state='readonly', width=22)
        self.preset_box.pack(side='left', padx=5)
        self.reload_button = ttk.Button(modelbar, text='Nạp checkpoint', command=self.load_model)
        self.reload_button.pack(side='left', padx=5)
        ttk.Label(root, textvariable=self.model_info).pack()
        bar = ttk.Frame(root)
        bar.pack(pady=8)
        ttk.Label(bar, text='Validation position:').pack(side='left')
        self.position_box = ttk.Spinbox(bar, from_=0, to=4999, textvariable=self.position, width=7)
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
        ttk.Label(root, text='Ảnh CIFAR gốc 32×32 được phóng to, không tạo thêm chi tiết. Ảnh riêng có thể ngoài phân phối.').pack()
        columns = ('strategy', 'prediction', 'correct', 'probability', 'context', 'time')
        self.table = ttk.Treeview(root, columns=columns, show='headings', height=4)
        for key, label, width in zip(columns, ['Cách chọn context', 'Dự đoán', 'Đúng / sai', 'Top-1 probability', 'Context / CIFAR ID', 'Task (ms)'],
                                     [205, 135, 85, 140, 240, 95]):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=70)
        self.table.tag_configure('correct', foreground='#126333')
        self.table.tag_configure('wrong', foreground='#a32424')
        self.table.pack(fill='x', padx=16, pady=5)
        self.table.bind('<<TreeviewSelect>>', self.show_context)
        self.details = tk.Text(root, height=5, wrap='word', font=('Consolas', 10), bg='#ffffff', relief='flat')
        self.details.pack(fill='both', expand=True, padx=16, pady=5)
        ttk.Label(root, text='Các hàng dùng CÙNG task weights, không phải 4 baseline train độc lập. Probability chưa hiệu chỉnh.').pack()
        self.save_button = ttk.Button(root, text='Lưu bằng chứng JSON…', command=self.save)
        self.save_button.pack(pady=4)
        ttk.Label(root, textvariable=self.status, wraplength=1120).pack(pady=(2, 8))
        self.set_controls()
        root.after(100, self.poll)
        if initialize:
            self.load_model()

    def set_controls(self):
        ready = self.engine is not None and not self.busy
        self.reload_button.configure(state='disabled' if self.busy else 'normal')
        self.preset_box.configure(state='disabled' if self.busy else 'readonly')
        for button in [self.validation_button, self.file_button, *self.example_buttons]:
            button.configure(state='normal' if ready else 'disabled')
        self.position_box.configure(state='normal' if ready else 'disabled')
        self.run_button.configure(state='normal' if ready and self.image is not None else 'disabled')
        self.save_button.configure(state='normal' if self.report is not None and not self.busy else 'disabled')

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
        # Drop old model before allocating a replacement on a 4 GB GPU.
        self.engine = None
        self.model_info.set('Đang xác minh và nạp ' + preset)
        self.status.set('Lần đầu dựng kho đặc trưng 9.000 ảnh có thể mất vài phút; không train.')
        def worker():
            try:
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.events.put(('ready', create_engine(preset, self.device,
                    lambda text: self.events.put(('status', text)))))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def show_image(self, widget, image):
        from PIL import Image, ImageTk
        scale = min(190/image.width, 190/image.height)
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
            self.truth.set(f"Validation #{query['validation_position']} | CIFAR ID={query['cifar_index']} | Nhãn thật: {query['target']}")
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
            image, query = external_input(path)
            self.image, self.query = image, query
            self.show_image(self.query_view, image)
            self.truth.set('Ảnh riêng — không biết nhãn thật; chỉ dự đoán trong 100 lớp CIFAR-100')
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
                    self.status.set('Sẵn sàng. Chọn ảnh rồi bấm So sánh dự đoán.')
                    self.load_validation()
                elif kind == 'result':
                    self.report, self.busy = value, False
                    for name in NAMES:
                        item = value['results'][name]
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
        self.details.insert('end', '\nCheckpoint SHA256: ' + self.report['metadata']['checkpoint_sha256'])
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
    parser.add_argument('--preset', choices=list(PRESETS), default='microbatch')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--validation-position', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-ui', action='store_true')
    args = parser.parse_args()
    try:
        if args.headless:
            if args.output and args.output.exists():
                raise FileExistsError('Output đã tồn tại; chọn tên mới')
            if not 0 <= args.validation_position < 5000:
                raise ValueError('Validation position phải trong 0..4999')
            engine = create_engine(args.preset, args.device)
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
            ShowcaseApp(root, args.preset, args.device, initialize=not args.check_ui)
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
