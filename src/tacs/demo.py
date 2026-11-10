"""Local Tk desktop demo; no web service, authentication or remote upload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import threading

from .inference import DemoEngine, default_artifacts, load_verified


class DemoApp:
    def __init__(self, root, settings, initialize=True):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.settings = root, settings
        self.events = queue.Queue()
        self.engine = self.image = self.report = None
        self.busy = True
        root.title('TACS — CIFAR-100 context selection (real checkpoint)')
        root.geometry('1160x790')
        self.status = tk.StringVar(value='Đang nạp checkpoint; lần đầu tạo cache có thể mất vài phút…')
        self.truth = tk.StringVar(value='Chưa chọn ảnh')
        ttk.Label(root, text='TACS / Two-Stage — Demo dữ liệu thật', font=('Arial', 17, 'bold')).pack(pady=8)
        ttk.Label(root, text='CIFAR-100 gốc 32×32 được phóng to, không tạo thêm chi tiết. Không train hoặc dùng tập test.').pack()
        controls = ttk.Frame(root)
        controls.pack(pady=10)
        ttk.Label(controls, text='Vị trí validation (0–4999):').grid(row=0, column=0)
        self.position = tk.StringVar(value='0')
        ttk.Spinbox(controls, from_=0, to=4999, textvariable=self.position, width=8).grid(row=0, column=1, padx=5)
        self.load_button = ttk.Button(controls, text='Chọn ảnh validation', command=self.load_validation, state='disabled')
        self.load_button.grid(row=0, column=2, padx=5)
        self.file_button = ttk.Button(controls, text='Mở ảnh riêng…', command=self.open_image, state='disabled')
        self.file_button.grid(row=0, column=3, padx=5)
        self.run_button = ttk.Button(controls, text='So sánh các cách chọn context', command=self.predict, state='disabled')
        self.run_button.grid(row=0, column=4, padx=5)
        pictures = ttk.Frame(root)
        pictures.pack()
        left, right = ttk.Frame(pictures), ttk.Frame(pictures)
        left.grid(row=0, column=0, padx=30)
        right.grid(row=0, column=1, padx=30)
        ttk.Label(left, text='Query').pack()
        self.query_view = ttk.Label(left)
        self.query_view.pack()
        ttk.Label(right, text='Context (chọn một hàng kết quả để xem)').pack()
        self.context_view = ttk.Label(right)
        self.context_view.pack()
        ttk.Label(root, textvariable=self.truth).pack(pady=5)
        self.table = ttk.Treeview(root, columns=('method', 'prediction', 'confidence', 'context', 'ms'), show='headings', height=4)
        for key, title, width in [('method', 'Can thiệp', 175), ('prediction', 'Dự đoán', 160),
                                  ('confidence', 'Top-1 probability', 140), ('context', 'Context / chỉ số CIFAR', 260),
                                  ('ms', 'Task forward (ms)', 150)]:
            self.table.heading(key, text=title)
            self.table.column(key, width=width)
        self.table.pack(fill='x', padx=15)
        self.table.bind('<<TreeviewSelect>>', self.show_context)
        self.details = tk.Text(root, height=8, wrap='word')
        self.details.pack(fill='both', expand=True, padx=15, pady=8)
        ttk.Label(root, text='Các hàng dùng cùng task weights; không phải các baseline train độc lập. Probability chưa được hiệu chỉnh.').pack()
        self.save_button = ttk.Button(root, text='Lưu kết quả JSON…', command=self.save, state='disabled')
        self.save_button.pack(pady=5)
        ttk.Label(root, textvariable=self.status, wraplength=1100).pack(pady=5)
        if initialize:
            threading.Thread(target=self.initialize, daemon=True).start()
        root.after(100, self.poll)

    def initialize(self):
        try:
            engine = DemoEngine(**self.settings, progress=lambda message: self.events.put(('status', message)))
            self.events.put(('ready', engine))
        except Exception as error:
            self.events.put(('error', str(error)))

    def show_image(self, widget, image):
        from PIL import Image, ImageTk
        scale = min(230 / image.width, 230 / image.height)
        preview = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.NEAREST)
        widget.image = ImageTk.PhotoImage(preview)
        widget.configure(image=widget.image, text='')

    def clear_results(self):
        self.report = None
        for item in self.table.get_children():
            self.table.delete(item)
        self.context_view.configure(image='', text='Chưa chạy')
        self.details.delete('1.0', 'end')
        self.save_button.configure(state='disabled')

    def load_validation(self):
        from tkinter import messagebox
        if self.busy or self.engine is None:
            return
        try:
            position = int(self.position.get())
            self.image, target = self.engine.validation_image(position)
            self.truth.set(f'Validation position={position} | Nhãn thật: {target}')
            self.show_image(self.query_view, self.image)
            self.clear_results()
        except (ValueError, IndexError) as error:
            messagebox.showerror('Ảnh không hợp lệ', str(error))

    def open_image(self):
        from tkinter import filedialog, messagebox
        from PIL import Image, ImageOps
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[('Image', '*.png *.jpg *.jpeg *.webp *.bmp'), ('All', '*.*')])
        if not path:
            return
        try:
            with Image.open(path) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError('Ảnh quá lớn; giới hạn 16 megapixel')
                self.image = ImageOps.exif_transpose(source).convert('RGB')
            self.show_image(self.query_view, self.image)
            self.truth.set('Ảnh riêng: chưa biết nhãn thật; mô hình chỉ nhận diện trong 100 lớp CIFAR-100')
            self.clear_results()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            messagebox.showerror('Không mở được ảnh', str(error))

    def predict(self):
        if self.busy or self.image is None:
            return
        self.busy = True
        self.clear_results()
        for button in (self.run_button, self.load_button, self.file_button):
            button.configure(state='disabled')
        self.status.set('Đang inference, không cập nhật trọng số…')
        image = self.image.copy()
        def worker():
            try:
                self.events.put(('result', self.engine.predict(image)))
            except Exception as error:
                self.events.put(('error', str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        from tkinter import messagebox
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'status':
                    self.status.set(value)
                elif kind == 'ready':
                    self.engine, self.busy = value, False
                    self.enable()
                    self.load_validation()
                    self.status.set(f"Sẵn sàng | weight={value.cfg.policy_weight} | seed={value.cfg.seed} | checkpoint epoch={value.metadata['checkpoint_epoch']}")
                elif kind == 'result':
                    self.report, self.busy = value, False
                    names = dict(selected='Selector đã học', random='Random trong top-32', frozen_cosine='Frozen cosine top-1', no_context='Bỏ context')
                    for name, item in value['results'].items():
                        context = '—' if item['context_index'] is None else f"{item['context_label']} / {item['context_index']}"
                        self.table.insert('', 'end', iid=name, values=(names[name], item['prediction'],
                            f"{item['top5'][0]['probability']:.2%}", context, f"{item['task_ms']:.2f}"))
                    self.table.selection_set('selected')
                    self.show_context()
                    self.enable()
                    self.save_button.configure(state='normal')
                    self.status.set(f"Xong | retrieval={value['retrieval_ms']:.1f} ms | selector={value['selector_ms']:.1f} ms | tổng so sánh={value['comparison_ms']:.1f} ms; không phải benchmark chuẩn")
                elif kind == 'error':
                    self.busy = False
                    self.status.set('Lỗi: ' + value)
                    if self.engine:
                        self.enable()
                    messagebox.showerror('Demo không chạy được', value)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def enable(self):
        for button in (self.run_button, self.load_button, self.file_button):
            button.configure(state='normal')

    def show_context(self, event=None):
        if not self.report or not self.table.selection():
            return
        item = self.report['results'][self.table.selection()[0]]
        if item['context_index'] is None:
            self.context_view.configure(image='', text='Không dùng context')
        else:
            self.show_image(self.context_view, self.engine.raw_image(item['context_index']))
        self.details.delete('1.0', 'end')
        self.details.insert('end', '\n'.join(f"{x['label']}: {x['probability']:.2%}" for x in item['top5']))
        self.details.insert('end', '\n\nCheckpoint SHA256: ' + self.report['metadata']['checkpoint_sha256'])

    def save(self):
        from tkinter import filedialog
        if not self.report:
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                Path(path).write_text(json.dumps(self.report, indent=2, ensure_ascii=False), encoding='utf-8')
            except OSError as error:
                from tkinter import messagebox
                messagebox.showerror('Không lưu được', str(error))


def main():
    defaults = default_artifacts()
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in defaults.items():
        parser.add_argument('--' + key.replace('_', '-'), type=Path, default=value)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--check-ui', action='store_true', help='Build/validate Tk widgets, no model or inference')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--validation-position', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = {k: getattr(args, k) for k in (*defaults, 'device')}
    try:
        if args.check_only:
            loaded = load_verified(args.checkpoint, args.standard_encoder, args.split_file, args.device)
            print(json.dumps(dict(status='preflight_passed', **loaded.metadata), indent=2))
        elif args.headless:
            if args.output and args.output.exists():
                raise FileExistsError('Refusing to overwrite output')
            engine = DemoEngine(**settings)
            image, target = engine.validation_image(args.validation_position)
            result = engine.predict(image)
            result.update(validation_position=args.validation_position, target=target)
            text = json.dumps(result, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x', encoding='utf-8') as stream:
                    stream.write(text)
            print(text)
        else:
            import tkinter as tk
            root = tk.Tk()
            if args.check_ui:
                root.withdraw()
            DemoApp(root, settings, initialize=not args.check_ui)
            if args.check_ui:
                root.update_idletasks()
                root.destroy()
                print('TK_WIDGETS_CHECK_PASSED (not a visual or inference check)')
            else:
                root.mainloop()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'DEMO FAILED: {error}\nNo checkpoint was changed.\n')


if __name__ == '__main__':
    main()
