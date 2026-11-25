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
from .explain import (
    HOW_IT_WORKS_TEXT,
    STRATEGY_DESCRIPTIONS,
    candidate_badges,
    candidate_rank_text,
    pipeline_card_texts,
    step_narration,
    strategy_group,
    summarize_outcome,
    top_stage_candidates,
)


ROOT = Path(__file__).resolve().parents[3]
PRESETS = {
    'microbatch': ('kaggle_microbatch_seed44/advantage_microbatch_seed44',
                   'fbc50a26867d321a32892f262a9d447f45e3be95dd0a3a345053cf04ee8425d8'),
    'effective_batch': ('colab_effective_batch_seed44/advantage_effective_batch_seed44',
                        '5e7406aa36bfd16e20628f27c61f3238db3926b1afe12cc552ddb2618b845fa4'),
}
NAMES = dict(
    selected='Selector đã học',
    frozen_cosine='Ảnh gần nhất (retrieval)',
    random='Random trong top-32',
    no_context='Bỏ context (cùng weights)',
    trained_no_context='No-context train riêng',
)
DATASETS = dict(
    cifar100=dict(title='CIFAR-100', classes=100, validation_size=5000, presets=list(PRESETS)),
    caltech101=dict(title='Caltech-101', classes=101, validation_size=505, presets=['full100']),
)
# Candidate examples from the full audit; confirmed again in batch-1 smoke.
EXAMPLES = [('Cùng đúng', 0), ('Ví dụ A', 120), ('Ví dụ B', 111)]
CALTECH_EXAMPLES = [('Cùng đúng', 0), ('Context giúp', 178), ('Context gây sai', 169)]

TONE_PALETTE = {
    'neutral': ('#eef2f7', '#cbd5e1', '#1e293b'),
    'help': ('#ecfdf5', '#10b981', '#065f46'),
    'hurt': ('#fef2f2', '#ef4444', '#991b1b'),
    'ok': ('#eff6ff', '#3b82f6', '#1e40af'),
    'warn': ('#fffbeb', '#f59e0b', '#92400e'),
}


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
    return dict(
        checkpoint=experiment / 'run/checkpoint_best.pt',
        standard_encoder=root / 'artifacts/policy_weight_pilot/kaggle_w0p03_seed44/policy_source_e427116799fe/reference/no_context/best_encoder.pt',
        split_file=root / 'data/splits/cifar100_seed42.json',
        data_root=root / 'data',
        cache_dir=root / 'artifacts/demo_cache',
    )


def create_engine(preset, device='auto', progress=print, dataset='cifar100'):
    if dataset == 'caltech101':
        if preset != 'full100':
            raise ValueError('Caltech-101 yêu cầu preset full100')
        from .caltech import CaltechDemoEngine
        return CaltechDemoEngine(
            ROOT / 'artifacts/caltech_demo/seed44_full100_v1',
            ROOT / 'data',
            ROOT / 'artifacts/demo_cache/caltech101',
            device,
            progress,
        )
    if dataset != 'cifar100':
        raise ValueError('Dataset không được hỗ trợ')
    engine = DemoEngine(**settings_for(preset), device=device, progress=progress)
    engine.dataset_name = 'cifar100'
    engine.metadata.update(
        dataset='cifar100',
        num_classes=100,
        preset=preset,
        advantage_scope=engine.cfg.advantage_scope,
    )
    return engine


def validation_input(engine, position):
    image, label = engine.validation_image(position)
    if getattr(engine, 'dataset_name', 'cifar100') == 'caltech101':
        index = engine.split.validation[position]
        return image, dict(
            kind='validation',
            dataset='caltech101',
            validation_position=position,
            dataset_index=index,
            relative_file=engine.dataset.inventory['items'][index]['file'],
            target=label,
            original_size=list(image.size),
        )
    return image, dict(
        kind='validation',
        validation_position=position,
        cifar_index=engine.split.validation[position],
        target=label,
        original_size=list(image.size),
    )


def external_input(path, dataset='cifar100'):
    if dataset not in DATASETS:
        raise ValueError('Dataset không được hỗ trợ')
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        if source.width * source.height > 16_000_000:
            raise ValueError('Ảnh quá lớn: tối đa 16 megapixel')
        image = ImageOps.exif_transpose(source).convert('RGB')
    return image, dict(
        kind='external',
        dataset=dataset,
        filename=Path(path).name,
        target=None,
        original_size=list(image.size),
        warning=(
            f"Ảnh ngoài phân phối; mô hình buộc chọn một trong {DATASETS[dataset]['classes']} "
            f"lớp {DATASETS[dataset]['title']}, không phát hiện unknown."
        ),
    )


def make_report(engine, image, query):
    if query.get('dataset', getattr(engine, 'dataset_name', 'cifar100')) != getattr(engine, 'dataset_name', 'cifar100'):
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

        self.tk = tk
        self.ttk = ttk
        self.root, self.device = root, device
        self.engine = self.image = self.query = self.report = None
        self.events = queue.Queue()
        self.busy = False
        self.active_step = 4
        self.pending_walkthrough = False
        self.stage_photo_refs = []
        self.stage_cards = []

        root.title('TACS Two-Stage - Demo trực quan CIFAR-100 / Caltech-101 (checkpoint thật)')
        root.geometry('1280x860')
        root.minsize(1080, 740)
        root.configure(bg='#f1f5f9')

        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#f1f5f9')
        style.configure('Card.TFrame', background='#ffffff')
        style.configure('TLabel', background='#f1f5f9', foreground='#0f172a', font=('Segoe UI', 10))
        style.configure('Card.TLabel', background='#ffffff', foreground='#0f172a', font=('Segoe UI', 10))
        style.configure('Muted.TLabel', background='#f1f5f9', foreground='#475569', font=('Segoe UI', 9))
        style.configure('CardMuted.TLabel', background='#ffffff', foreground='#475569', font=('Segoe UI', 9))
        style.configure('TButton', font=('Segoe UI', 9), padding=(8, 4))
        style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'), padding=(10, 5),
                        background='#1d4ed8', foreground='#ffffff')
        style.map('Accent.TButton',
                  background=[('active', '#1e40af'), ('disabled', '#94a3b8')],
                  foreground=[('disabled', '#f8fafc')])
        style.configure('TNotebook', background='#f1f5f9', borderwidth=0)
        style.configure('TNotebook.Tab', font=('Segoe UI', 9, 'bold'), padding=(12, 6))
        style.configure('Treeview', font=('Segoe UI', 9), rowheight=28, background='#ffffff', fieldbackground='#ffffff')
        style.configure('Treeview.Heading', font=('Segoe UI', 9, 'bold'), background='#e2e8f0', foreground='#0f172a')

        self.status = tk.StringVar(value='Chưa nạp mô hình')
        self.model_info = tk.StringVar(value='Hai preset là hai checkpoint riêng; không có kết quả train giả.')
        self.truth = tk.StringVar(value='Chọn ảnh validation hoặc mở ảnh riêng')
        self.context_caption = tk.StringVar(value='Chưa chạy suy luận')
        self.context_connector = tk.StringVar(value='+ ghép patch\n(role_v2)')
        self.top5_title = tk.StringVar(value='TOP-5 XÁC SUẤT DỰ ĐOÁN')
        self.row_explain = tk.StringVar(value='Bấm từng hàng trong bảng bên dưới để xem cách chọn context và Top-5 tương ứng.')
        self.candidate_detail = tk.StringVar(value='Bấm vào bất kỳ ảnh ứng viên nào ở Giai đoạn 1 hoặc 2 để xem điểm số chi tiết.')
        self.stage1_note = tk.StringVar(value='Chưa chạy suy luận.')
        self.narration = tk.StringVar(value=step_narration(4, None, None, DATASETS[dataset]['title']))
        self.verdict_text = tk.StringVar(value=summarize_outcome(None)[1])
        self.preset = tk.StringVar(value=preset)
        self.dataset_choice = tk.StringVar(value=dataset)
        self.subtitle = tk.StringVar(value='Chọn dataset và checkpoint; không train trong demo.')
        self.preview_note = tk.StringVar()
        self.footer_note = tk.StringVar()
        self.position = tk.StringVar(value='0')

        self._build_header(root)
        self._build_pipeline_strip(root)
        self._build_tabs(root)
        self._build_footer(root)

        self.set_controls()
        self.update_dataset_labels()
        self._refresh_pipeline_strip()
        self._draw_top5_bars(None, None)
        root.after(100, self.poll)
        if initialize:
            self.load_model()

    def _build_header(self, parent):
        tk, ttk = self.tk, self.ttk
        header = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        header.pack(fill='x', padx=12, pady=(8, 5))

        row1 = tk.Frame(header, bg='#ffffff')
        row1.pack(fill='x', padx=12, pady=(8, 4))
        left = tk.Frame(row1, bg='#ffffff')
        left.pack(side='left', fill='x', expand=True)
        tk.Label(
            left,
            text='TACS Two-Stage | Ảnh nào thực sự giúp dự đoán?',
            font=('Segoe UI', 14, 'bold'),
            bg='#ffffff',
            fg='#0f172a',
            anchor='w',
        ).pack(anchor='w')
        tk.Label(
            left,
            textvariable=self.subtitle,
            font=('Segoe UI', 9),
            bg='#ffffff',
            fg='#475569',
            anchor='w',
        ).pack(anchor='w')

        right = tk.Frame(row1, bg='#ffffff')
        right.pack(side='right')
        tk.Label(right, text='Dataset:', bg='#ffffff', fg='#334155', font=('Segoe UI', 9, 'bold')).pack(side='left', padx=(0, 4))
        self.dataset_box = ttk.Combobox(
            right, values=list(DATASETS), textvariable=self.dataset_choice, state='readonly', width=12,
        )
        self.dataset_box.pack(side='left', padx=(0, 8))
        self.dataset_box.bind('<<ComboboxSelected>>', self.dataset_changed)

        tk.Label(right, text='Checkpoint:', bg='#ffffff', fg='#334155', font=('Segoe UI', 9, 'bold')).pack(side='left', padx=(0, 4))
        self.preset_box = ttk.Combobox(
            right, values=DATASETS[self.dataset_choice.get()]['presets'],
            textvariable=self.preset, state='readonly', width=16,
        )
        self.preset_box.pack(side='left', padx=(0, 6))
        self.preset_box.bind('<<ComboboxSelected>>', lambda event: self.load_model())

        self.reload_button = ttk.Button(right, text='Nạp lại', command=self.load_model)
        self.reload_button.pack(side='left', padx=(0, 6))
        self.help_button = ttk.Button(right, text='? Cách chạy', command=self.show_help_dialog)
        self.help_button.pack(side='left')

        row2 = tk.Frame(header, bg='#f8fafc', highlightbackground='#e2e8f0', highlightthickness=1)
        row2.pack(fill='x', padx=12, pady=(2, 6))

        controls_left = tk.Frame(row2, bg='#f8fafc')
        controls_left.pack(side='left', padx=6, pady=5)
        tk.Label(controls_left, text='Validation #:', bg='#f8fafc', fg='#334155', font=('Segoe UI', 9, 'bold')).pack(side='left')
        self.position_box = ttk.Spinbox(
            controls_left,
            from_=0,
            to=DATASETS[self.dataset_choice.get()]['validation_size'] - 1,
            textvariable=self.position,
            width=6,
        )
        self.position_box.pack(side='left', padx=(4, 6))
        self.validation_button = ttk.Button(controls_left, text='Chọn ảnh', command=self.load_validation)
        self.validation_button.pack(side='left', padx=(0, 4))
        self.file_button = ttk.Button(controls_left, text='Mở ảnh riêng...', command=self.open_image)
        self.file_button.pack(side='left', padx=(0, 8))

        tk.Label(controls_left, text='Ví dụ nhanh:', bg='#f8fafc', fg='#475569', font=('Segoe UI', 9)).pack(side='left', padx=(2, 4))
        self.example_buttons = []
        for name, position in EXAMPLES:
            button = ttk.Button(
                controls_left,
                text=f'{name} · {position}',
                command=lambda p=position: self.choose_example(p),
            )
            button.pack(side='left', padx=2)
            self.example_buttons.append(button)

        controls_right = tk.Frame(row2, bg='#f8fafc')
        controls_right.pack(side='right', padx=6, pady=5)
        self.run_button = ttk.Button(controls_right, text='So sánh dự đoán', style='Accent.TButton', command=self.predict)
        self.run_button.pack(side='left', padx=(0, 4))
        self.walk_button = ttk.Button(controls_right, text='Xem từng bước', command=self.start_walkthrough)
        self.walk_button.pack(side='left', padx=(0, 4))
        self.save_button = ttk.Button(controls_right, text='Lưu JSON...', command=self.save)
        self.save_button.pack(side='left')

        tk.Label(
            header,
            textvariable=self.model_info,
            font=('Segoe UI', 8),
            bg='#ffffff',
            fg='#475569',
            anchor='w',
        ).pack(fill='x', padx=12, pady=(0, 6))

    def _build_pipeline_strip(self, parent):
        tk, ttk = self.tk, self.ttk
        wrapper = tk.Frame(parent, bg='#f1f5f9')
        wrapper.pack(fill='x', padx=12, pady=(0, 5))

        strip = tk.Frame(wrapper, bg='#f1f5f9')
        strip.pack(fill='x')
        self.stage_cards = []
        for idx in range(4):
            step_num = idx + 1
            card = tk.Frame(
                strip,
                bg='#ffffff',
                highlightbackground='#cbd5e1',
                highlightthickness=1,
                cursor='hand2',
                padx=8,
                pady=5,
            )
            card.pack(side='left', fill='both', expand=True)
            title_lbl = tk.Label(card, text='', font=('Segoe UI', 9, 'bold'), bg='#ffffff', fg='#1e293b', anchor='w')
            title_lbl.pack(fill='x')
            metric_lbl = tk.Label(card, text='', font=('Segoe UI', 10, 'bold'), bg='#ffffff', fg='#0f172a', anchor='w')
            metric_lbl.pack(fill='x')
            detail_lbl = tk.Label(card, text='', font=('Segoe UI', 8), bg='#ffffff', fg='#475569', anchor='w')
            detail_lbl.pack(fill='x')
            for widget in (card, title_lbl, metric_lbl, detail_lbl):
                widget.bind('<Button-1>', lambda e, s=step_num: self.select_step(s))
            self.stage_cards.append(dict(frame=card, title=title_lbl, metric=metric_lbl, detail=detail_lbl))
            if idx < 3:
                tk.Label(strip, text='->', font=('Segoe UI', 11, 'bold'), bg='#f1f5f9', fg='#64748b').pack(side='left', padx=4)

        narr_bar = tk.Frame(wrapper, bg='#eff6ff', highlightbackground='#bfdbfe', highlightthickness=1)
        narr_bar.pack(fill='x', pady=(4, 0))
        tk.Label(
            narr_bar,
            textvariable=self.narration,
            font=('Segoe UI', 9),
            bg='#eff6ff',
            fg='#1e3a8a',
            justify='left',
            anchor='w',
            wraplength=1040,
            padx=10,
            pady=5,
        ).pack(side='left', fill='x', expand=True)
        nav = tk.Frame(narr_bar, bg='#eff6ff')
        nav.pack(side='right', padx=6, pady=3)
        self.prev_step_btn = ttk.Button(nav, text='< Bước trước', command=lambda: self.step_delta(-1))
        self.prev_step_btn.pack(side='left', padx=2)
        self.next_step_btn = ttk.Button(nav, text='Bước tiếp >', command=lambda: self.step_delta(1))
        self.next_step_btn.pack(side='left', padx=2)

    def _build_tabs(self, parent):
        tk, ttk = self.tk, self.ttk
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, padx=12, pady=(0, 4))

        self.tab_overview = tk.Frame(self.notebook, bg='#f1f5f9')
        self.tab_stages = tk.Frame(self.notebook, bg='#f1f5f9')
        self.tab_evidence = tk.Frame(self.notebook, bg='#f1f5f9')

        self.notebook.add(self.tab_overview, text='1. Tổng quan: Query + Context -> Dự đoán & So sánh')
        self.notebook.add(self.tab_stages, text='2. Giai đoạn 1 -> 2: Lọc Top-32 & Selector chấm điểm')
        self.notebook.add(self.tab_evidence, text='3. Bằng chứng & Chi tiết kỹ thuật')

        self._build_overview_tab(self.tab_overview)
        self._build_stages_tab(self.tab_stages)
        self._build_evidence_tab(self.tab_evidence)

    def _build_overview_tab(self, parent):
        tk, ttk = self.tk, self.ttk
        top_card = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        top_card.pack(fill='x', padx=4, pady=(6, 5))

        flow = tk.Frame(top_card, bg='#ffffff')
        flow.pack(fill='x', padx=10, pady=8)

        # Column 1: Query image
        q_box = tk.Frame(flow, bg='#f8fafc', highlightbackground='#cbd5e1', highlightthickness=1, padx=8, pady=6)
        q_box.pack(side='left', fill='y')
        self.query_box = q_box
        tk.Label(q_box, text='QUERY - Ảnh cần phân loại', font=('Segoe UI', 10, 'bold'), bg='#f8fafc', fg='#0f172a').pack()
        q_holder = tk.Frame(q_box, bg='#ffffff', width=215, height=165, highlightbackground='#e2e8f0', highlightthickness=1)
        q_holder.pack(pady=4)
        q_holder.pack_propagate(False)
        self.query_view = tk.Label(q_holder, text='Chưa chọn ảnh', bg='#ffffff', fg='#64748b', font=('Segoe UI', 9))
        self.query_view.pack(expand=True)
        tk.Label(q_box, textvariable=self.truth, font=('Segoe UI', 8), bg='#f8fafc', fg='#334155', wraplength=215, justify='center').pack()

        # Connector Query + Context
        tk.Label(
            flow,
            textvariable=self.context_connector,
            font=('Segoe UI', 9, 'bold'),
            bg='#ffffff',
            fg='#475569',
            justify='center',
            width=12,
        ).pack(side='left', padx=4)

        # Column 2: Selected row's Context image
        c_box = tk.Frame(flow, bg='#f8fafc', highlightbackground='#cbd5e1', highlightthickness=1, padx=8, pady=6)
        c_box.pack(side='left', fill='y')
        self.context_box = c_box
        tk.Label(c_box, text='CONTEXT - Ảnh phụ của hàng đang chọn', font=('Segoe UI', 10, 'bold'), bg='#f8fafc', fg='#0f172a').pack()
        c_holder = tk.Frame(c_box, bg='#ffffff', width=215, height=165, highlightbackground='#e2e8f0', highlightthickness=1)
        c_holder.pack(pady=4)
        c_holder.pack_propagate(False)
        self.context_view = tk.Label(c_holder, text='Chưa chạy', bg='#ffffff', fg='#64748b', font=('Segoe UI', 9), wraplength=190)
        self.context_view.pack(expand=True)
        tk.Label(c_box, textvariable=self.context_caption, font=('Segoe UI', 8), bg='#f8fafc', fg='#334155', wraplength=215, justify='center').pack()

        # Connector -> Task Network
        tk.Label(
            flow,
            text='->\nTask\nNetwork',
            font=('Segoe UI', 9, 'bold'),
            bg='#ffffff',
            fg='#1d4ed8',
            justify='center',
            width=8,
        ).pack(side='left', padx=4)

        # Column 3: Top-5 Horizontal Bar Chart
        p_box = tk.Frame(flow, bg='#f8fafc', highlightbackground='#cbd5e1', highlightthickness=1, padx=8, pady=6)
        p_box.pack(side='left', fill='both', expand=True)
        tk.Label(p_box, textvariable=self.top5_title, font=('Segoe UI', 10, 'bold'), bg='#f8fafc', fg='#0f172a', anchor='w').pack(fill='x')
        self.top5_canvas = tk.Canvas(p_box, height=165, bg='#ffffff', highlightthickness=1, highlightbackground='#e2e8f0')
        self.top5_canvas.pack(fill='both', expand=True, pady=4)
        self.top5_canvas.bind('<Configure>', lambda e: self._redraw_current_top5())
        tk.Label(
            p_box,
            text='Xác suất softmax chưa hiệu chỉnh. Bấm hàng khác trong bảng để quan sát Top-5 dịch chuyển.',
            font=('Segoe UI', 8),
            bg='#f8fafc',
            fg='#64748b',
            anchor='w',
        ).pack(fill='x')

        # Verdict callout banner
        self.verdict_frame = tk.Frame(parent, bg='#eef2f7', highlightbackground='#cbd5e1', highlightthickness=1)
        self.verdict_frame.pack(fill='x', padx=4, pady=(0, 5))
        self.verdict_label = tk.Label(
            self.verdict_frame,
            textvariable=self.verdict_text,
            font=('Segoe UI', 10, 'bold'),
            bg='#eef2f7',
            fg='#1e293b',
            anchor='w',
            justify='left',
            wraplength=1180,
            padx=10,
            pady=5,
        )
        self.verdict_label.pack(fill='x')

        # Comparison table card
        table_card = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        table_card.pack(fill='both', expand=True, padx=4, pady=(0, 4))
        header_row = tk.Frame(table_card, bg='#ffffff')
        header_row.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(
            header_row,
            text='Bảng đối chứng (bấm từng hàng để xem Context và Top-5 phía trên):',
            font=('Segoe UI', 10, 'bold'),
            bg='#ffffff',
            fg='#0f172a',
        ).pack(side='left')
        tk.Label(
            header_row,
            textvariable=self.preview_note,
            font=('Segoe UI', 8),
            bg='#ffffff',
            fg='#64748b',
        ).pack(side='right')

        columns = ('strategy', 'group', 'prediction', 'correct', 'probability', 'context', 'rank', 'time')
        self.table = ttk.Treeview(table_card, columns=columns, show='headings', height=5)
        specs = [
            ('strategy', 'Mô hình / cách lấy context', 190),
            ('group', 'Nhóm trọng số', 175),
            ('prediction', 'Dự đoán', 120),
            ('correct', 'Đúng / sai', 75),
            ('probability', 'Top-1 prob', 80),
            ('context', 'Context (nhãn / ID)', 175),
            ('rank', 'Hạng GĐ1 -> GĐ2', 135),
            ('time', 'Task (ms)', 75),
        ]
        for key, label, width in specs:
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=65)
        self.table.tag_configure('correct', foreground='#15803d')
        self.table.tag_configure('wrong', foreground='#b91c1c')
        self.table.tag_configure('control', background='#f8fafc')
        self.table.pack(fill='both', expand=True, padx=8, pady=(2, 4))
        self.table.bind('<<TreeviewSelect>>', self.show_context)

        tk.Label(
            table_card,
            textvariable=self.row_explain,
            font=('Segoe UI', 8),
            bg='#f8fafc',
            fg='#334155',
            anchor='w',
            justify='left',
            wraplength=1180,
            padx=8,
            pady=4,
        ).pack(fill='x', padx=8, pady=(0, 6))

    def _build_stages_tab(self, parent):
        tk = self.tk
        # Stage 1 box
        self.stage1_box = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        self.stage1_box.pack(fill='x', padx=4, pady=(6, 5))
        s1_head = tk.Frame(self.stage1_box, bg='#ffffff')
        s1_head.pack(fill='x', padx=10, pady=(6, 2))
        tk.Label(
            s1_head,
            text='GIAI ĐOẠN 1 - Frozen DINOv3 Cosine Retrieval (Top 8 / 32 ứng viên gần Query nhất từ pool)',
            font=('Segoe UI', 10, 'bold'),
            bg='#ffffff',
            fg='#0f766e',
        ).pack(side='left')
        tk.Label(
            s1_head,
            textvariable=self.stage1_note,
            font=('Segoe UI', 8, 'bold'),
            bg='#ffffff',
            fg='#1d4ed8',
        ).pack(side='right')
        self.stage1_grid = tk.Frame(self.stage1_box, bg='#ffffff')
        self.stage1_grid.pack(fill='x', padx=8, pady=(2, 8))

        # Stage 2 box
        self.stage2_box = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        self.stage2_box.pack(fill='x', padx=4, pady=(0, 5))
        s2_head = tk.Frame(self.stage2_box, bg='#ffffff')
        s2_head.pack(fill='x', padx=10, pady=(6, 2))
        tk.Label(
            s2_head,
            text='GIAI ĐOẠN 2 - Learned Selector chấm điểm lại 32 ứng viên (Top 8 theo điểm Selector -> chọn #1)',
            font=('Segoe UI', 10, 'bold'),
            bg='#ffffff',
            fg='#1d4ed8',
        ).pack(side='left')
        tk.Label(
            s2_head,
            text='Màu xanh dương = ảnh Selector chọn; so "từ GĐ1 #k" để thấy Selector đảo hạng so với cosine.',
            font=('Segoe UI', 8),
            bg='#ffffff',
            fg='#475569',
        ).pack(side='right')
        self.stage2_grid = tk.Frame(self.stage2_box, bg='#ffffff')
        self.stage2_grid.pack(fill='x', padx=8, pady=(2, 8))

        # Candidate inspector detail strip
        inspector = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        inspector.pack(fill='both', expand=True, padx=4, pady=(0, 4))
        tk.Label(
            inspector,
            text='Chi tiết ứng viên đang chọn:',
            font=('Segoe UI', 9, 'bold'),
            bg='#ffffff',
            fg='#0f172a',
            anchor='w',
            padx=10,
            pady=4,
        ).pack(fill='x')
        tk.Label(
            inspector,
            textvariable=self.candidate_detail,
            font=('Consolas', 10),
            bg='#f8fafc',
            fg='#1e293b',
            anchor='w',
            justify='left',
            wraplength=1180,
            padx=10,
            pady=8,
        ).pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self._render_stage_galleries()

    def _build_evidence_tab(self, parent):
        tk = self.tk
        card = tk.Frame(parent, bg='#ffffff', highlightbackground='#cbd5e1', highlightthickness=1)
        card.pack(fill='both', expand=True, padx=4, pady=6)
        tk.Label(
            card,
            text='Bằng chứng kiểm định, SHA256 checkpoint và thông số suy luận (không cập nhật trọng số)',
            font=('Segoe UI', 10, 'bold'),
            bg='#ffffff',
            fg='#0f172a',
            anchor='w',
            padx=10,
            pady=6,
        ).pack(fill='x')
        self.details = tk.Text(
            card,
            height=18,
            wrap='word',
            font=('Consolas', 10),
            bg='#f8fafc',
            fg='#0f172a',
            relief='flat',
            padx=10,
            pady=8,
        )
        self.details.pack(fill='both', expand=True, padx=10, pady=(0, 10))

    def _build_footer(self, parent):
        tk = self.tk
        foot = tk.Frame(parent, bg='#e2e8f0')
        foot.pack(fill='x', padx=12, pady=(0, 8))
        tk.Label(
            foot,
            textvariable=self.status,
            font=('Segoe UI', 9, 'bold'),
            bg='#e2e8f0',
            fg='#0f172a',
            anchor='w',
            padx=8,
            pady=4,
        ).pack(side='left', fill='x', expand=True)
        tk.Label(
            foot,
            textvariable=self.footer_note,
            font=('Segoe UI', 8),
            bg='#e2e8f0',
            fg='#334155',
            anchor='e',
            padx=8,
            pady=4,
        ).pack(side='right')

    def show_help_dialog(self):
        tk, ttk = self.tk, self.ttk
        win = tk.Toplevel(self.root)
        win.title('Hướng dẫn cách chạy TACS Two-Stage')
        win.geometry('760x540')
        win.configure(bg='#ffffff')
        win.transient(self.root)
        text = tk.Text(win, wrap='word', font=('Segoe UI', 10), bg='#ffffff', fg='#0f172a', relief='flat', padx=14, pady=12)
        text.pack(fill='both', expand=True)
        text.insert('1.0', HOW_IT_WORKS_TEXT)
        text.configure(state='disabled')
        ttk.Button(win, text='Đóng', command=win.destroy).pack(pady=8)

    def set_controls(self):
        ready = self.engine is not None and not self.busy
        self.reload_button.configure(state='disabled' if self.busy else 'normal')
        self.preset_box.configure(state='disabled' if self.busy else 'readonly')
        self.dataset_box.configure(state='disabled' if self.busy else 'readonly')
        for button in [self.validation_button, self.file_button, *self.example_buttons]:
            button.configure(state='normal' if ready else 'disabled')
        self.position_box.configure(state='normal' if ready else 'disabled')
        can_run = ready and self.image is not None
        self.run_button.configure(state='normal' if can_run else 'disabled')
        self.walk_button.configure(state='normal' if can_run else 'disabled')
        self.save_button.configure(state='normal' if self.report is not None and not self.busy else 'disabled')
        step_state = 'normal' if self.report is not None and not self.busy else 'disabled'
        self.prev_step_btn.configure(state=step_state if self.active_step > 1 else 'disabled')
        self.next_step_btn.configure(state=step_state if self.active_step < 4 else 'disabled')

    def update_dataset_labels(self):
        name = self.dataset_choice.get()
        meta = DATASETS[name]
        self.subtitle.set(f"{meta['title']} | DINOv3 ViT-S/16 | role_v2 | seed 44 | Offline - không train")
        self.position_box.configure(to=meta['validation_size'] - 1)
        self.preview_note.set(
            'Caltech: preview ảnh nguồn (model nhận 256x256).'
            if name == 'caltech101'
            else 'CIFAR gốc 32x32 được phóng to, không tạo thêm chi tiết.'
        )
        self.footer_note.set(
            '4 hàng đầu: cùng weights Two-Stage | Hàng 5: checkpoint No-context riêng 100 ep.'
            if name == 'caltech101'
            else '4 hàng dùng CÙNG task weights (không phải 4 baseline train riêng).'
        )
        examples = CALTECH_EXAMPLES if name == 'caltech101' else EXAMPLES
        for button, (label, position) in zip(self.example_buttons, examples):
            button.configure(text=f'{label} · {position}', command=lambda p=position: self.choose_example(p))
        self._refresh_pipeline_strip()

    def dataset_changed(self, event=None):
        if self.busy:
            return
        self.preset_box.configure(values=DATASETS[self.dataset_choice.get()]['presets'])
        self.preset.set(DATASETS[self.dataset_choice.get()]['presets'][0])
        self.position.set('0')
        self.update_dataset_labels()
        self.load_model()

    def clear_results(self):
        self.report = None
        self.active_step = 4
        for iid in self.table.get_children():
            self.table.delete(iid)
        self.context_view.configure(image='', text='Chưa chạy')
        self.context_caption.set('Bấm "So sánh dự đoán" hoặc "Xem từng bước"')
        self.context_connector.set('+ ghép patch\n(role_v2)')
        self.top5_title.set('TOP-5 XÁC SUẤT DỰ ĐOÁN')
        self.row_explain.set('Bấm từng hàng trong bảng bên dưới để xem cách chọn context và Top-5 tương ứng.')
        self.candidate_detail.set('Chưa có dữ liệu ứng viên. Hãy bấm "So sánh dự đoán" trước.')
        self.stage1_note.set('Chưa chạy suy luận.')
        self.details.delete('1.0', 'end')
        self._set_verdict('neutral', summarize_outcome(None)[1])
        self._draw_top5_bars(None, None)
        self._render_stage_galleries()
        self._refresh_pipeline_strip()
        self.set_controls()

    def _set_verdict(self, tone, text):
        bg, border, fg = TONE_PALETTE.get(tone, TONE_PALETTE['neutral'])
        self.verdict_frame.configure(bg=bg, highlightbackground=border)
        self.verdict_label.configure(bg=bg, fg=fg)
        self.verdict_text.set(text)

    def _refresh_pipeline_strip(self):
        title = DATASETS[self.dataset_choice.get()]['title']
        meta = self.engine.metadata if self.engine is not None else {'num_classes': DATASETS[self.dataset_choice.get()]['classes']}
        payloads = pipeline_card_texts(self.report, self.query, meta, title)
        for idx, (card, data) in enumerate(zip(self.stage_cards, payloads), start=1):
            active = (self.report is not None and idx == self.active_step) or (self.report is None and idx == 1 and self.query is not None)
            bg = '#dbeafe' if active else '#ffffff'
            border = '#1d4ed8' if active else '#cbd5e1'
            thick = 2 if active else 1
            card['frame'].configure(bg=bg, highlightbackground=border, highlightthickness=thick)
            card['title'].configure(text=data['title'], bg=bg, fg='#1e3a8a' if active else '#334155')
            card['metric'].configure(text=data['metric'], bg=bg, fg='#0f172a')
            card['detail'].configure(text=data['detail'], bg=bg, fg='#334155' if active else '#64748b')
        self.narration.set(step_narration(self.active_step, self.report, self.query, title))

        # Highlight Stage 1 / Stage 2 boxes when inspecting step 2 or 3
        if hasattr(self, 'stage1_box'):
            self.stage1_box.configure(
                highlightbackground='#0f766e' if self.active_step == 2 else '#cbd5e1',
                highlightthickness=2 if self.active_step == 2 else 1,
            )
            self.stage2_box.configure(
                highlightbackground='#1d4ed8' if self.active_step == 3 else '#cbd5e1',
                highlightthickness=2 if self.active_step == 3 else 1,
            )

    def select_step(self, step):
        self.active_step = max(1, min(4, int(step)))
        if self.report is not None:
            if self.active_step in (2, 3):
                self.notebook.select(self.tab_stages)
            else:
                self.notebook.select(self.tab_overview)
        self._refresh_pipeline_strip()
        self.set_controls()

    def step_delta(self, delta):
        if self.report is None:
            return
        self.select_step(self.active_step + delta)

    def start_walkthrough(self):
        if self.busy or self.engine is None or self.image is None:
            return
        if self.report is not None:
            self.select_step(1)
            return
        self.pending_walkthrough = True
        self.predict()

    def load_model(self):
        if self.busy:
            return
        self.busy = True
        self.image = self.query = None
        self.query_view.configure(image='', text='Chưa chọn ảnh')
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
                self.events.put((
                    'ready',
                    create_engine(
                        preset,
                        self.device,
                        lambda text: self.events.put(('status', text)),
                        dataset=dataset,
                    ),
                ))
            except Exception as exc:
                self.events.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def show_image(self, widget, image, max_w=205, max_h=155):
        from PIL import Image, ImageTk
        scale = min(max_w / image.width, max_h / image.height)
        preview = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.NEAREST if image.size == (32, 32) else Image.Resampling.LANCZOS,
        )
        widget.image = ImageTk.PhotoImage(preview)
        widget.configure(image=widget.image, text='')

    def load_validation(self):
        if self.busy or self.engine is None:
            return
        try:
            image, query = validation_input(self.engine, int(self.position.get()))
            self.image, self.query = image, query
            self.show_image(self.query_view, image)
            index = query.get('dataset_index', query.get('cifar_index'))
            self.truth.set(
                f"Val #{query['validation_position']} | ID={index}\nNhãn thật: {query['target']} ({image.width}x{image.height})"
            )
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
            image, query = external_input(path, self.dataset_choice.get())
            self.image, self.query = image, query
            self.show_image(self.query_view, image)
            meta = DATASETS[self.dataset_choice.get()]
            self.truth.set(
                f"Ảnh riêng: {query['filename']}\nChưa rõ nhãn thật ({meta['classes']} lớp {meta['title']})"
            )
            self.clear_results()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            self.error_dialog(str(exc))

    def predict(self):
        if self.busy or self.engine is None or self.image is None:
            return
        self.busy = True
        self.clear_results()
        self.status.set('Đang chạy GĐ1 (Retrieval) -> GĐ2 (Selector) -> GĐ3 (Task Network), không cập nhật trọng số...')
        engine, image, query = self.engine, self.image.copy(), dict(self.query)

        def worker():
            try:
                self.events.put(('result', make_report(engine, image, query)))
            except Exception as exc:
                self.events.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _draw_top5_bars(self, item, target_label):
        canvas = self.top5_canvas
        canvas.delete('all')
        w = max(canvas.winfo_width(), 380)
        h = max(canvas.winfo_height(), 160)
        if not item or not item.get('top5'):
            canvas.create_text(
                w / 2, h / 2,
                text='Chưa có dự đoán.\nBấm "So sánh dự đoán" để xem biểu đồ Top-5.',
                fill='#64748b',
                font=('Segoe UI', 10),
                justify='center',
            )
            return

        top5 = item['top5'][:5]
        pad_top = 10
        row_h = max(24, int((h - 2 * pad_top) / len(top5)))
        label_w = 135
        pct_w = 62
        bar_x0 = label_w + 8
        bar_max_w = max(80, w - bar_x0 - pct_w - 14)

        for idx, entry in enumerate(top5):
            y0 = pad_top + idx * row_h
            ymid = y0 + row_h / 2
            lbl = str(entry['label'])
            prob = max(0.0, min(1.0, float(entry['probability'])))
            is_truth = (target_label is not None and lbl == target_label)
            is_top1 = (idx == 0)

            short_lbl = lbl if len(lbl) <= 17 else lbl[:15] + '..'
            display_lbl = f'#{idx + 1} {short_lbl}'
            lbl_color = '#15803d' if is_truth else ('#0f172a' if is_top1 else '#475569')
            lbl_font = ('Segoe UI', 9, 'bold') if (is_top1 or is_truth) else ('Segoe UI', 9)
            canvas.create_text(8, ymid, text=display_lbl, anchor='w', fill=lbl_color, font=lbl_font)

            # Background track
            bar_top = y0 + 5
            bar_bot = y0 + row_h - 6
            canvas.create_rectangle(bar_x0, bar_top, bar_x0 + bar_max_w, bar_bot, fill='#e2e8f0', outline='')

            # Value bar
            fill_w = max(2, int(bar_max_w * prob))
            if is_truth:
                bar_color = '#16a34a'
            elif is_top1:
                bar_color = '#dc2626' if (target_label is not None) else '#2563eb'
            else:
                bar_color = '#94a3b8'
            canvas.create_rectangle(bar_x0, bar_top, bar_x0 + fill_w, bar_bot, fill=bar_color, outline='')

            pct_txt = f'{prob * 100:.1f}%'
            if is_truth:
                pct_txt += ' *'
            canvas.create_text(
                bar_x0 + bar_max_w + 8,
                ymid,
                text=pct_txt,
                anchor='w',
                fill=lbl_color,
                font=lbl_font,
            )

    def _redraw_current_top5(self):
        if not self.report or not self.table.selection():
            self._draw_top5_bars(None, None)
            return
        name = self.table.selection()[0]
        target = (self.report.get('query') or {}).get('target')
        self._draw_top5_bars(self.report['results'].get(name), target)

    def _render_stage_galleries(self):
        tk = self.tk
        for container in (self.stage1_grid, self.stage2_grid):
            for child in container.winfo_children():
                child.destroy()
        self.stage_photo_refs = []

        stage1, stage2 = top_stage_candidates(self.report, limit=8)
        if not stage1 or self.engine is None:
            for container in (self.stage1_grid, self.stage2_grid):
                tk.Label(
                    container,
                    text='Chưa có dữ liệu ứng viên. Bấm "So sánh dự đoán" để xem Top-8 ở GĐ1 (Cosine) và GĐ2 (Selector).',
                    bg='#ffffff',
                    fg='#64748b',
                    font=('Segoe UI', 9),
                    pady=24,
                ).pack()
            return

        sel_rank1 = (self.report.get('pipeline') or {}).get('selected_stage1_rank')
        sel_label = (self.report.get('results', {}).get('selected') or {}).get('context_label', '')
        if sel_rank1 is not None and sel_rank1 > 8:
            self.stage1_note.set(
                f'Ảnh Selector chọn ("{sel_label}") ở hạng #{sel_rank1}/32 ở GĐ1 (ngoài Top-8) -> được đẩy lên #1 ở GĐ2!'
            )
        elif sel_rank1 is not None:
            self.stage1_note.set(
                f'Ảnh Selector chọn ("{sel_label}") nằm ở hạng #{sel_rank1}/32 ở GĐ1 -> lên #1 ở GĐ2.'
            )
        else:
            self.stage1_note.set('')

        self._populate_gallery_row(self.stage1_grid, stage1, mode='stage1')
        self._populate_gallery_row(self.stage2_grid, stage2, mode='stage2')

        # Auto-show selected candidate details in inspector
        selected_cand = next((c for c in stage2 if c.get('is_selected')), stage2[0])
        self._inspect_candidate(selected_cand)

    def _populate_gallery_row(self, container, items, mode):
        tk = self.tk
        from PIL import Image, ImageTk

        for col, cand in enumerate(items):
            is_winner = bool(cand.get('is_selected'))
            is_top1 = bool(cand.get('is_frozen_top1'))
            border_color = '#1d4ed8' if is_winner else ('#0f766e' if is_top1 else '#cbd5e1')
            bg_color = '#eff6ff' if is_winner else '#f8fafc'
            card = tk.Frame(
                container,
                bg=bg_color,
                highlightbackground=border_color,
                highlightthickness=2 if is_winner else 1,
                padx=4,
                pady=4,
                cursor='hand2',
            )
            card.grid(row=0, column=col, padx=3, pady=2, sticky='nsew')
            container.grid_columnconfigure(col, weight=1)

            rank_num = cand['stage1_rank'] if mode == 'stage1' else cand['stage2_rank']
            top_row = tk.Frame(card, bg=bg_color)
            top_row.pack(fill='x')
            tk.Label(top_row, text=f'#{rank_num}', font=('Segoe UI', 8, 'bold'), bg=bg_color, fg='#0f172a').pack(side='left')
            badges = candidate_badges(cand)
            if badges:
                b_text, b_color = badges[0]
                tk.Label(
                    top_row,
                    text=b_text,
                    font=('Segoe UI', 8, 'bold'),
                    bg=b_color,
                    fg='#ffffff',
                    padx=4,
                    pady=0,
                ).pack(side='right')

            img_holder = tk.Frame(card, bg='#ffffff', width=116, height=82)
            img_holder.pack(pady=3)
            img_holder.pack_propagate(False)
            raw = self.engine.raw_image(cand['dataset_index'])
            scale = min(112 / raw.width, 78 / raw.height)
            thumb = raw.resize(
                (max(1, round(raw.width * scale)), max(1, round(raw.height * scale))),
                Image.Resampling.NEAREST if raw.size == (32, 32) else Image.Resampling.LANCZOS,
            )
            photo = ImageTk.PhotoImage(thumb)
            self.stage_photo_refs.append(photo)
            img_lbl = tk.Label(img_holder, image=photo, bg='#ffffff')
            img_lbl.pack(expand=True)

            lbl = cand['label']
            short = lbl if len(lbl) <= 15 else lbl[:13] + '..'
            lbl_w = tk.Label(card, text=short, font=('Segoe UI', 8, 'bold'), bg=bg_color, fg='#0f172a')
            lbl_w.pack()
            if mode == 'stage1':
                meta_txt = f"cos={cand['cosine_score']:.3f} | GĐ2 #{cand['stage2_rank']}"
            else:
                meta_txt = f"điểm={cand['selector_score']:.2f} | GĐ1 #{cand['stage1_rank']}"
            meta_w = tk.Label(card, text=meta_txt, font=('Segoe UI', 8), bg=bg_color, fg='#475569')
            meta_w.pack()

            for w in (card, top_row, img_holder, img_lbl, lbl_w, meta_w):
                w.bind('<Button-1>', lambda e, c=cand: self._inspect_candidate(c))

    def _inspect_candidate(self, cand):
        roles = []
        if cand.get('is_selected'):
            roles.append('ĐƯỢC SELECTOR CHỌN (GĐ2 #1)')
        if cand.get('is_frozen_top1'):
            roles.append('Gần nhất theo Frozen Cosine (GĐ1 #1)')
        if cand.get('is_random'):
            roles.append('Mẫu Random trong top-32 (seed 44)')
        role_txt = ', '.join(roles) if roles else 'Ứng viên trong Top-32'
        self.candidate_detail.set(
            f"Ứng viên: {cand['label']} (Dataset ID={cand['dataset_index']}, Pool pos={cand['pool_position']})\n"
            f"Giai đoạn 1 (Frozen DINOv3 cosine): hạng #{cand['stage1_rank']}/32 | cosine = {cand['cosine_score']:.4f}\n"
            f"Giai đoạn 2 (Learned Selector):     hạng #{cand['stage2_rank']}/32 | dot-product score = {cand['selector_score']:.4f}\n"
            f"Vai trò trong bảng đối chứng:       {role_txt}"
        )

    def _populate_evidence_tab(self):
        self.details.delete('1.0', 'end')
        if not self.report:
            return
        meta = self.report.get('metadata') or {}
        query = self.report.get('query') or {}
        lines = [
            f"Dataset: {meta.get('dataset', 'cifar100')} | Preset: {meta.get('preset', '-')} | Device: {meta.get('device', '-')} | AMP: {meta.get('amp')}",
            f"Protocol: {meta.get('protocol', 'cifar100_seed42')} | Pool size: {meta.get('candidate_count', '-')}",
            f"Two-Stage checkpoint SHA256:  {meta.get('checkpoint_sha256', '-')}",
        ]
        if meta.get('no_context_checkpoint_sha256'):
            lines.append(f"No-context checkpoint SHA256: {meta.get('no_context_checkpoint_sha256')}")
        lines.extend([
            f"Query RGB SHA256:             {query.get('rgb_sha256', '-')}",
            f"Query info: kind={query.get('kind')} | target={query.get('target')} | original_size={query.get('original_size')}",
            f"Timing (batch=1, không phải benchmark): retrieval={self.report.get('retrieval_ms', 0):.1f} ms | "
            f"selector={self.report.get('selector_ms', 0):.1f} ms | total_comparison={self.report.get('comparison_ms', 0):.1f} ms",
            f"Flags: weights_updated={meta.get('weights_updated')} | test_evaluated={meta.get('test_evaluated')} | random_seed={self.report.get('random_seed')}",
            '',
            '--- CHI TIẾT TOP-5 TỪNG HÀNG ---',
        ])
        for name, item in self.report['results'].items():
            top5_str = ' | '.join(f"{x['label']} {x['probability']:.2%}" for x in item['top5'])
            ctx = '-' if item['context_index'] is None else f"{item['context_label']} (ID {item['context_index']}, {candidate_rank_text(item)})"
            sha = item.get('checkpoint_sha256', meta.get('checkpoint_sha256', '-'))
            lines.append(f"[{NAMES[name]}] ({strategy_group(name)})")
            lines.append(f"  Context: {ctx} | SHA256: {sha[:16]}...")
            lines.append(f"  Top-5:   {top5_str}")
        lines.append('')
        lines.append('Lưu ý: ' + str(self.report.get('warning', '')))
        self.details.insert('1.0', '\n'.join(lines))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'status':
                    self.status.set(value)
                elif kind == 'ready':
                    self.engine, self.busy = value, False
                    self.model_info.set(
                        f"{value.metadata['preset']} | best epoch {value.metadata['checkpoint_epoch']} | "
                        f"Validation lúc train {value.metadata['checkpoint_val_accuracy']:.2%} | {value.metadata['device']}"
                    )
                    if value.metadata.get('dataset') == 'caltech101':
                        self.model_info.set(
                            f"Two-Stage: best epoch {value.metadata['checkpoint_epoch']}, "
                            f"val {value.metadata['checkpoint_val_accuracy']:.2%} | "
                            f"No-context riêng: epoch {value.metadata['no_context_checkpoint_epoch']}, "
                            f"val {value.metadata['no_context_val_accuracy']:.2%} | {value.metadata['device']}"
                        )
                    self.status.set('Sẵn sàng. Chọn ảnh rồi bấm "So sánh dự đoán" hoặc "Xem từng bước".')
                    self.load_validation()
                elif kind == 'result':
                    self.report, self.busy = value, False
                    for name, item in value['results'].items():
                        correct = item['correct']
                        context = '-' if item['context_index'] is None else f"{item['context_label']} / {item['context_index']}"
                        tags = []
                        if correct is True:
                            tags.append('correct')
                        elif correct is False:
                            tags.append('wrong')
                        if name == 'trained_no_context':
                            tags.append('control')
                        self.table.insert(
                            '',
                            'end',
                            iid=name,
                            tags=tuple(tags),
                            values=(
                                NAMES[name],
                                strategy_group(name),
                                item['prediction'],
                                '-' if correct is None else ('Đúng' if correct else 'Sai'),
                                f"{item['top5'][0]['probability']:.1%}",
                                context,
                                candidate_rank_text(item),
                                f"{item['task_ms']:.1f}",
                            ),
                        )
                    tone, summary = summarize_outcome(value)
                    self._set_verdict(tone, summary)
                    self._render_stage_galleries()
                    self._populate_evidence_tab()
                    self.table.selection_set('selected')
                    self.show_context()
                    self.status.set(
                        f"Xong | GĐ1 retrieval {value['retrieval_ms']:.1f} ms | "
                        f"GĐ2 selector {value['selector_ms']:.1f} ms | "
                        f"tổng so sánh {value['comparison_ms']:.1f} ms (không phải benchmark)"
                    )
                    if self.pending_walkthrough:
                        self.pending_walkthrough = False
                        self.select_step(1)
                    else:
                        self.select_step(4)
                elif kind == 'error':
                    self.busy = False
                    self.pending_walkthrough = False
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
        target = (self.report.get('query') or {}).get('target')

        if item['context_index'] is None:
            msg = (
                'Không dùng ảnh phụ\n(Checkpoint No-context train riêng 100 ep)'
                if name == 'trained_no_context'
                else 'Không dùng ảnh phụ\n(Bỏ context trên cùng weights Two-Stage)'
            )
            self.context_view.configure(image='', text=msg)
            self.context_caption.set(strategy_group(name))
            self.context_connector.set('(không dùng\nảnh phụ)')
        else:
            self.show_image(self.context_view, self.engine.raw_image(item['context_index']))
            rank_txt = candidate_rank_text(item)
            cos_v = item.get('cosine_score')
            sel_v = item.get('selector_score')
            score_line = ''
            if isinstance(cos_v, (int, float)) and isinstance(sel_v, (int, float)):
                score_line = f' | cos={cos_v:.3f}, sel={sel_v:.2f}'
            self.context_caption.set(
                f"{item['context_label']} (ID={item['context_index']})\n{rank_txt}{score_line}"
            )
            self.context_connector.set('+ ghép patch\n(role_v2)')

        self.top5_title.set(f"TOP-5 XÁC SUẤT - {NAMES[name]} ({strategy_group(name)})")
        self._draw_top5_bars(item, target)
        sha = item.get('checkpoint_sha256', self.report['metadata'].get('checkpoint_sha256', '-'))
        desc = STRATEGY_DESCRIPTIONS.get(name, '')
        self.row_explain.set(f"{NAMES[name]}: {desc} (SHA256: {sha[:12]}...)")

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
    parser.add_argument('--dataset', choices=list(DATASETS), default='cifar100')
    parser.add_argument('--preset', choices=[*PRESETS, 'full100'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--validation-position', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-ui', action='store_true')
    args = parser.parse_args()
    args.preset = args.preset or DATASETS[args.dataset]['presets'][0]
    try:
        if args.preset not in DATASETS[args.dataset]['presets']:
            raise ValueError('Preset không thuộc dataset đã chọn')
        if args.headless:
            if args.output and args.output.exists():
                raise FileExistsError('Output đã tồn tại; chọn tên mới')
            size = DATASETS[args.dataset]['validation_size']
            if not 0 <= args.validation_position < size:
                raise ValueError(f'Validation position phải trong 0..{size - 1}')
            engine = create_engine(args.preset, args.device, dataset=args.dataset)
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
            ShowcaseApp(root, args.preset, args.device, initialize=not args.check_ui, dataset=args.dataset)
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
