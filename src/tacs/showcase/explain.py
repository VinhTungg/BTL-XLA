"""Plain-language explanations and pipeline summaries for the TACS showcase GUI."""
from __future__ import annotations

STRATEGY_GROUPS = {
    'selected': 'Cùng weights Two-Stage',
    'frozen_cosine': 'Cùng weights Two-Stage',
    'random': 'Cùng weights Two-Stage',
    'no_context': 'Cùng weights Two-Stage',
    'trained_no_context': 'Checkpoint riêng (100 ep)',
}

STRATEGY_DESCRIPTIONS = {
    'selected': (
        'GĐ1 lọc Top-32 bằng Frozen DINOv3 (cosine) -> GĐ2 Selector đã học chấm điểm '
        'và chọn 1 ảnh context -> Task Network (role_v2) ghép patch Query + Context để phân loại.'
    ),
    'frozen_cosine': (
        'Can thiệp trên cùng trọng số Task Network Two-Stage: lấy thẳng ảnh giống nhất '
        'ở GĐ1 (hạng #1 theo cosine của Frozen DINOv3), bỏ qua bước chọn của Selector.'
    ),
    'random': (
        'Can thiệp trên cùng trọng số Task Network Two-Stage: chọn ngẫu nhiên 1 ảnh '
        'trong Top-32 của GĐ1 (1 lần lấy mẫu với seed 44, không phải trung bình nhiều lần).'
    ),
    'no_context': (
        'Can thiệp trên cùng trọng số Task Network Two-Stage: chỉ đưa ảnh Query vào '
        'Task Network và bỏ hoàn toàn ảnh phụ để kiểm tra tác dụng cục bộ của context.'
    ),
    'trained_no_context': (
        'Baseline độc lập (checkpoint riêng đã train 100 epoch không dùng context, SHA256 khác). '
        'Dùng để đối chứng công bằng ở mức hệ thống, không đồng nhất với hàng Bỏ context ở trên.'
    ),
}

HOW_IT_WORKS_TEXT = """CÁCH MÔ HÌNH TACS TWO-STAGE HOẠT ĐỘNG (SUY LUẬN OFFLINE)

1. Đầu vào (Query):
   - Ảnh gốc được hiển thị đúng tỷ lệ để dễ quan sát, còn mô hình nhận bản Resize 256x256
     và chuẩn hóa theo ImageNet. Không train lại và không cập nhật trọng số khi chạy demo.

2. Giai đoạn 1 - Lọc ứng viên (Frozen DINOv3 Retrieval):
   - Encoder DINOv3 ViT-S/16 cố định (frozen) trích xuất vector đặc trưng cho Query,
     so khớp độ tương đồng cosine với toàn bộ kho ảnh train cố định (606 ảnh ở Caltech-101
     hoặc 9.000 ảnh ở CIFAR-100) để lọc ra Top-32 ứng viên giống nhất.

3. Giai đoạn 2 - Chọn ảnh hỗ trợ (Learned Selector):
   - Selector đã được huấn luyện theo bài toán chấm điểm lại 32 ứng viên bằng tích vô hướng
     (dot product) và chọn đúng 1 ảnh context có điểm cao nhất.
   - Điểm quan trọng: ảnh giống nhất theo cosine (GĐ1 #1) chưa chắc là ảnh mà Selector chọn
     (GĐ2 #1), và ảnh context cũng không bắt buộc phải cùng lớp với Query.

4. Giai đoạn 3 - Phân loại (Task Network role_v2):
   - Ảnh Query và ảnh Context được chia thành các patch token (16x16), bổ sung vector vai trò
     (role_v2) cho patch của Context để phân biệt vai trò, rồi đưa qua Transformer để ra Top-5.
   - Nhãn thật của ảnh Context KHÔNG được đưa vào để suy luận nhãn Query.

5. Cách đọc bảng so sánh (tránh hiểu nhầm):
   - 4 hàng đầu (Selector đã học, Ảnh gần nhất, Random top-32, Bỏ context) đều dùng CÙNG
     trọng số Task Network Two-Stage để khảo sát ảnh hưởng của cách chọn context.
   - Hàng "No-context train riêng" (trên Caltech-101) là một checkpoint baseline độc lập
     với SHA256 riêng. Một ví dụ context giúp không chứng minh Two-Stage thắng toàn tập.
"""


def strategy_group(name: str) -> str:
    return STRATEGY_GROUPS.get(name, 'Khác')


def candidate_rank_text(item: dict) -> str:
    r1 = item.get('stage1_rank')
    r2 = item.get('stage2_rank')
    if r1 is None and r2 is None:
        return '-'
    if r1 is not None and r2 is not None:
        return f'GĐ1 #{r1} -> GĐ2 #{r2}'
    if r1 is not None:
        return f'GĐ1 #{r1}'
    return f'GĐ2 #{r2}'


def top_stage_candidates(report: dict | None, limit: int = 8) -> tuple[list[dict], list[dict]]:
    if not report:
        return [], []
    pipeline = report.get('pipeline') or {}
    candidates = list(pipeline.get('candidates') or [])
    if not candidates:
        return [], []
    stage1 = sorted(candidates, key=lambda c: c['stage1_rank'])[:limit]
    stage2 = sorted(candidates, key=lambda c: c['stage2_rank'])[:limit]
    return stage1, stage2


def candidate_badges(cand: dict) -> list[tuple[str, str]]:
    """Return (label, color_hex) badges for a candidate thumbnail."""
    badges = []
    if cand.get('is_selected'):
        badges.append(('GĐ2 CHỌN', '#1d4ed8'))
    if cand.get('is_frozen_top1'):
        badges.append(('GĐ1 #1', '#0f766e'))
    if cand.get('is_random'):
        badges.append(('Random', '#9333ea'))
    return badges


def summarize_outcome(report: dict | None) -> tuple[str, str]:
    """Return (tone, plain-Vietnamese summary) strictly grounded in real report data.

    tone is one of: 'neutral', 'help', 'hurt', 'ok', 'warn'.
    """
    if not report:
        return ('neutral', 'Chọn ảnh rồi bấm "So sánh dự đoán" hoặc "Xem từng bước" để theo dõi luồng chạy 3 giai đoạn.')

    results = report.get('results') or {}
    query = report.get('query') or {}
    selected = results.get('selected')
    no_ctx = results.get('no_context')
    control = results.get('trained_no_context')
    if not selected or not no_ctx:
        return ('neutral', 'Đã có kết quả suy luận. Bấm từng hàng trong bảng để xem ảnh context và Top-5.')

    sel_pred = selected['prediction']
    sel_prob = selected['top5'][0]['probability'] * 100.0
    no_pred = no_ctx['prediction']
    no_prob = no_ctx['top5'][0]['probability'] * 100.0
    ctx_label = selected.get('context_label') or '?'
    r1 = selected.get('stage1_rank')
    rank_hint = f' (hạng #{r1} ở GĐ1 -> #1 ở GĐ2)' if r1 is not None else ''

    target = query.get('target')
    if target is None:
        diff_note = (
            f'bỏ context cùng weights đoán "{no_pred}" ({no_prob:.1f}%).'
            if no_pred != sel_pred
            else f'bỏ context cùng weights cũng đoán "{no_pred}" ({no_prob:.1f}%).'
        )
        return (
            'warn',
            f'Ảnh riêng (không có nhãn thật): Selector chọn context "{ctx_label}"{rank_hint}, '
            f'dự đoán "{sel_pred}" ({sel_prob:.1f}%); {diff_note} Không tự chấm đúng/sai.',
        )

    sel_ok = bool(selected.get('correct'))
    no_ok = bool(no_ctx.get('correct'))
    ctrl_note = ''
    if control is not None:
        ctrl_ok = bool(control.get('correct'))
        ctrl_pred = control['prediction']
        ctrl_prob = control['top5'][0]['probability'] * 100.0
        if ctrl_ok:
            ctrl_note = f' Đối chứng độc lập (No-context train riêng) cũng đoán ĐÚNG "{ctrl_pred}" ({ctrl_prob:.1f}%).'
        else:
            ctrl_note = f' Đối chứng độc lập (No-context train riêng) đoán SAI "{ctrl_pred}" ({ctrl_prob:.1f}%).'

    if sel_ok and not no_ok:
        return (
            'help',
            f'Context CÓ LỢI (cùng weights Two-Stage): có context "{ctx_label}"{rank_hint} đoán ĐÚNG '
            f'"{sel_pred}" ({sel_prob:.1f}%), còn bỏ context đoán SAI "{no_pred}" ({no_prob:.1f}%).'
            f'{ctrl_note}',
        )
    if (not sel_ok) and no_ok:
        return (
            'hurt',
            f'Context GÂY NHIỄU (cùng weights Two-Stage): chọn context "{ctx_label}"{rank_hint} làm đoán SAI '
            f'"{sel_pred}" ({sel_prob:.1f}%), trong khi bỏ context đoán ĐÚNG "{no_pred}" ({no_prob:.1f}%).'
            f'{ctrl_note}',
        )
    if sel_ok and no_ok:
        delta = sel_prob - no_prob
        sign = '+' if delta >= 0 else ''
        return (
            'ok',
            f'CÙNG ĐÚNG trên weights Two-Stage: đều đoán "{sel_pred}" (có context "{ctx_label}"{rank_hint}: '
            f'{sel_prob:.1f}%, bỏ context: {no_prob:.1f}%, chênh {sign}{delta:.1f} điểm %).{ctrl_note}',
        )
    return (
        'hurt',
        f'CÙNG SAI trên weights Two-Stage (nhãn thật: "{target}"): có context "{ctx_label}"{rank_hint} đoán '
        f'"{sel_pred}" ({sel_prob:.1f}%), bỏ context đoán "{no_pred}" ({no_prob:.1f}%).{ctrl_note}',
    )


def pipeline_card_texts(
    report: dict | None,
    query: dict | None,
    metadata: dict | None,
    dataset_title: str,
) -> list[dict]:
    """Return 4 step-card payloads (title, metric, detail) for the pipeline strip."""
    pool_count = (metadata or {}).get('candidate_count')
    pool_txt = f'{pool_count:,}' if isinstance(pool_count, int) else 'pool'
    num_classes = (metadata or {}).get('num_classes') or '?'

    if query is None:
        c1 = dict(title='1. Ảnh Query', metric='Chưa chọn ảnh', detail='Chọn validation hoặc mở ảnh riêng')
    else:
        size = query.get('original_size') or ['?', '?']
        target = query.get('target')
        label_txt = f'Nhãn: {target}' if target is not None else 'Ảnh riêng (chưa rõ nhãn)'
        c1 = dict(title='1. Ảnh Query', metric=label_txt, detail=f'Gốc {size[0]}x{size[1]} -> 256x256')

    if not report:
        return [
            c1,
            dict(title='2. GĐ1: Retrieval (Frozen)', metric=f'Kho {pool_txt} -> Top-32', detail='Frozen DINOv3 lọc theo cosine'),
            dict(title='3. GĐ2: Selector đã học', metric='32 ứng viên -> Chọn 1', detail='Chấm điểm ích lợi cho tác vụ'),
            dict(title='4. GĐ3: Task Network', metric=f'Phân loại {num_classes} lớp', detail='Ghép patch Query + Context (role_v2)'),
        ]

    results = report.get('results') or {}
    pipeline = report.get('pipeline') or {}
    topk_n = pipeline.get('topk_count', 32)
    pool_n = pipeline.get('pool_size', pool_count or '?')
    pool_str = f'{pool_n:,}' if isinstance(pool_n, int) else str(pool_n)

    frozen = results.get('frozen_cosine') or {}
    f_label = frozen.get('context_label') or '-'
    f_cos = frozen.get('cosine_score')
    f_score_txt = f'cos={f_cos:.3f}' if isinstance(f_cos, (int, float)) else ''
    ret_ms = report.get('retrieval_ms')
    ret_ms_txt = f'{ret_ms:.0f} ms' if isinstance(ret_ms, (int, float)) else ''
    c2 = dict(
        title='2. GĐ1: Retrieval (Frozen)',
        metric=f'{pool_str} -> Top-{topk_n}',
        detail=f'Gần nhất: {f_label} {f_score_txt} ({ret_ms_txt})'.strip(),
    )

    sel = results.get('selected') or {}
    s_label = sel.get('context_label') or '-'
    s_r1 = sel.get('stage1_rank')
    s_score = sel.get('selector_score')
    rank_str = f'GĐ1 #{s_r1} -> #1' if s_r1 is not None else 'Chọn #1'
    s_score_txt = f' | điểm {s_score:.2f}' if isinstance(s_score, (int, float)) else ''
    c3 = dict(
        title='3. GĐ2: Selector đã học',
        metric=f'Chọn: {s_label}',
        detail=f'{rank_str}{s_score_txt}',
    )

    pred = sel.get('prediction') or '-'
    prob = (sel.get('top5') or [{}])[0].get('probability')
    prob_txt = f'{prob * 100:.1f}%' if isinstance(prob, (int, float)) else ''
    correct = sel.get('correct')
    verdict = 'ĐÚNG' if correct is True else ('SAI' if correct is False else 'Chưa rõ nhãn')
    c4 = dict(
        title='4. GĐ3: Task Network',
        metric=f'{pred} ({prob_txt})'.strip(),
        detail=f'{verdict} | {dataset_title} ({num_classes} lớp)',
    )
    return [c1, c2, c3, c4]


def step_narration(step: int, report: dict | None, query: dict | None, dataset_title: str) -> str:
    """Return focused Vietnamese narration for interactive step 1..4."""
    if not report:
        return (
            'Bấm "So sánh dự đoán" để chạy ngay, hoặc bấm "Xem từng bước" để đi qua lần lượt '
            'Bước 1 (Query) -> Bước 2 (GĐ1 Retrieval) -> Bước 3 (GĐ2 Selector) -> Bước 4 (GĐ3 Task Network).'
        )
    results = report.get('results') or {}
    pipeline = report.get('pipeline') or {}
    pool_n = pipeline.get('pool_size') or (report.get('metadata') or {}).get('candidate_count') or '?'
    topk_n = pipeline.get('topk_count', 32)
    sel = results.get('selected') or {}
    frozen = results.get('frozen_cosine') or {}
    no_ctx = results.get('no_context') or {}

    if step == 1:
        size = (query or {}).get('original_size') or ['?', '?']
        target = (query or {}).get('target')
        truth_str = f'nhãn thật là "{target}"' if target is not None else 'là ảnh riêng (không có nhãn thật)'
        return (
            f'BƯỚC 1 - ẢNH ĐẦU VÀO (QUERY): Ảnh {dataset_title} kích thước gốc {size[0]}x{size[1]} ({truth_str}). '
            f'Trước khi vào mạng, ảnh được Resize 256x256 và chuẩn hóa ImageNet. '
            f'Kho ứng viên (candidate pool) cố định gồm {pool_n} ảnh train, tách biệt hoàn toàn với tập validation.'
        )
    if step == 2:
        f_label = frozen.get('context_label') or '-'
        f_cos = frozen.get('cosine_score')
        f_cos_str = f'{f_cos:.3f}' if isinstance(f_cos, (int, float)) else '-'
        s_r1 = sel.get('stage1_rank')
        s_label = sel.get('context_label') or '-'
        extra = (
            f' Ảnh mà Selector chọn ở bước sau ("{s_label}") đang nằm ở hạng #{s_r1} trong danh sách cosine này.'
            if s_r1 is not None
            else ''
        )
        return (
            f'BƯỚC 2 - GIAI ĐOẠN 1 (FROZEN RETRIEVAL): Encoder DINOv3 cố định so cosine giữa Query với toàn bộ '
            f'{pool_n} ảnh pool để lọc nhanh Top-{topk_n} ứng viên gần nhất ({report.get("retrieval_ms", 0):.1f} ms). '
            f'Ảnh gần nhất (GĐ1 #1) là "{f_label}" (cosine={f_cos_str}).{extra}'
        )
    if step == 3:
        s_label = sel.get('context_label') or '-'
        s_r1 = sel.get('stage1_rank')
        s_score = sel.get('selector_score')
        f_label = frozen.get('context_label') or '-'
        s_score_str = f'{s_score:.2f}' if isinstance(s_score, (int, float)) else '-'
        same_as_top1 = (s_r1 == 1)
        contrast = (
            'Ở ảnh này, Selector đồng thuận với GĐ1 #1.'
            if same_as_top1
            else f'Selector ĐẢO HẠNG: đẩy ảnh "{s_label}" từ hạng #{s_r1} (GĐ1) lên #1 (vượt qua ảnh gần nhất "{f_label}").'
        )
        return (
            f'BƯỚC 3 - GIAI ĐOẠN 2 (LEARNED SELECTOR): Selector đã học chấm điểm lại {topk_n} ứng viên theo '
            f'tích vô hướng đặc trưng đã học ({report.get("selector_ms", 0):.1f} ms) và chọn "{s_label}" '
            f'(điểm {s_score_str}). {contrast} Lưu ý: nhãn của ảnh context không được cấp cho mô hình.'
        )
    sel_pred = sel.get('prediction') or '-'
    sel_p = (sel.get('top5') or [{}])[0].get('probability', 0.0) * 100.0
    no_pred = no_ctx.get('prediction') or '-'
    no_p = (no_ctx.get('top5') or [{}])[0].get('probability', 0.0) * 100.0
    return (
        f'BƯỚC 4 - GIAI ĐOẠN 3 (TASK NETWORK role_v2 & SO SÁNH): Task Network ghép patch của Query với patch của '
        f'Context (đã cộng vector vai trò role_v2) -> dự đoán "{sel_pred}" ({sel_p:.1f}%), so với khi bỏ context '
        f'trên cùng weights là "{no_pred}" ({no_p:.1f}%). Bấm từng hàng trong bảng bên dưới để đổi góc nhìn.'
    )
