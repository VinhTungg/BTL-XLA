"""Kiểm tra trực quan ít nhất 20 query và ghi nhận lỗi - Nhiệm vụ Thành viên 3 (Gói 5).

Script này:
1. Lấy mẫu 20 query cùng 8 candidates tương ứng (tổng cộng 160 candidates).
2. Kiểm tra tính toàn vẹn:
   - Không trùng index giữa query và candidates (chống data leakage).
   - Kiểm tra phân bố lớp candidate (tỷ lệ cùng lớp vs khác lớp).
   - Kiểm tra dải giá trị pixel sau khi khôi phục chuẩn hóa (denormalization).
3. Xuất hình ảnh trực quan chất lượng cao lưu tại docs/visual_inspection_20_queries.png.
4. Ghi nhận nhật ký audit chi tiết từng query phục vụ báo cáo nghiệm thu.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from tacs.data import ContextDataset
from tacs.device import configure_stdout
from tacs.visualize_batch import _denormalize_image, make_context_grid

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def create_mock_cifar10_pool(seed: int = 42) -> tuple[Subset, Subset]:
    """Tạo tập dữ liệu mô phỏng CIFAR-10 chuẩn hóa để kiểm thử trực quan."""
    generator = torch.Generator().manual_seed(seed)
    
    # 500 mẫu mô phỏng (chuẩn hóa theo mean/std của CIFAR-10)
    # mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)
    num_samples = 500
    images = torch.randn(num_samples, 3, 32, 32, generator=generator)
    # 10 lớp cân bằng
    targets = torch.tensor([i % 10 for i in range(num_samples)], dtype=torch.long)
    
    dataset = TensorDataset(images, targets)
    query_subset = Subset(dataset, list(range(100)))
    candidate_subset = Subset(dataset, list(range(100, num_samples)))
    return query_subset, candidate_subset


def main() -> None:
    configure_stdout()
    print("=" * 80)
    print("   KIỂM TRA TRỰC QUAN 20 QUERIES & GHI NHẬN LỖI (GÓI 5 - THÀNH VIÊN 3)")
    print("=" * 80)

    query_ds, cand_ds = create_mock_cifar10_pool(seed=42)
    num_queries_to_test = 20
    num_candidates = 8

    context_ds = ContextDataset(
        query_dataset=query_ds,
        candidate_dataset=cand_ds,
        num_candidates=num_candidates,
        seed=42,
    )

    loader = DataLoader(context_ds, batch_size=num_queries_to_test, shuffle=False)
    batch = next(iter(loader))

    queries = batch["query"]
    candidates = batch["candidates"]
    targets = batch["target"]
    query_indices = batch["query_index"]
    candidate_indices = batch["candidate_indices"]
    candidate_targets = batch["candidate_targets"]

    print(f"[*] Số lượng query kiểm tra      : {num_queries_to_test}")
    print(f"[*] Số lượng candidates mỗi query: {num_candidates}")
    print(f"[*] Tổng số cặp ảnh được audit   : {num_queries_to_test * num_candidates}")
    print("-" * 80)

    # Bảng nhật ký kiểm tra 20 query
    print(f"{'Q#':<4} | {'Q_Idx':<6} | {'Class Q':<12} | {'Same-Class Cands':<17} | {'Diff-Class':<11} | {'Leakage':<8} | {'Pixel Range':<15} | {'Trạng thái'}")
    print("-" * 95)

    leakage_detected = False
    anomaly_detected = False
    audit_records = []

    for i in range(num_queries_to_test):
        q_idx = int(query_indices[i].item())
        q_target = int(targets[i].item())
        q_name = CIFAR_CLASSES[q_target]
        c_idxs = candidate_indices[i].tolist()
        c_targets = candidate_targets[i].tolist()

        # 1. Kiểm tra leakage
        is_leaking = q_idx in c_idxs
        if is_leaking:
            leakage_detected = True

        # 2. Đếm candidate cùng lớp vs khác lớp
        same_class_count = sum(1 for ct in c_targets if ct == q_target)
        diff_class_count = num_candidates - same_class_count

        # 3. Kiểm tra dải giá trị pixel sau denormalize
        denorm_q = _denormalize_image(queries[i])
        denorm_c = _denormalize_image(candidates[i])
        min_val = min(denorm_q.min().item(), denorm_c.min().item())
        max_val = max(denorm_q.max().item(), denorm_c.max().item())
        has_nan = torch.isnan(denorm_q).any().item() or torch.isnan(denorm_c).any().item()

        pixel_ok = (0.0 <= min_val) and (max_val <= 1.0) and not has_nan
        if not pixel_ok or is_leaking:
            status = "FAIL"
            anomaly_detected = True
        else:
            status = "PASS"

        range_str = f"[{min_val:.2f}, {max_val:.2f}]"
        print(f"#{i+1:<3} | {q_idx:<6} | {q_name:<12} | {same_class_count:<17} | {diff_class_count:<11} | {'LỖI!' if is_leaking else 'Không':<8} | {range_str:<15} | {status}")

        audit_records.append({
            "query_no": i + 1,
            "query_idx": q_idx,
            "query_class": q_name,
            "query_label": q_target,
            "same_class": same_class_count,
            "diff_class": diff_class_count,
            "candidate_labels": [CIFAR_CLASSES[c] for c in c_targets],
            "min_val": min_val,
            "max_val": max_val,
            "status": status,
        })

    print("-" * 95)
    print("\n--- TỔNG KẾT NGHIỆM THU KIỂM TRA TRỰC QUAN (GÓI 5) ---")
    print(f"[*] Rò rỉ dữ liệu (Query trùng Candidate Pool): {'CÓ (LỖI)' if leakage_detected else 'KHÔNG (0/20 query)'}")
    print(f"[*] Dải giá trị hình ảnh [0, 1] hợp lệ        : {'KHÔNG HỢP LỆ' if anomaly_detected else 'ĐẠT (20/20 query)'}")
    print(f"[*] Tỷ lệ candidate cùng lớp trung bình       : {sum(r['same_class'] for r in audit_records) / (num_queries_to_test * num_candidates) * 100:.2f}%")
    print(f"[*] Tỷ lệ candidate khác lớp trung bình       : {sum(r['diff_class'] for r in audit_records) / (num_queries_to_test * num_candidates) * 100:.2f}%")

    # 4. Xuất ảnh trực quan vào docs/visual_inspection_20_queries.png
    output_dir = Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "visual_inspection_20_queries.png"

    fig = make_context_grid(batch, num_queries=num_queries_to_test, figsize=(18, 2.4 * num_queries_to_test))
    fig.suptitle(
        "Nghiệm thu trực quan Tuần 1 (Thành viên 3): 20 Queries x 8 Candidates (160 Context Images)",
        fontsize=14,
        y=1.002,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Đã lưu hình ảnh trực quan kiểm tra tại : {out_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
