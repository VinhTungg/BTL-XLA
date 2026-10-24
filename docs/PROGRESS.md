# Tiến độ

Cập nhật: 2026-10-24

| Hạng mục                      | Trạng thái | Ghi chú                                                          |
| ----------------------------- | ---------- | ---------------------------------------------------------------- |
| Đọc và phân tích paper        | Xong       | Bản Anh là nguồn chuẩn                                           |
| Kiểm kê source                | Xong       | Không có code TACS chính thức trong workspace                    |
| Phân tích dự án tham khảo     | Xong       | CLIP + Weaviate; không tái sử dụng cho lõi training              |
| Skeleton TACS                 | Xong       | Selector, classifier, hybrid loss                                |
| Unit test CPU                 | Xong       | 37 test pass, 1 integration skip mặc định; bật cờ thì integration đạt 1/1 |
| Giải thích TACS trong báo cáo | Xong       | Có ví dụ, luồng xử lý, Gumbel và reward                          |
| Pipeline CIFAR-10             | Xong       | Split 36.000/9.000/5.000/10.000; candidate sampling tái lập được |
| Benchmark DataLoader          | Xong       | Môi trường sạch: 426,33 query/s trên CPU; RSS 773,31 MB          |
| Baseline huấn luyện thật      | Xong       | Hoàn thành 3 baseline: No-context, Random, Similarity (M2)       |
| Reproduction                  | Xong       | Đã train/test 5 epoch cho đủ 5 mô hình trên CIFAR-10 thật      |
| Cải tiến two-stage            | Xong       | Top-k dùng shared embedding bank; benchmark CPU nhanh hơn 7,27x   |
| Ablation TACS/Two-Stage        | Xong       | Temperature, policy weight và top-K; có xác nhận 5 epoch       |
| Báo cáo tổng thể            | Xong       | Đã tích hợp benchmark, ablation và mean ± SD ba seed        |

## Nghiệm thu Tuần 2 - Milestone M2 (3 Baseline Đối sánh)

- **Người 1:** Xây dựng hạ tầng huấn luyện chung [src/tacs/train.py](file:///e:/BTL-XLA/src/tacs/train.py), CLI chọn baseline, hệ thống quản lý checkpoint và module trích xuất đặc trưng & Cosine Similarity Retrieval [src/tacs/retrieval.py](file:///e:/BTL-XLA/src/tacs/retrieval.py).
- **Người 2:** Viết bộ kiểm thử hạ tầng [tests/test_baselines.py](file:///e:/BTL-XLA/tests/test_baselines.py), triển khai mô hình chung [src/tacs/baselines.py](file:///e:/BTL-XLA/src/tacs/baselines.py), huấn luyện No-context và khóa encoder chuẩn tại `outputs/no_context/best_encoder.pt`.
- **Người 3:** Xây dựng Random-context trên `ContextDataset`, tích hợp `SimilarityContextDataset`, huấn luyện Similarity-context, xây dựng script phân tích [scripts/analyze_baselines.py](file:///e:/BTL-XLA/scripts/analyze_baselines.py), vẽ biểu đồ [docs/baseline_comparison_curves.png](file:///e:/BTL-XLA/docs/baseline_comparison_curves.png) và xuất bản báo cáo chi tiết [docs/BAO_CAO_TIEN_DO_TUAN_2.md](file:///e:/BTL-XLA/docs/BAO_CAO_TIEN_DO_TUAN_2.md).
- **Hệ thống kiểm thử:** Đạt **18/18 unit tests pass 100%**.

## Việc tiếp theo

1. Chuẩn bị slide thuyết trình và kịch bản demo nếu môn học yêu cầu.
2. Nếu cần khẳng định thống kê, mở rộng lên ít nhất 5 seed.
3. Phân tích selector/reward để giảm độ nhạy Gumbel giữa các seed.

## Benchmark chính thức 5 epoch

Kết quả ngày 2026-10-24 trên CPU, batch size 64, toàn bộ split
36.000/9.000/5.000/10.000:

| Mô hình | Validation | Test |
|---|---:|---:|
| No-context | 41,92% | 42,47% |
| Random-context | 42,02% | 42,54% |
| Similarity-context | **42,44%** | **42,67%** |
| TACS Full | 40,76% | 42,05% |
| Two-Stage TACS | 41,22% | 41,91% |

