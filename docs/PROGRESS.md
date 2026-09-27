# Tiến độ

Cập nhật: 2026-09-20

| Hạng mục                      | Trạng thái | Ghi chú                                                          |
| ----------------------------- | ---------- | ---------------------------------------------------------------- |
| Đọc và phân tích paper        | Xong       | Bản Anh là nguồn chuẩn                                           |
| Kiểm kê source                | Xong       | Không có code TACS chính thức trong workspace                    |
| Phân tích dự án tham khảo     | Xong       | CLIP + Weaviate; không tái sử dụng cho lõi training              |
| Skeleton TACS                 | Xong       | Selector, classifier, hybrid loss                                |
| Unit test CPU                 | Xong       | 7/7 test pass; kiểm tra split, sampling, forward và gradient     |
| Giải thích TACS trong báo cáo | Xong       | Có ví dụ, luồng xử lý, Gumbel và reward                          |
| Pipeline CIFAR-10             | Xong       | Split 36.000/9.000/5.000/10.000; candidate sampling tái lập được |
| Benchmark DataLoader          | Xong       | Môi trường sạch: 426,33 query/s trên CPU; RSS 773,31 MB          |
| Baseline huấn luyện thật      | Xong       | Hoàn thành 3 baseline: No-context, Random, Similarity (M2)       |
| Reproduction                  | Chưa làm   | Trọng tâm Tuần 3 (M3)                                            |
| Cải tiến two-stage            | Chưa làm   | Tuần 4 (M4)                                                      |

## Nghiệm thu Tuần 2 - Milestone M2 (3 Baseline Đối sánh)

- **Người 1:** Xây dựng hạ tầng huấn luyện chung [src/tacs/train.py](file:///e:/BTL-XLA/src/tacs/train.py), CLI chọn baseline, hệ thống quản lý checkpoint và module trích xuất đặc trưng & Cosine Similarity Retrieval [src/tacs/retrieval.py](file:///e:/BTL-XLA/src/tacs/retrieval.py).
- **Người 2:** Viết bộ kiểm thử hạ tầng [tests/test_baselines.py](file:///e:/BTL-XLA/tests/test_baselines.py), triển khai mô hình chung [src/tacs/baselines.py](file:///e:/BTL-XLA/src/tacs/baselines.py), huấn luyện No-context và khóa encoder chuẩn tại `outputs/no_context/best_encoder.pt`.
- **Người 3:** Xây dựng Random-context trên `ContextDataset`, tích hợp `SimilarityContextDataset`, huấn luyện Similarity-context, xây dựng script phân tích [scripts/analyze_baselines.py](file:///e:/BTL-XLA/scripts/analyze_baselines.py), vẽ biểu đồ [docs/baseline_comparison_curves.png](file:///e:/BTL-XLA/docs/baseline_comparison_curves.png) và xuất bản báo cáo chi tiết [docs/BAO_CAO_TIEN_DO_TUAN_2.md](file:///e:/BTL-XLA/docs/BAO_CAO_TIEN_DO_TUAN_2.md).
- **Hệ thống kiểm thử:** Đạt **18/18 unit tests pass 100%**.

## Việc tiếp theo

1. Chuyển sang Tuần 3 (Milestone M3 - TACS Reproduction).
2. Xây dựng vòng lặp huấn luyện phối hợp: tối ưu Selector bằng Gumbel-Softmax + Policy Gradient REINFORCE và Classifier bằng Cross-Entropy Task Loss.
3. Thực hiện nghiên cứu thực nghiệm bóc tách (Ablation Study) và so sánh hiệu quả với 3 baseline Tuần 2.

