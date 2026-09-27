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
| Baseline huấn luyện thật      | Chưa làm   | Bước đầu của M2                                                  |
| Reproduction                  | Chưa làm   | M2                                                               |
| Cải tiến two-stage            | Chưa làm   | M3                                                               |

## Nghiệm thu Người 1 - Tuần 1

- Cài đặt thành công project trong môi trường `.venv-clean` từ hướng dẫn README.
- 7/7 unit test đạt trong 0,205 giây.
- Smoke training chạy đủ 5 bước; loss hữu hạn và gradient hoạt động.
- DataLoader xử lý 3.200 query trong 7,506 giây trên CPU.
- README đã được cập nhật theo trạng thái thực tế của project.

## Nghiệm thu Người 2 - Tuần 1

- Đã thêm script trực quan hóa query và candidates tại `src/tacs/visualize_batch.py`.
- Đã thêm unit test cho chức năng trực quan hóa tại `tests/test_visualize.py`.
- Unit test kiểm tra lưới gồm query và 8 candidates cho mỗi query, tổng cộng 18 axes với 2 query.
- Script hiển thị label query/candidate và lưu hình minh họa ra file PNG.
- Đã chạy test visualization: `1/1` test đạt.
- Phần kiểm thử dùng dữ liệu tổng hợp; việc tạo ảnh từ CIFAR-10 thật còn phụ thuộc dữ liệu CIFAR-10 đã tải đầy đủ.

## Nghiệm thu Người 3 - Tuần 1

- **Gói 1:** Viết script kiểm tra môi trường `scripts/check_env.py`, checklist nghiệm thu hệ thống và cẩm nang xử lý 5 lỗi thường gặp (PowerShell policy, ModuleNotFoundError, charmap UTF-8 console, CUDA fallback, DataLoader multiprocessing).
- **Gói 2:** Hoàn thành review toàn diện API DataLoader (`build_cifar10`, `ContextDataset`, `cifar10_transforms`), xác nhận cấu trúc output dictionary và xây dựng code ví dụ sử dụng (Quickstart Recipe).
- **Gói 3:** Review thuật toán chia phân tầng `stratified_split`, xây dựng bảng thống kê chi tiết 10 lớp trên 50.000 mẫu (Query 36.000, Candidate Pool 9.000, Val 5.000, Test 10.000), xác minh toán học không giao nhau giữa các tập (0% rò rỉ dữ liệu).
- **Gói 4:** Viết script `scripts/verify_tacs_pipeline.py`, ghép thành công `ContextDataset` với `DataLoader` và đưa 1 batch qua `TACSModel`; kiểm tra tensor shape, one-hot selection weights qua Gumbel-Softmax, tính `TACSLoss` và backward gradients về Selector đạt 6/6 tham số.
- **Gói 5:** Viết script `scripts/inspect_20_queries.py`, thực hiện kiểm tra trực quan chi tiết 20 query (160 ảnh candidates), lưu ảnh minh họa tại `docs/visual_inspection_20_queries.png`. Xác nhận 0/20 query bị rò rỉ dữ liệu, 100% ảnh đạt dải giá trị chuẩn `[0.00, 1.00]`.
- **Gói 6:** Xuất bản tài liệu tổng hợp trung tâm `docs/BAO_CAO_TIEN_DO_TUAN_1.md` báo cáo toàn diện kết quả kỹ thuật của cả 3 thành viên và thiết lập quy chuẩn báo cáo tuần của dự án.

## Việc tiếp theo

1. Cả nhóm nghiệm thu chéo toàn bộ kết quả Tuần 1 (trực quan hóa và audit 20 query).
2. Bước sang Tuần 2 (Milestone M2) theo kế hoạch phân công chi tiết tại `WEEK_2_ASSIGNMENT.md`.
3. Xây dựng hạ tầng thí nghiệm chung, sau đó lần lượt triển khai và bàn giao 3 baseline: No-context → Random-context → Similarity-context.

