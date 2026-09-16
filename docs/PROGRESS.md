# Tiến độ

Cập nhật: 2026-09-16

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Đọc và phân tích paper | Xong | Bản Anh là nguồn chuẩn |
| Kiểm kê source | Xong | Không có code TACS chính thức trong workspace |
| Phân tích dự án tham khảo | Xong | CLIP + Weaviate; không tái sử dụng cho lõi training |
| Skeleton TACS | Xong | Selector, classifier, hybrid loss |
| Unit test CPU | Xong | 8/8 test pass; kiểm tra split, sampling, visualization, forward và gradient |
| Kiểm tra môi trường máy số 2 | Xong một phần | Python 3.13.7, PyTorch 2.10.0+cpu; chưa có torchvision, CUDA hoặc GPU |
| Giải thích TACS trong báo cáo | Xong | Có ví dụ, luồng xử lý, Gumbel và reward |
| Dataset + baseline | Chưa làm | M1 |
| Reproduction | Chưa làm | M2 |
| Cải tiến two-stage | Chưa làm | M3 |

## Nghiệm thu Người 1 - Tuần 1

**Trạng thái: Đạt phần xây dựng skeleton và pipeline; chưa nghiệm thu đầy đủ dữ liệu CIFAR-10 thật.**

- Đã xây dựng skeleton TACS gồm selector, classifier và hybrid loss.
- Đã thêm cơ chế chọn thiết bị tự động: ưu tiên CUDA khi khả dụng và fallback về CPU.
- Đã xây dựng module CIFAR-10 với transform train/evaluation, chia query-training, candidate pool, validation và test.
- Đã triển khai candidate sampler tái lập theo `seed=42`, mặc định `num_candidates=8`.
- Đã tích hợp DataLoader với `TACSModel` và kiểm tra forward trên batch dữ liệu tổng hợp.
- Toàn bộ unit test hiện tại đạt `8/8`; smoke training 5 bước chạy thành công trên CPU.
- README đã có lệnh chạy test, smoke training và kiểm tra Python/PyTorch/CUDA/GPU.
- Chưa hoàn tất nghiệm thu dữ liệu thật vì máy chưa có `torchvision`, CUDA hoặc GPU; chưa tải CIFAR-10 và benchmark DataLoader trên dữ liệu thật.

## Nghiệm thu Người 2 - Tuần 1

**Trạng thái: Đạt phần kiểm thử và kiểm tra môi trường; chưa nghiệm thu đầy đủ pipeline CIFAR-10 thật.**

- Đã kiểm tra project trên máy số 2 với Python `3.13.7` và PyTorch `2.10.0+cpu`.
- Đã xác nhận CUDA không khả dụng và máy không phát hiện GPU; project vẫn chạy được trên CPU.
- Đã chạy toàn bộ unit test: `8/8` test đạt, gồm split, sampling, visualization, forward và gradient.
- Đã kiểm tra batch tích hợp với TACS có shape query `[B, 3, 32, 32]`, candidates `[B, 8, 3, 32, 32]` và target `[B]` trên dữ liệu tổng hợp.
- Đã xác nhận candidate sampling tái lập theo seed và query không trùng candidate.
- Đã kiểm tra script trực quan hóa query và candidates qua `tests/test_visualize.py`.
- Smoke training 5 bước chạy thành công trên CPU.
- Chưa hoàn tất nghiệm thu CIFAR-10 thật vì máy chưa có `torchvision`; chưa thể tải dataset và tạo ảnh trực quan từ dữ liệu thật.

## Đối chiếu nhiệm vụ Người 2 - Tuần 1

| Gói | Nhiệm vụ Người 2 | Trạng thái hiện tại | Bằng chứng / phần còn thiếu |
|---|---|---|---|
| 1 | Cài project và kiểm tra Python, PyTorch, CUDA, GPU | Đạt một phần | Máy số 2 chạy test và smoke training trên CPU; Python 3.13.7, PyTorch 2.10.0+cpu; chưa có `torchvision`, CUDA hoặc GPU |
| 2 | Viết test kích thước ảnh, label, batch và augmentation | Đạt một phần | Có test batch/shape, label và pipeline TACS trong `tests/test_data.py`; chưa chạy được CIFAR-10 thật do thiếu `torchvision` |
| 3 | Viết test chống trùng và rò rỉ giữa các tập | Đạt một phần | `SplitIndices.validate` và test split tái lập/cân bằng đã có; chưa có test riêng kiểm tra test set CIFAR-10 không giao với train vì chưa tải dataset thật |
| 4 | Test index, shape, seed, query trùng candidate và batch cuối | Đạt | `tests/test_data.py` kiểm tra seed, index, shape, query-candidate và forward TACS bằng dữ liệu tổng hợp |
| 5 | Viết script hiển thị query, candidates, label và index | Đạt một phần | Có `src/tacs/visualize_batch.py` và test tạo grid; chưa tạo ảnh từ CIFAR-10 thật |

**Kết luận:** tiến độ của Người 2 đã đạt phần khung code và kiểm thử tổng hợp, nhưng chưa nghiệm thu hoàn toàn Tuần 1. Điều kiện còn thiếu là bổ sung `torchvision`, tải CIFAR-10, chạy lại loader/split/visualization trên dữ liệu thật và xác nhận không rò rỉ với test set.

## Việc tiếp theo

1. Người 3 kiểm tra trực quan ít nhất 20 query và ghi nhận lỗi.
2. Cả nhóm review chéo, sau đó chạy no-context baseline.
