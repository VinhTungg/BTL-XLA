# TACS: tái hiện, thực nghiệm và demo

Tái hiện cơ chế **Task-Aligned Context Selection** cho phân loại ảnh và phát triển
truy xuất context hai giai đoạn. Phạm vi hiện tại: **CIFAR-100 và Caltech-101**.
Ứng dụng demo dùng dữ liệu/checkpoint thật, suy luận offline, không cần train lại.

## Bắt đầu từ đâu?

| Công việc | Tài liệu |
|---|---|
| Chạy demo Caltech-101 với ảnh nguồn rõ hơn | [Hướng dẫn Caltech](docs/CALTECH101_DEMO_GUIDE.md) |
| Chạy demo CIFAR-100 | [Hướng dẫn showcase](docs/SHOWCASE_GUIDE.md) |
| Dựng slide và luyện nói | [Kịch bản 15 slide, demo và hỏi đáp](docs/SLIDE_SCRIPT.md) |
| Xem kết quả Caltech đã xác minh | [Kết quả full100](docs/CALTECH101_FULL100_RESULTS.md) |
| Tìm toàn bộ báo cáo, pilot và tài liệu lịch sử | [Mục lục tài liệu](docs/README.md) |
| Xem phạm vi dọn project và các file được giữ | [Biên bản cleanup](docs/PROJECT_CLEANUP.md) |

## Chạy demo trên máy đã có môi trường

Từ thư mục gốc repository, mở PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101
```

CIFAR-100:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset cifar100
```

Chờ trạng thái **Sẵn sàng**, chọn ảnh rồi bấm **So sánh dự đoán**.
Caltech có ví dụ nhanh tại validation position 0, 178 và 169.
Ứng dụng hiển thị query, context, top-5 và cho phép lưu JSON bằng chứng với tên mới.

Không chỉ clone source rồi mặc định checkpoint đã có sẵn. Demo cần:

- Môi trường `.venv` đã cài project và thư viện.
- Dữ liệu thật trong `data/`, bao gồm split đúng với lần train.
- Các checkpoint, manifest và frozen reference trong `artifacts/` theo hướng dẫn từng dataset.
- Cache demo hiện có được giữ để tránh mã hóa lại pool trước buổi bảo vệ.

Thiếu hoặc không khớp artifact phải báo lỗi, không thay bằng model chưa train.
Ảnh ngoài các lớp đã học vẫn bị ép chọn một nhãn; demo chưa phát hiện unknown.
Ảnh riêng không có nhãn thật thì không tự chấm đúng/sai.

## Trạng thái kết quả

| Thí nghiệm | Ngân sách và tập đánh giá | Kết quả được phép báo cáo |
|---|---|---|
| CIFAR-100 Two-Stage role_v2, seed 44 | Pilot 20 epoch, validation | Microbatch 87,92%; effective-batch 87,80% |
| Caltech-101 No-context, seed 44 | 100 epoch, best validation | 95,64%, best epoch 21 |
| Caltech-101 Two-Stage role_v2, seed 44 | 100 epoch, best validation | 94,26%, best epoch 54 |

Caltech dùng protocol riêng, chưa đánh giá test. Pilot CIFAR mới cũng chưa test.
Không trộn với các kết quả test lịch sử của `legacy_v1`, không gọi đây là tái hiện
toàn bộ bài báo và không tuyên bố Two-Stage đã tăng accuracy so với baseline độc lập.
Trong demo, **bỏ context trên cùng weights** khác với **No-context train riêng**.

## Cài môi trường trên máy mới

Yêu cầu Python theo `pyproject.toml` là 3.10 trở lên. Chỉ tạo `.venv` nếu chưa có:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Project không định nghĩa extra `dev`. Nếu cần tiện ích vẽ biểu đồ:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[viz]"
```

Kiểm tra Python, Torch và CUDA:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_env.py
```

Lệnh cài môi trường không tải sẵn các artifact thí nghiệm hoặc dữ liệu demo.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101 --check-ui
```

`--check-ui` chỉ tạo/hủy widget, không kiểm tra trực quan hay suy luận checkpoint.
Hai integration test dữ liệu thật mặc định được skip; chỉ bật
`TACS_RUN_REAL_CIFAR_TESTS=1` khi chủ động kiểm thử với dữ liệu local phù hợp.
Unit test có fixture tổng hợp để kiểm tra kỹ thuật, không dùng làm số liệu thực nghiệm.

## Huấn luyện và nghiên cứu tiếp

Cleanup không thay model, loss, split, recipe hay checkpoint. Đọc đúng tài liệu
trước khi mở thêm lượt train:

- [Caltech trên Colab](docs/CALTECH101_COLAB_GUIDE.md).
- [Pilot microbatch/effective-batch](docs/EFFECTIVE_BATCH_PILOT.md).
- [Bản sửa vai trò query/context](docs/CONTEXT_ROLE_FIX.md).
- [Kế hoạch tái hiện và phát triển](docs/ROADMAP_REPRODUCTION_TO_DEMO.md).
  Đây là kế hoạch có mốc lịch sử, không phải danh sách tất cả mục đã hoàn thành.

Xem CLI mà không train:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.train --help
.\.venv\Scripts\python.exe -X utf8 -m tacs.multiseed --plan-only
```

Trainer CIFAR còn hỗ trợ CIFAR-10/tiny để tương thích và kiểm thử. Khi train
CIFAR-100 phải chỉ định dataset/backbone phù hợp, không dựa vào mặc định cũ.
Run mới dùng `role_v2`; checkpoint legacy không tự được nâng cấp sang version mới.
Candidate bank, proxy gradient và tích lũy gradient là các lựa chọn triển khai
cần khai báo khi đối chiếu với paper.

## Cấu trúc project

```text
src/tacs/       Model, loss, dữ liệu, trainer, audit và demo
scripts/        Đóng gói cloud, thu hồi kết quả, kiểm tra và launcher
tests/          Unit test và integration test opt-in
notebooks/      Notebook Colab đã version hóa
docs/           Hướng dẫn hiện hành, kết quả đã xác minh và tài liệu lịch sử
data/           Dataset và split thật, không commit Git
artifacts/      Checkpoint, bundle, archive và bằng chứng, không commit Git
outputs/        Các run đầu ra được giữ để truy vết
cifar100/       Checkpoint legacy seed 44 được giữ để đối chiếu
translated/     Bản dịch bài báo
```

Không xóa `artifacts/`, `outputs/`, `cifar100/`, dữ liệu hoặc Git objects chỉ vì
chúng có dung lượng lớn. Đây không phải cache dùng một lần. Xem biên bản cleanup
để biết nhóm có thể xóa/tạo lại và nhóm cần duyệt riêng.
