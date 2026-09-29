# TACS Improvement Project

Tái hiện và cải tiến **Task-Aligned Context Selection (TACS)** cho bài toán phân loại ảnh CIFAR-10.

TACS học cách chọn ảnh tham chiếu hữu ích cho ảnh truy vấn. Ảnh được chọn không nhất thiết giống ảnh truy vấn nhất, mà phải giúp mô hình downstream phân loại chính xác hơn.

## Mục tiêu

1. Xây dựng baseline không dùng context.
2. So sánh random context và similarity context.
3. Tái hiện TACS bằng Gumbel-Softmax và policy gradient.
4. Cải tiến tốc độ truy xuất bằng phương pháp two-stage.
5. Đánh giá bằng accuracy, tốc độ và bộ nhớ.

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Windows, Linux hoặc macOS.
- CUDA không bắt buộc; chương trình tự dùng CPU nếu không có GPU.

## Cài đặt

Tạo và kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài project và dependency:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Thiết lập đường dẫn source:

```powershell
$env:PYTHONPATH="src"
```

## Tải CIFAR-10

Chỉ cần chạy một lần:

```powershell
python -c "from tacs.data import build_cifar10; build_cifar10('data', 'data/splits/cifar10_seed42.json', download=True)"
```

Dữ liệu được chia cố định bằng seed 42:

- 36.000 ảnh query train.
- 9.000 ảnh candidate pool.
- 5.000 ảnh validation.
- 10.000 ảnh test.

## Chạy kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Hiện có 7 unit test kiểm tra shape, chia dữ liệu, candidate sampling, forward pass và gradient.

## Chạy smoke training

```powershell
python -m tacs.train_smoke --steps 5
```

Smoke training dùng dữ liệu giả để kiểm tra nhanh model và loss, không phải kết quả huấn luyện chính thức.

## Benchmark DataLoader

```powershell
python -m tacs.bench_data
```

Có thể thay đổi cấu hình:

```powershell
python -m tacs.bench_data --batch-size 64 --batches 50 --workers 0 --device auto
```

Benchmark báo cáo số query mỗi giây và RAM sử dụng.

Sau khi chạy xong:

```powershell
Remove-Item Env:PYTHONPATH
```

## Cấu trúc chính

- `src/tacs/model.py`: Selector và mạng phân loại.
- `src/tacs/losses.py`: task loss và policy loss.
- `src/tacs/data.py`: CIFAR-10, split và candidate sampling.
- `src/tacs/device.py`: thiết bị và seed.
- `src/tacs/bench_data.py`: benchmark pipeline dữ liệu.
- `tests/`: unit test.
- `docs/`: báo cáo và tài liệu quản lý dự án.

## Tài liệu

- `docs/PROJECT_PLAN.md`: kế hoạch tổng thể.
- `docs/WEEK_1_ASSIGNMENT.md`: phân công tuần 1.
- `docs/WEEK_2_ASSIGNMENT.md`: phân công tuần 2 và chuỗi bàn giao baseline.
- `docs/REPORT.md`: báo cáo kỹ thuật.
- `docs/PROGRESS.md`: tiến độ dự án.
- `docs/DECISIONS.md`: quyết định kỹ thuật.
- `docs/BAO_CAO_TIEN_DO_TUAN_1.md`: báo cáo tiến độ và nghiệm thu toàn diện Tuần 1.
- `docs/BAO_CAO_TIEN_DO_TUAN_2.md`: báo cáo tiến độ và kết quả thực nghiệm 3 baseline Tuần 2.
