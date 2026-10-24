# TACS Improvement Project

Tái hiện và cải tiến **Task-Aligned Context Selection (TACS)** cho bài toán phân loại ảnh CIFAR-10 và CIFAR-100.

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

## Tải CIFAR-10 hoặc CIFAR-100

Chỉ cần chạy một lần:

```powershell
python -c "from tacs.data import build_cifar10; build_cifar10('data', 'data/splits/cifar10_seed42.json', download=True)"
python -c "from tacs.data import build_cifar100; build_cifar100('data', 'data/splits/cifar100_seed42.json', download=True)"
```

Cả hai bộ dữ liệu có 50.000 ảnh train và được chia cố định bằng seed 42:

- 36.000 ảnh query train.
- 9.000 ảnh candidate pool.
- 5.000 ảnh validation.
- 10.000 ảnh test.

## Chạy kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Bộ test bao phủ shape, chia dữ liệu, candidate sampling, forward pass,
gradient, retrieval, trainer và kiểm tra tích hợp CIFAR-10. Integration test dùng
dữ liệu thật được tắt mặc định và chỉ chạy khi đặt
`TACS_RUN_REAL_CIFAR_TESTS=1`.

## Chế độ dữ liệu của trainer

Trainer mặc định dùng CIFAR-10 thật; thêm `--dataset cifar100` để
chạy CIFAR-100 thật. `num_classes`, file split, output directory và standard
encoder được chọn theo dataset. `--dry-run` chỉ giới hạn số mẫu và số
epoch; nó vẫn đọc dữ liệu thật. Trainer không còn chế độ dữ liệu mock.

Ví dụ smoke run trên CIFAR-100:

```powershell
python -m tacs.train --dataset cifar100 --baseline no_context --dry-run --device cpu
```

## Backbone DINOv3 ViT-S/16

Chế độ gần thiết lập paper dùng backbone DINOv3 ViT-S/16 qua `timm`:

```powershell
python -m tacs.train --dataset cifar100 --backbone dinov3_vits16 --baseline no_context --device cuda
```

Profile này tự động dùng embedding 384 chiều, ảnh 256×256, ImageNet
normalization và pretrained weights `vit_small_patch16_dinov3.lvd1689m`.
Lần chạy đầu cần Internet để tải weights và người dùng phải tuân thủ
DINOv3 License. Thêm `--no-pretrained-backbone` chỉ để kiểm tra kỹ thuật;
không dùng kết quả đó làm thí nghiệm báo cáo.

Commit này thay backbone feature encoder. Ghép patch token query/context qua
Task Network theo paper được triển khai ở nhiệm vụ tiếp theo.

Mỗi `metrics.json` ghi rõ `data.source`, kích thước các tập và hash của
file split để xác minh kết quả được tạo từ đúng dữ liệu.

## Standard encoder và cache retrieval

Phải train `no_context` trước khi chạy `similarity` hoặc `two_stage`.
Hai mode retrieval chỉ chấp nhận `best_encoder.pt` có metadata xác nhận
baseline `no_context`, đúng dataset và đúng `embedding_dim`. Checkpoint cũ không có
provenance sẽ bị từ chối và cần train lại.

Cache candidate embedding được kiểm tra bằng hash của split, danh sách
candidate, standard encoder và embedding dimension. Cache cũ, sai kích thước
hoặc không khớp checkpoint sẽ được tự động tái tạo.

## TACS Full và Two-Stage

- `--baseline tacs` chấm điểm **toàn bộ candidate pool** cho mỗi query. Candidate
  được mã hóa thành một bank dùng chung, làm mới mỗi epoch, nên DataLoader không
  nhân bản 9.000 ảnh cho từng query. Bank được detach để giới hạn bộ nhớ; encoder
  vẫn nhận gradient qua nhánh query.
- `--baseline two_stage --topk-candidates 32` quét toàn pool bằng cosine từ
  standard encoder, sau đó chỉ đưa top-32 vào TACS selector. Dataset chỉ
  chuyển các vị trí top-k; trainer gom embedding từ một candidate bank dùng chung,
  được làm mới mỗi train/evaluation epoch, thay vì mã hóa lại `B*k` ảnh
  candidate trong từng batch.
- `--topk-candidates` chỉ tác động đến Two-Stage, không thu nhỏ TACS Full.
- Hai mode dùng cùng kiến trúc và cùng khởi tạo model theo seed. Có thể đặt
  `--selector-temperature` và `--policy-weight`. Temperature điều khiển gradient
  của Gumbel-Softmax relaxation; validation/test luôn chọn argmax xác định.
- Để so sánh cùng cơ chế gradient và giữ bộ nhớ ổn định, candidate embedding ở
  cả hai mode đều detach; selector encoder vẫn học qua query embedding.

`metrics.json` ghi strategy, số candidate bị retrieval quét, số candidate được
selector đánh giá, thời gian setup/retrieval, latency/query end-to-end và
throughput validation để việc so sánh không nhầm giữa Full và top-k.

## Benchmark 5 mô hình

Trainer ghi metrics schema v2 với peak RSS, peak CUDA allocated/reserved, thời
gian setup, latency end-to-end và cấu hình thí nghiệm. Sau khi train đủ năm mô
hình với `--eval-test`, chạy:

```powershell
python scripts/analyze_all_models.py --outputs-root outputs --output-dir docs
```

Script kiểm tra cả năm lần chạy dùng CIFAR-10 thật, đúng kích thước đầy đủ, cùng
split, seed, epoch, batch size và embedding dimension. Artifact dry-run, thiếu
test result hoặc schema cũ sẽ bị từ chối. Kết quả gồm:

- `docs/all_models_benchmark.csv`
- `docs/all_models_benchmark.json`
- `docs/all_models_benchmark.md`
- `docs/all_models_comparison.png`

Chỉ để kiểm tra pipeline bằng dry-run, có thể thêm `--allow-incomplete`; dữ liệu
này không được dùng làm kết quả báo cáo.

## Chạy smoke training

```powershell
python -m tacs.train_smoke --steps 5
```

Smoke training dùng một vài batch CIFAR-10 thật để kiểm tra nhanh model
và loss; không phải kết quả huấn luyện chính thức.

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
- `src/tacs/data.py`: CIFAR-10/CIFAR-100, split và candidate sampling.
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
