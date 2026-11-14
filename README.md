# TACS Improvement Project

Tái hiện và cải tiến **Task-Aligned Context Selection (TACS)** cho bài toán phân loại ảnh CIFAR-10 và CIFAR-100.

TACS học cách chọn ảnh tham chiếu hữu ích cho ảnh truy vấn. Ảnh được chọn không nhất thiết giống ảnh truy vấn nhất, mà phải giúp mô hình downstream phân loại chính xác hơn.

**Cập nhật 04/11/2026:** DINOv3 mặc định dùng `role_v2` để phân biệt query/context.
Run mới lưu trong namespace `role_v2`; checkpoint cũ phải chọn `--task-network-version legacy_v1`.
Xem [nguyên nhân, kiểm thử checkpoint và hướng dẫn pilot](docs/CONTEXT_ROLE_FIX.md).
Bản sửa kiến trúc chưa phải bằng chứng tăng accuracy sau huấn luyện.

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

Trainer dùng AdamW và `CosineAnnealingLR` cho mọi mô hình để việc so
sánh công bằng; có thể thêm linear warm-up bằng `--warmup-epochs`. Profile DINOv3 mặc định 100 epochs theo paper; profile
`tiny` giữ 5 epochs cho kiểm tra nhanh. Có thể ghi đè bằng `--epochs`,
`--lr` và `--min-lr`. Khi VRAM không đủ cho batch paper, dùng
`--batch-size 4 --gradient-accumulation-steps 16` để giữ batch hiệu dụng
64. Mỗi epoch lưu `learning_rate` và
`next_learning_rate` trong `metrics.json`; checkpoint lưu cả optimizer và
scheduler state.

Trên CUDA, trainer mặc định dùng automatic mixed precision FP16 để giảm VRAM
và tăng throughput nhưng không thay đổi dataset hay kiến trúc. Có thể tắt bằng
`--no-amp` để debug số học FP32. DataLoader tự bật pinned memory và tensor được
chuyển lên GPU theo chế độ non-blocking.

Sau mỗi epoch, `checkpoint_latest.pt` lưu model, optimizer, scheduler, AMP
scaler, lịch sử và RNG state. Khi chạy lại cùng lệnh, trainer tự tiếp tục từ
checkpoint này; với run cũ chưa có checkpoint latest, trainer dùng
`checkpoint_best.pt` tương thích làm fallback. Dùng `--no-resume` để chủ động
train lại từ đầu.

## Task Network ghép patch token

Nhánh phân loại TACS bám theo Sec. 4.2 của paper: query và context được
patch-embed riêng, sau đó nối thành một chuỗi token trước các Transformer
block. Logit được sinh từ CLS token chung. Nhánh reward no-context chỉ
chứa patch token của query, không dùng context rỗng giả.

Profile `tiny` dùng Task Network Transformer nhỏ để unit test CPU. Profile
`dinov3_vits16` dùng một ViT-S/16 DINOv3 cho Selector và một ViT-S/16
DINOv3 riêng cho Downstream Task Network, đúng cấu trúc hai module của TACS.

Ba baseline cũng dùng chính Task Network này: `no_context` chỉ truyền query
patches, còn `random` và `similarity` truyền cặp query/context patches. Encoder
dùng để lập cache similarity được đóng băng. Như vậy các mô hình
khác nhau ở cách chọn context thay vì khác cấu trúc classifier.

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

- Utility score trong TACS Selector dùng tích vô hướng trực tiếp
  `z_queryᵀ z_candidate` theo Eq. (1), không chuẩn hóa L2. Cosine similarity
  chỉ được dùng cho baseline `similarity` và tầng lọc thô của
  `two_stage`.
- `--baseline tacs` chấm điểm **toàn bộ candidate pool** cho mỗi query. Candidate
  được mã hóa thành một bank dùng chung, làm mới mỗi epoch, nên DataLoader không
  nhân bản 9.000 ảnh cho từng query. Bank được detach để giới hạn bộ nhớ; encoder
  vẫn nhận gradient qua nhánh query. Sau lựa chọn, trainer chỉ nạp ảnh
  context thắng cuộc vào Task Network. Proxy straight-through có giá trị forward
  bằng 0 giữ gradient từ task loss về utility score mà không cần đưa toàn
  bộ pool ảnh 256×256 lên GPU.
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

Trainer ghi metrics schema v6 với peak RSS, peak CUDA allocated/reserved, thời
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

## Chạy thí nghiệm 3 seed

Runner nhiều seed mặc định chạy đủ năm mô hình trên CIFAR-100 với DINOv3
ViT-S/16, 100 epochs và các seed `42 43 44`. Mỗi lần chạy nằm trong một process
riêng để giải phóng VRAM trước mô hình kế tiếp. `no_context` luôn được chạy trước
`similarity` và `two_stage` để tạo standard encoder đúng seed.

Kiểm tra trước toàn bộ lệnh và đường dẫn mà không train:

```powershell
python -m tacs.multiseed --plan-only
```

Smoke test nhanh pipeline nhiều seed bằng cấu hình nhỏ:

```powershell
python -m tacs.multiseed --dataset cifar10 --backbone tiny --dry-run --device cpu
```

Chạy thí nghiệm chính thức:

```powershell
python -m tacs.multiseed --device cuda --num-workers 4
```

Kết quả mỗi seed được lưu tại
`outputs/cifar100/dinov3_vits16/seed_<seed>/<baseline>/`. Runner tự bỏ qua run
đã hoàn chỉnh và tiếp tục từ run còn thiếu; thêm `--force` khi thật sự muốn chạy
lại từ đầu. Một run đang dở được tiếp tục từ checkpoint gần nhất; `--force`
đồng thời tắt resume. Cả ba seed dùng chung split mặc định
`data/splits/cifar100_seed42.json`, do
đó độ lệch chuẩn phản ánh biến thiên huấn luyện thay vì thay đổi tập dữ liệu.

Ví dụ phù hợp GPU laptop/Colab T4, dùng micro-batch 4 nhưng giữ batch hiệu dụng 64:

```powershell
python -m tacs.multiseed --device cuda --batch-size 4 --gradient-accumulation-steps 16 --lr 1e-4 --warmup-epochs 5 --num-workers 2
```

Sau khi hoàn tất, mean và sample standard deviation (`ddof=1`) được xuất ra:

- `outputs/cifar100/dinov3_vits16/multiseed_summary.json`
- `outputs/cifar100/dinov3_vits16/multiseed_summary.csv`
- `outputs/cifar100/dinov3_vits16/multiseed_summary.md`

## Chạy trên Google Colab

Notebook `notebooks/TACS_Colab.ipynb` tự mount Google Drive, cài project, tải
CIFAR-100, kiểm tra GPU và chạy runner theo từng seed. Output v2 được ghi vào
`MyDrive/TACS_outputs_v2`, nên khi Colab ngắt phiên chỉ cần kết nối lại và chạy lại
notebook để resume.

Tạo bundle nhẹ gồm source và split CIFAR-100, không kèm checkpoint pilot cũ:

```powershell
.\scripts\package_colab.ps1 -WithoutCheckpoint
```

File `artifacts/TACS_Colab_Bundle.zip` sinh ra không được commit. Upload file đó
vào thư mục gốc Google Drive, mở notebook trên Colab, chọn runtime GPU và chạy
các cell theo thứ tự. Notebook mặc định dùng batch vật lý 4, tích lũy 16 bước
(batch hiệu dụng 64), learning rate `1e-4`, warm-up 5 epoch và quản lý cả ba
seed `42 43 44` trong cùng một lệnh. Thư mục v2 tách biệt hoàn toàn với kết quả
pilot cũ. Khi Colab ngắt, chạy lại notebook; runner
tự bỏ qua run hoàn chỉnh và resume run còn dở. Chỉ coi thí nghiệm là xong khi
cell xác minh cuối cùng xác nhận đủ `metrics.json`, kết quả test và ba file
`multiseed_summary.json`, `multiseed_summary.csv`, `multiseed_summary.md`.

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

- [Demo offline CIFAR-100](docs/SHOWCASE_GUIDE.md): chạy
  `.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase`, chọn checkpoint đã audit,
  xem query/context/top-5 và lưu bằng chứng JSON. Lệnh này chỉ inference CIFAR-100.
  [Caltech-101 cho demo ảnh rõ](docs/CALTECH101_DEMO_PLAN.md) là nhánh bổ sung
  đã có loader/trainer và [notebook Colab tự chạy](docs/CALTECH101_COLAB_GUIDE.md).
  Cần train checkpoint riêng rồi mới tích hợp Caltech vào GUI; chưa có trong demo hiện tại.

- [Kế hoạch tiếp theo từ tái hiện đến demo và bảo vệ](docs/ROADMAP_REPRODUCTION_TO_DEMO.md):
  kế hoạch hiện hành ngày 30/10/2026 cho CIFAR-100/DINOv3, gồm audit paper,
  protocol, train nhiều seed, cải tiến, ngân sách GPU, demo và phân công ba người.
- `docs/PROJECT_PLAN.md`: kế hoạch tổng thể.
- `docs/WEEK_1_ASSIGNMENT.md`: phân công tuần 1.
- `docs/WEEK_2_ASSIGNMENT.md`: phân công tuần 2 và chuỗi bàn giao baseline.
- `docs/REPORT.md`: báo cáo kỹ thuật.
- `docs/PROGRESS.md`: tiến độ dự án.
- `docs/DECISIONS.md`: quyết định kỹ thuật.
- `docs/CIFAR100_TACS_SEED42_RESULTS.md`: kết quả TACS 100 epoch trên CIFAR-100,
  manifest artifact và checksum checkpoint.
- `docs/BAO_CAO_TIEN_DO_TUAN_1.md`: báo cáo tiến độ và nghiệm thu toàn diện Tuần 1.
- `docs/BAO_CAO_TIEN_DO_TUAN_2.md`: báo cáo tiến độ và kết quả thực nghiệm 3 baseline Tuần 2.
