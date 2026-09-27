# BÁO CÁO TIẾN ĐỘ TUẦN 1
## Dự án: Tái hiện và Cải tiến TACS (Task-Aligned Context Selection) trên CIFAR-10
**Học phần:** Xử lý ảnh / Thị giác máy tính (BTL-XLA)  
**Thời gian thực hiện:** Tuần 1  
**Nhân sự tham gia:** Toàn bộ 3 thành viên (Người 1, Người 2, Người 3)  
**Trạng thái nghiệm thu:** **HOÀN THÀNH 100% CÁC MỤC TIÊU NỀN MÓNG** (Đủ điều kiện chuyển giao sang Tuần 2)

---

## 📌 QUY CHUẨN BÁO CÁO DÀNH CHO CÁC THÀNH VIÊN VÀ TRỢ LÝ AI (AI GUIDELINES)

> [!IMPORTANT]
> **THỐNG NHẤT QUY TẮC BÁO CÁO TOÀN DỰ ÁN:**
> 1. **Bãi bỏ hoàn toàn báo cáo cá nhân riêng lẻ** (Không sử dụng các file dạng `BAO_CAO_THANH_VIEN_X.md`).
> 2. **Tất cả các tuần đều dùng báo cáo tiến độ tuần chung của nhóm**, đặt tên theo quy chuẩn:
>    - `docs/BAO_CAO_TIEN_DO_TUAN_1.md` (Tuần 1 - Môi trường & Pipeline dữ liệu)
>    - `docs/BAO_CAO_TIEN_DO_TUAN_2.md` (Tuần 2 - Huấn luyện 3 Baseline M2)
>    - `docs/BAO_CAO_TIEN_DO_TUAN_3.md` (Tuần 3 - TACS Full & Đánh giá)
>    - `docs/BAO_CAO_TIEN_DO_TUAN_4.md` (Tuần 4 - Two-Stage TACS & Báo cáo cuối kỳ)
> 3. **Chỉ dẫn cho AI Assistant khi hỗ trợ các thành viên ở các tuần sau:**
>    - Khi thành viên yêu cầu viết/cập nhật báo cáo cho tuần $X$, AI **BẮT BUỘC** đọc file phân công tương ứng `docs/WEEK_{X}_ASSIGNMENT.md` và file `docs/PROGRESS.md`.
>    - Báo cáo phải phản ánh đầy đủ công việc của **cả 3 thành viên** theo đúng chuỗi bàn giao: *Người xây dựng → Người kiểm thử/tích hợp → Người chạy lại/nghiệm thu*.
>    - Giữ cấu trúc chuẩn: Tổng quan → Bảng phân công & đóng góp từng người → Chi tiết kỹ thuật & mã nguồn → Số liệu thực nghiệm & Test suite → Cẩm nang xử lý lỗi → Đánh giá chuyển giao (Gate Check).
>    - Mọi kết luận kỹ thuật phải đi kèm số liệu định lượng, công thức toán học hoặc đường dẫn file dẫn chứng (`file:///...`).

---

# MỤC LỤC BÁO CÁO TUẦN 1
1. [Mục tiêu và Yêu cầu Tuần 1](#1-mục-tiêu-và-yêu-cầu-tuần-1)
2. [Bảng tổng hợp đóng góp của 3 thành viên](#2-bảng-tổng-hợp-đóng-góp-của-3-thành-viên)
3. [Chi tiết kết quả kỹ thuật Tuần 1](#3-chi-tiết-kết-quả-kỹ-thuật-tuần-1)
   - [3.1. Chuẩn hóa môi trường & Cẩm nang khắc phục sự cố (Gói 1)](#31-chuẩn-hóa-môi-trường--cẩm-nang-khắc-phục-sự-cố-gói-1)
   - [3.2. Pipeline CIFAR-10 DataLoader & Quickstart Recipe (Gói 2)](#32-pipeline-cifar-10-dataloader--quickstart-recipe-gói-2)
   - [3.3. Phân tầng dữ liệu & Chứng minh Không Rò Rỉ (Gói 3)](#33-phân-tầng-dữ-liệu--chứng-minh-không-rò-rỉ-gói-3)
   - [3.4. Candidate Sampler & Ghép nối TACSModel (Gói 4)](#34-candidate-sampler--ghép-nối-tacsmodel-gói-4)
   - [3.5. Trực quan hóa & Audit trực quan 20 Query (Gói 5)](#35-trực-quan-hóa--audit-trực-quan-20-query-gói-5)
   - [3.6. Benchmark hiệu năng DataLoader & Bộ nhớ (Gói 5 & 6)](#36-benchmark-hiệu-năng-dataloader--bộ-nhớ-gói-5--6)
4. [Kết quả Kiểm thử Hệ thống (Test Suite)](#4-kết-quả-kiểm-thử-hệ-thống-test-suite)
5. [Đánh giá Nghiệm thu & Điều kiện chuyển tiếp Tuần 2](#5-đánh-giá-nghiệm-thu--điều-kiện-chuyển-tiếp-tuần-2)

---

## 1. Mục tiêu và Yêu cầu Tuần 1

Căn cứ theo [WEEK_1_ASSIGNMENT.md](file:///e:/BTL-XLA/docs/WEEK_1_ASSIGNMENT.md), chủ đề cốt lõi của Tuần 1 là: **"Chuẩn hóa môi trường và xây dựng pipeline dữ liệu CIFAR-10"**.

### Quy chuẩn dữ liệu CIFAR-10 được phê duyệt:
Tập dữ liệu CIFAR-10 gồm $60.000$ ảnh màu kích thước $32 \times 32$ thuộc 10 lớp đối tượng. Nhóm thiết lập cấu hình phân tách cố định bằng `seed=42`:
- **Query-Training:** $36.000$ ảnh ($72\%$ tập train gốc).
- **Candidate Pool:** $9.000$ ảnh ($18\%$ tập train gốc, `candidate_ratio=0.20` trên phần còn lại sau validation).
- **Validation:** $5.000$ ảnh ($10\%$ tập train gốc) dùng để tinh chỉnh siêu tham số và early stopping.
- **Test Set:** $10.000$ ảnh hoàn toàn độc lập, cách ly tuyệt đối khỏi quá trình huấn luyện và candidate pool.
- **Số lượng ứng viên lấy mẫu:** $N = 8$ candidates/query.

---

## 2. Bảng tổng hợp đóng góp của 3 thành viên

| Gói công việc | Người 1 (Lead / Core Pipeline) | Người 2 (Test & Trực quan hóa) | Người 3 (Tích hợp & Tài liệu) |
|---|---|---|---|
| **Gói 1: Môi trường** | Cập nhật dependencies, thiết lập cơ chế fallback linh hoạt CPU/GPU tại [device.py](file:///e:/BTL-XLA/src/tacs/device.py). | Chạy kiểm thử độc lập môi trường trên máy thứ 2. | Xây dựng script kiểm tra tự động [check_env.py](file:///e:/BTL-XLA/scripts/check_env.py) và bảng cẩm nang 6 lỗi E01–E06. |
| **Gói 2: DataLoader** | Viết module [data.py](file:///e:/BTL-XLA/src/tacs/data.py), hàm `build_cifar10` và logic `cifar10_transforms`. | Xây dựng các ca kiểm thử kích thước batch, shape và label. | Review kiến trúc API DataLoader, viết Quickstart Recipe mẫu cho các thành viên. |
| **Gói 3: Split & Pool** | Thuật toán `stratified_split`, cơ chế lưu/đọc index `SplitIndices.save/load`. | Viết unit test chống trùng lặp, kiểm tra phân bố đều các lớp. | Lập bảng thống kê 10 lớp trên 50.000 mẫu, chứng minh toán học No-Leakage ($I_q \cap I_c = \emptyset$). |
| **Gói 4: Sampler & TACS** | Xây dựng `ContextDataset`, lấy mẫu $N=8$ candidates ngẫu nhiên có seed thay đổi theo epoch. | Viết test kiểm tra tính tái lập (`reproducible`) và loại trừ query index khỏi candidate. | Viết script [verify_tacs_pipeline.py](file:///e:/BTL-XLA/scripts/verify_tacs_pipeline.py), kiểm tra forward/backward pass và gradient về 6/6 tham số Selector. |
| **Gói 5: Test & Trực quan** | Đo benchmark tốc độ nạp dữ liệu (426.33 query/s) và tiêu thụ RAM (773.31 MB). | Xây dựng module trực quan hóa [visualize_batch.py](file:///e:/BTL-XLA/src/tacs/visualize_batch.py) và unit test [test_visualize.py](file:///e:/BTL-XLA/tests/test_visualize.py). | Viết script [inspect_20_queries.py](file:///e:/BTL-XLA/scripts/inspect_20_queries.py), kiểm tra trực quan chi tiết 20 query và xuất ảnh minh họa [visual_inspection_20_queries.png](file:///e:/BTL-XLA/docs/visual_inspection_20_queries.png). |
| **Gói 6: Báo cáo & Nghiệm thu** | Xác minh cài đặt từ môi trường sạch (`.venv-clean`), hoàn thiện [README.md](file:///e:/BTL-XLA/README.md). | Chạy toàn bộ test suite, kiểm thử tương thích hiển thị. | Cập nhật [REPORT.md](file:///e:/BTL-XLA/docs/REPORT.md), [PROGRESS.md](file:///e:/BTL-XLA/docs/PROGRESS.md) và biên soạn Báo cáo Tiến độ Tuần 1. |

---

## 3. Chi tiết kết quả kỹ thuật Tuần 1

### 3.1. Chuẩn hóa môi trường & Cẩm nang khắc phục sự cố (Gói 1)
Nhóm đã triển khai script tự động kiểm tra tương thích [scripts/check_env.py](file:///e:/BTL-XLA/scripts/check_env.py). Môi trường nghiệm thu đạt chuẩn:
- **Python Runtime:** `3.13.x` / `3.11.x`.
- **PyTorch & Torchvision:** Phiên bản tương thích CPU/CUDA (`2.x`).
- **Thư viện phụ trợ:** NumPy, Matplotlib.
- Cơ chế `tacs.device.resolve_device('auto')` cho phép code chạy thông suốt trên cả máy có GPU CUDA và máy thuần CPU mà không cần sửa code thủ công.

#### Ma trận xử lý lỗi thường gặp (Troubleshooting Matrix):
1. **Lỗi E01 (PowerShell Execution Policy):** Chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` trước khi kích hoạt venv.
2. **Lỗi E02 (ModuleNotFoundError: 'tacs'):** Chạy `pip install -e .` hoặc `$env:PYTHONPATH="src"` trên PowerShell.
3. **Lỗi E03 (UnicodeEncodeError cp1252 trên Windows):** Gọi `configure_stdout()` từ `tacs.device` hoặc chạy lệnh PowerShell: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`.
4. **Lỗi E04 (CUDA unavailable):** Chỉ định cờ `--device auto` hoặc `--device cpu`.
5. **Lỗi E05 (DataLoader worker crash trên Windows):** Luôn bọc mã thực thi trong `if __name__ == '__main__':` và đặt `num_workers=0` trên Windows.
6. **Lỗi E06 (Mạng tải CIFAR-10 bị timeout):** Sử dụng split cache tại `data/splits/cifar10_seed42.json` hoặc nạp offline.

---

### 3.2. Pipeline CIFAR-10 DataLoader & Quickstart Recipe (Gói 2)
Thiết kế tiền xử lý trong [src/tacs/data.py](file:///e:/BTL-XLA/src/tacs/data.py) tuân thủ nghiêm ngặt nguyên lý thị giác máy tính:
- **Tập Query-Train:** Áp dụng Data Augmentation gồm `RandomCrop(size=32, padding=4)` và `RandomHorizontalFlip()`, sau đó chuẩn hóa theo giá trị chuẩn của CIFAR-10 (`mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`).
- **Candidate Pool, Validation & Test:** **Chỉ áp dụng chuẩn hóa (Normalize), tuyệt đối không dùng biến dạng ngẫu nhiên**. Điều này bảo toàn chất lượng ảnh tham chiếu làm ngữ cảnh, tránh gây nhiễu cho bộ chọn Selector.

#### Cấu trúc Dictionary trả về từ ContextDataset:
Mỗi batch dữ liệu cung cấp đầy đủ thông tin metadata truy vết:
```python
{
    "query": torch.Tensor,              # [B, 3, 32, 32] - Ảnh truy vấn
    "candidates": torch.Tensor,         # [B, N, 3, 32, 32] - Nhóm N ảnh ứng viên
    "target": torch.Tensor,             # [B] - Nhãn của query (0-9)
    "query_index": torch.Tensor,        # [B] - Index gốc trong tập CIFAR-10
    "candidate_indices": torch.Tensor,  # [B, N] - Index gốc của các candidates trong pool
    "candidate_targets": torch.Tensor   # [B, N] - Nhãn lớp của các candidates
}
```

#### Quickstart Recipe mẫu để sử dụng trong các Baseline:
```python
import torch
from torch.utils.data import DataLoader
from tacs.data import build_cifar10, ContextDataset

# 1. Khởi tạo 4 tập dữ liệu đã phân tầng
datasets = build_cifar10(root="data", split_file="data/splits/cifar10_seed42.json", download=False)

# 2. Bọc query và candidate vào ContextDataset
train_dataset = ContextDataset(
    query_dataset=datasets.query_train,
    candidate_dataset=datasets.candidate_pool,
    num_candidates=8,
    seed=42,
)

# 3. Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
batch = next(iter(train_loader))
print(f"Query shape: {batch['query'].shape}, Candidates shape: {batch['candidates'].shape}")
```

---

### 3.3. Phân tầng dữ liệu & Chứng minh Không Rò Rỉ (Gói 3)

#### Bảng phân bố chi tiết 10 lớp trên các tập dữ liệu:

| Class ID | Tên lớp CIFAR-10 | Query-Train (72%) | Candidate Pool (18%) | Validation (10%) | Test Set (Độc lập) | Tổng cộng |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Airplane | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 1 | Automobile | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 2 | Bird | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 3 | Cat | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 4 | Deer | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 5 | Dog | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 6 | Frog | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 7 | Horse | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 8 | Ship | 3.600 | 900 | 500 | 1.000 | 6.000 |
| 9 | Truck | 3.600 | 900 | 500 | 1.000 | 6.000 |
| **Tổng** | **10 lớp** | **36.000** | **9.000** | **5.000** | **10.000** | **60.000** |

#### Chứng minh toán học Không Rò Rỉ Dữ Liệu (No-Leakage Proof):
1. **Tính chất rời nhau tuyệt đối (Disjoint Sets):**
   $$\mathcal{I}_{\text{query}} \cap \mathcal{I}_{\text{candidate}} = \emptyset, \quad \mathcal{I}_{\text{query}} \cap \mathcal{I}_{\text{val}} = \emptyset, \quad \mathcal{I}_{\text{candidate}} \cap \mathcal{I}_{\text{val}} = \emptyset$$
   Đảm bảo không có bất kỳ mẫu nào vừa đóng vai trò huấn luyện vừa là context tham chiếu hay đánh giá xác thực.
2. **Tính bao phủ toàn vẹn (Full Coverage):**
   $$\mathcal{I}_{\text{query}} \cup \mathcal{I}_{\text{candidate}} \cup \mathcal{I}_{\text{val}} = \{0, 1, \dots, 49.999\}$$
   Độ dài hợp đúng bằng $50.000$ mẫu của tập train CIFAR-10 ban đầu.
3. **Cách ly tập Test:**
   Tập kiểm thử $10.000$ mẫu hoàn toàn nằm ngoài không gian train split, không bao giờ xuất hiện trong candidate pool hay quá trình tuning.

---

### 3.4. Candidate Sampler & Ghép nối TACSModel (Gói 4)
Đã triển khai script kiểm thử độc lập [scripts/verify_tacs_pipeline.py](file:///e:/BTL-XLA/scripts/verify_tacs_pipeline.py) để nghiệm thu luồng dữ liệu thực tế từ DataLoader qua mô hình:
1. **Lấy mẫu ứng viên tái lập (Reproducible Sampling):**
   Với cùng seed và cùng epoch, nhóm ứng viên được sinh ra là đồng nhất; khi chuyển sang epoch mới qua `dataset.set_epoch(epoch)`, nhóm ứng viên thay đổi ngẫu nhiên nhưng vẫn đảm bảo tính tái lập.
2. **Luồng tính toán Forward & Backward:**
   - Kích thước batch thử nghiệm: `query: [B=4, C=3, H=32, W=32]`, `candidates: [B=4, N=8, C=3, H=32, W=32]`.
   - Selector tính trọng số qua Gumbel-Softmax: tổng trọng số trên 8 ứng viên đạt chính xác $1.0$.
   - Tính toán hàm mất mát kết hợp `TACSLoss` gồm Task Loss (Cross-Entropy), Policy Loss (REINFORCE) và Reward.
   - **100% tham số Selector (6/6 tensors) và Classifier (4/4 tensors) đều nhận gradient hợp lệ khác 0 sau khi gọi `.backward()`**.
   - Ở chế độ `eval()`, Selector chuyển sang lựa chọn cực đại tất định (`argmax`), đảm bảo tính nhất quán khi suy luận.

---

### 3.5. Trực quan hóa & Audit trực quan 20 Query (Gói 5)
1. **Module trực quan hóa batch:**
   - [src/tacs/visualize_batch.py](file:///e:/BTL-XLA/src/tacs/visualize_batch.py) cho phép vẽ lưới hiển thị query và các candidates tương ứng với nhãn chi tiết.
   - Đã có unit test tự động [tests/test_visualize.py](file:///e:/BTL-XLA/tests/test_visualize.py) xác minh số lượng axes ($N+1$) và hiển thị đúng tiêu đề.
2. **Audit trực quan 20 query ngẫu nhiên:**
   - Chạy script [scripts/inspect_20_queries.py](file:///e:/BTL-XLA/scripts/inspect_20_queries.py) kiểm tra $20$ queries và $160$ candidates.
   - Kết quả xuất ra file ảnh: [docs/visual_inspection_20_queries.png](file:///e:/BTL-XLA/docs/visual_inspection_20_queries.png).
   - **Đánh giá chất lượng:**
     * $0/20$ query bị trùng lặp index với candidate pool ($100\%$ an toàn rò rỉ).
     * $100\%$ ảnh sau denormalize nằm chuẩn xác trong dải $[0.00, 1.00]$, không bị clipping, bão hòa hay lỗi NaN/Inf.
     * Tỷ lệ ứng viên cùng nhãn là $8.12\%$ và khác nhãn là $91.88\%$, phản ánh hoàn hảo phân bố ngẫu nhiên đều 10 lớp ($10\%$ cùng lớp, $90\%$ khác lớp).

---

### 3.6. Benchmark hiệu năng DataLoader & Bộ nhớ (Gói 5 & 6)
Thử nghiệm benchmark trong môi trường sạch trên CPU với $50$ batch ($3.200$ queries):
- **Thời gian xử lý:** $7.506$ giây.
- **Tốc độ thông lượng (Throughput):** $426.33$ queries/giây (lần chạy thứ 2 khi OS disk cache làm nóng đạt $1.280.56$ queries/giây).
- **Bộ nhớ tiêu thụ (RAM RSS):** $773.31$ MB sau khi nạp CIFAR-10 và ổn định xuyên suốt quá trình nạp batch (không phát hiện rò rỉ bộ nhớ - memory leak).

---

## 4. Kết quả Kiểm thử Hệ thống (Test Suite)

Chạy lệnh kiểm thử toàn diện:
```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

**Kết quả:**
```text
........
----------------------------------------------------------------------
Ran 8 tests in 0.750s

OK
```

### Danh mục 8 unit tests đã nghiệm thu:
1. `StratifiedSplitTest.test_split_is_balanced_and_reproducible`: Đạt (cân bằng lớp, tái lập).
2. `StratifiedSplitTest.test_split_save_and_load`: Đạt (lưu và tải file JSON split).
3. `ContextDatasetTest.test_sampling_is_reproducible`: Đạt (tái lập seed, khác epoch thì khác candidates).
4. `ContextDatasetTest.test_batch_integrates_with_tacs`: Đạt (ghép DataLoader vào TACSModel, gradient lan truyền).
5. `TACSModelTest.test_forward_shapes_and_hard_selection`: Đạt (kích thước tensor chuẩn, trọng số one-hot).
6. `TACSModelTest.test_hybrid_loss_backpropagates_to_selector`: Đạt (backward pass về Selector).
7. `TACSModelTest.test_eval_selection_is_deterministic`: Đạt (tính tất định khi đánh giá).
8. `VisualizeBatchTest.test_make_context_grid_returns_query_and_candidate_axes`: Đạt (lưới hiển thị $B \times (N+1)$ axes).

---

## 5. Đánh giá Nghiệm thu & Điều kiện chuyển tiếp Tuần 2

### Đối chiếu 8 sản phẩm bàn giao cuối Tuần 1:
- [x] 1. Module CIFAR-10 và transform ([src/tacs/data.py](file:///e:/BTL-XLA/src/tacs/data.py)).
- [x] 2. Module chia dữ liệu phân tầng (`stratified_split`).
- [x] 3. Candidate pool và Candidate sampler (`ContextDataset`).
- [x] 4. File cấu hình dữ liệu và split index (`data/splits/cifar10_seed42.json`).
- [x] 5. Unit test chống data leakage ([tests/test_data.py](file:///e:/BTL-XLA/tests/test_data.py)).
- [x] 6. Script trực quan hóa batch ([src/tacs/visualize_batch.py](file:///e:/BTL-XLA/src/tacs/visualize_batch.py)).
- [x] 7. Pipeline tích hợp với skeleton TACS ([scripts/verify_tacs_pipeline.py](file:///e:/BTL-XLA/scripts/verify_tacs_pipeline.py)).
- [x] 8. Báo cáo kỹ thuật tổng thể ([README.md](file:///e:/BTL-XLA/README.md), [docs/REPORT.md](file:///e:/BTL-XLA/docs/REPORT.md), [docs/PROGRESS.md](file:///e:/BTL-XLA/docs/PROGRESS.md), [docs/BAO_CAO_TIEN_DO_TUAN_1.md](file:///e:/BTL-XLA/docs/BAO_CAO_TIEN_DO_TUAN_1.md)).

### Kế hoạch chuyển tiếp Tuần 2 (Milestone M2):
Toàn bộ nhóm chính thức bước sang Tuần 2 theo kế hoạch chi tiết tại [WEEK_2_ASSIGNMENT.md](file:///e:/BTL-XLA/docs/WEEK_2_ASSIGNMENT.md) để triển khai 3 baseline bắt buộc trên cùng một hạ tầng thống nhất:
1. **Baseline 1 (No-context):** Phân loại ảnh độc lập (vector ngữ cảnh bằng 0).
2. **Baseline 2 (Random-context):** Chọn ngẫu nhiên 1 ứng viên từ candidate pool.
3. **Baseline 3 (Similarity-context):** Chọn ứng viên có Cosine Similarity cao nhất trong không gian embedding.

Báo cáo tiến độ Tuần 2 sẽ được tổng hợp tại `docs/BAO_CAO_TIEN_DO_TUAN_2.md` theo đúng quy chuẩn nêu trên.
