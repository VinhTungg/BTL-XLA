# BÁO CÁO TỔNG KẾT BÀI TẬP LỚN
## MÔN HỌC: XỬ LÝ ẢNH & THỊ GIÁC MÁY TÍNH (BTL-XLA)

---

# ĐỀ TÀI: TÁI HIỆN VÀ CẢI TIẾN PHƯƠNG PHÁP TASK-ALIGNED CONTEXT SELECTION (TACS) CHO BÀI TOÁN PHÂN LOẠI ẢNH

- **Giảng viên hướng dẫn:** [Tên Giảng viên hướng dẫn]
- **Nhóm thực hiện:** Nhóm BTL-XLA
- **Danh sách thành viên:**
  1. **Thành viên 1 (Trưởng nhóm):** [Họ và tên] - MSSV: [Mã số SV] - Phụ trách: Môi trường, Dependency, Lõi huấn luyện Baseline.
  2. **Thành viên 2:** [Họ và tên] - MSSV: [Mã số SV] - Phụ trách: Unit Test, Trực quan hóa dữ liệu (Visualization), Thống kê hiệu năng.
  3. **Thành viên 3:** [Họ và tên] - MSSV: [Mã số SV] - Phụ trách: Data Pipeline, Kiểm định chống rò rỉ (No-leakage), Tích hợp DataLoader - TACSModel, Audit trực quan, Quản lý tài liệu.
- **Thời gian thực hiện:** Học kỳ [I / II] - Năm học 2025 - 2026
- **Kho lưu trữ mã nguồn (Repository):** thư mục gốc của dự án `BTL XLA`

---

## TÓM TẮT DỰ ÁN (EXECUTIVE ABSTRACT)

Trong các bài toán thị giác máy tính dựa trên truy xuất (Retrieval-Augmented Computer Vision), phương pháp truyền thống thường giả định rằng các mẫu tham chiếu có độ tương đồng hình ảnh cao nhất (Visual Similarity) sẽ mang lại hiệu quả tốt nhất cho mô hình dự đoán. Tuy nhiên, nghiên cứu gần đây từ bài báo tại CVPR 2026 (*"Learning What Helps: Task-Aligned Context Selection for Vision Tasks"*) đã chỉ ra rằng ảnh trông giống nhất chưa chắc đã là ảnh hữu ích nhất. Khung làm việc **TACS (Task-Aligned Context Selection)** ra đời nhằm chuyển dịch mục tiêu: **dạy mô hình học cách chọn ảnh ngữ cảnh (context) giúp tối đa hóa độ chính xác của tác vụ xuôi dòng (downstream task), ngay cả khi ảnh đó khác lớp hoặc mang tính tương phản ranh giới**.

Đồ án này tập trung vào 3 mục tiêu cốt lõi:
1. **Chuẩn hóa nền tảng dữ liệu và môi trường:** Xây dựng pipeline dữ liệu CIFAR-10 với chiến lược chia phân tầng độc lập (Query-train 36k, Candidate pool 9k, Validation 5k, Test 10k) đảm bảo tuyệt đối không rò rỉ dữ liệu ($0\%$ leakage).
2. **Tái hiện nguyên lý TACS:** Xây dựng mô hình 2 khối (TaskAlignedSelector và Downstream Classifier) liên kết bằng cơ chế Gumbel-Softmax straight-through và hàm mất mát kết hợp Task Loss + Policy Gradient Reward.
3. **Đề xuất cải tiến Hai giai đoạn (Two-Stage Task-Aligned Retrieval):** Kết hợp trích xuất đặc trưng thô nhanh (Coarse Retrieval) với bộ tái xếp hạng theo nhiệm vụ (Task-Aligned Reranking) nhằm giải quyết bài toán nút thắt cổ chai tính toán và bộ nhớ khi kích thước Candidate Pool mở rộng.

Thực nghiệm chính thức huấn luyện 5 epoch cho năm mô hình trên toàn bộ
CIFAR-10. Ở cấu hình mặc định và seed 42, Similarity-context đạt test
accuracy cao nhất (42,67%). Ablation cho thấy giảm policy weight từ 1,0 xuống
0,1 giúp TACS đạt 43,32% tại seed 42; tuy nhiên, đối chứng ba seed cho kết
quả trung bình TACS 42,83% ± 1,01 và Similarity-context 43,05% ± 1,32.

---

# MỤC LỤC
1. [CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ](#chương-1-giới-thiệu--đặt-vấn-đề)
2. [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & PHƯƠNG PHÁP TACS GỐC](#chương-2-cơ-sở-lý-thuyết--phương-pháp-tacs-gốc)
3. [CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT CẢI TIẾN (TWO-STAGE TACS)](#chương-3-phương-pháp-đề-xuất-cải-tiến-two-stage-tacs)
4. [CHƯƠNG 4: THIẾT KẾ THỰC NGHIỆM & PIPELINE DỮ LIỆU](#chương-4-thiết-kế-thực-nghiệm--pipeline-dữ-liệu)
5. [CHƯƠNG 5: KẾT QUẢ TRIỂN KHAI & ĐÁNH GIÁ HỆ THỐNG](#chương-5-kết-quả-triển-khai--đánh-giá-hệ-thống)
6. [CHƯƠNG 6: TIẾN ĐỘ THỰC HIỆN VÀ KHẢ NĂNG TÁI LẬP](#chương-6-tiến-độ-thực-hiện-và-khả-năng-tái-lập)
7. [CHƯƠNG 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN](#chương-7-kết-luận--hướng-phát-triển)
8. [TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)

---

# CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ

## 1.1. Bối cảnh nghiên cứu
Trong thị giác máy tính hiện đại, việc sử dụng ngữ cảnh hỗ trợ (In-Context Learning hoặc Retrieval-Augmented Visual Inference) đã mở ra tiềm năng lớn cho các mô hình phân loại ảnh, phân đoạn và nhận diện vật thể. Thay vì đưa ra quyết định độc lập trên một bức ảnh truy vấn (query), mô hình được cung cấp thêm một hoặc nhiều ảnh tham chiếu (context/candidates) để so sánh, đối chiếu trước khi suy luận.

## 1.2. Giới hạn của phương pháp truyền thống (Similarity-based Retrieval)
Các hệ thống hiện nay chủ yếu dùng vector nhúng (embedding) từ các mô hình nền tảng (CLIP, DINOv2) và đo khoảng cách Cosine Similarity:
$$\text{sim}(x_q, c_i) = \frac{\mathbf{e}(x_q) \cdot \mathbf{e}(c_i)}{\|\mathbf{e}(x_q)\| \|\mathbf{e}(c_i)\|}$$
Ứng viên $c^*$ có độ tương đồng cao nhất sẽ được chọn làm ngữ cảnh. Tuy nhiên, cách tiếp cận này tồn tại hai nhược điểm cốt tử:
1. **Trùng lặp thông tin (Redundancy):** Một bức ảnh quá giống ảnh truy vấn thường không cung cấp thêm bất kỳ tri thức mới nào để giúp mô hình giải quyết các ca khó.
2. **Không căn chỉnh theo mục tiêu nhiệm vụ (Task-Misalignment):** Trong bài toán phân biệt ranh giới khó (ví dụ: phân biệt Chó vs. Mèo, Chim vs. Máy bay), bức ảnh có giá trị nhất có thể là một bức ảnh thuộc lớp dễ nhầm lẫn nhưng có góc chụp hoặc đặc trưng đối lập rõ rệt, giúp mô hình nhận biết điểm phân tách ranh giới lớp.

## 1.3. Mục tiêu và đóng góp của đồ án
Đồ án này đặt ra các đóng góp cụ thể:
- Làm chủ và tái hiện kiến trúc **TACS** trên tập dữ liệu chuẩn CIFAR-10.
- Xây dựng một quy trình chia dữ liệu và DataLoader chuẩn mực, chứng minh $0\%$ rò rỉ dữ liệu giữa tập huấn luyện, kho ứng viên và kiểm thử.
- Đề xuất kiến trúc **Two-Stage Task-Aligned Retrieval** giảm giai đoạn
  selector có thể học từ $O(BND)$ xuống $O(BkD)$ với $k \ll N$; coarse
  retrieval vẫn quét $N$ embedding bằng phép nhân ma trận chi phí thấp.
- Tối ưu candidate embedding bank dùng chung, loại bỏ việc mã hóa lại
  $B \times k$ ảnh trong mỗi batch của Two-Stage.
- Thực hiện benchmark năm mô hình, ablation siêu tham số và đối chứng
  ba seed, báo cáo cả accuracy, latency, throughput và RAM.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & PHƯƠNG PHÁP TACS GỐC

## 2.1. Kiến trúc tổng thể của TACS
Mô hình TACS bao gồm hai module phối hợp chặt chẽ:
1. **Module 1 - TaskAlignedSelector:** Nhận ảnh truy vấn $x_q$ và nhóm $N$ ảnh ứng viên $\{c_1, c_2, \dots, c_N\}$. Selector chấm điểm tiện ích (utility score) và lựa chọn ra 1 ứng viên tối ưu $c^*$.
2. **Module 2 - Downstream Task Network (Classifier):** Nhận cặp biểu diễn kết hợp $\mathbf{z} = [\mathbf{e}(x_q); \mathbf{e}(c^*)]$ và dự đoán phân bố xác suất nhãn của $x_q$.

```
┌────────────────────────────────────────────────────────┐
│                      TACS PIPELINE                     │
└────────────────────────────────────────────────────────┘
  Query Image (x_q) ──────────► [ Vision Encoder ] ──► e(x_q) ──┐
                                                                 │
  Candidate Pool (c_1..c_N) ──► [ Vision Encoder ] ──► e(c_i) ──┼──► [ Selector Score ]
                                                                 │            │
                                                                 │     Gumbel-Softmax
                                                                 │            │
                                                                 │            ▼
                                                                 └──► Selected e(c*)
                                                                              │
                                                                       Concat [e(q); e(c*)]
                                                                              │
                                                                              ▼
                                                                     [ Classifier Head ]
                                                                              │
                                                                              ▼
                                                                        Logits / Loss
```

## 2.2. Cơ chế lựa chọn rời rạc vi phân được (Straight-Through Gumbel-Softmax)
Thao tác chọn 1 ứng viên từ tập $N$ phần tử vốn là thao tác rời rạc (chọn chỉ số $i^* = \arg\max s_i$), do đó đạo hàm bằng 0 tại hầu hết mọi điểm khiến gradient không thể lan truyền về Selector.

TACS giải quyết vấn đề này bằng kỹ thuật **Gumbel-Softmax Straight-Through**:
1. Thêm nhiễu Gumbel $g_i \sim \text{Gumbel}(0, 1)$ vào utility score $s_i$:
   $$g_i = -\log(-\log(u_i)), \quad u_i \sim \text{Uniform}(0, 1)$$
2. Tính trọng số mềm theo nhiệt độ $\tau$:
   $$w_i^{\text{soft}} = \frac{\exp((s_i + g_i)/\tau)}{\sum_{j=1}^N \exp((s_j + g_j)/\tau)}$$
3. Ở lượt Forward, biến đổi thành vector One-Hot cứng:
   $$w_i^{\text{hard}} = \begin{cases} 1 & \text{nếu } i = \arg\max_j w_j^{\text{soft}} \\ 0 & \text{ngược lại} \end{cases}$$
4. Ở lượt Backward, gradient được truyền thẳng qua xấp xỉ liên tục:
   $$w = w^{\text{soft}} + \text{stop\_gradient}(w^{\text{hard}} - w^{\text{soft}})$$

## 2.3. Tối ưu hóa Policy Gradient & Định nghĩa Phần thưởng (Reward)
Selector được mô hình hóa như một Policy chọn hành động. Phần thưởng $r$ được xác định dựa trên mức độ ngữ cảnh $c^*$ giúp cải thiện hàm mất mát so với khi không có ngữ cảnh:
$$r = \mathcal{L}_{\text{no\_context}} - \mathcal{L}_{\text{context}}$$
- $r > 0$: Ứng viên giúp giảm loss, phân loại chính xác hơn $\rightarrow$ Thưởng.
- $r < 0$: Ứng viên gây nhiễu, làm tăng loss $\rightarrow$ Phạt.

Để giảm phương sai của gradient, phần thưởng được chuẩn hóa thành Advantage $A$ trong từng mini-batch:
$$A_b = \frac{r_b - \mu_r}{\sigma_r + \epsilon}$$
Hàm mất mát Policy được tối ưu theo công thức:
$$\mathcal{L}_{\text{policy}} = - \frac{1}{B} \sum_{b=1}^B \log P(c^* | x_q) \cdot A_b$$

## 2.4. Hàm mất mát kết hợp (Hybrid Loss)
Hàm mất mát tổng thể kết hợp cả hai mục tiêu:
$$\mathcal{L}_{TACS} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{policy}}$$
- $\mathcal{L}_{\text{task}}$: Cross-Entropy loss của mạng phân loại khi có context.
- $\lambda$: Hệ số điều chỉnh độ quan trọng của policy loss. Source dùng
  mặc định $\lambda = 1,0$; ablation khảo sát thêm $\lambda = 0,1$ và
  $\lambda = 0$ (task-loss-only).

---

# CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT CẢI TIẾN (TWO-STAGE TACS)

## 3.1. Điểm nghẽn tính toán của TACS gốc
Trong TACS nguyên bản, nếu kho ứng viên mở rộng lên kích thước $N = 10.000$ hoặc $50.000$ mẫu:
- Mỗi ảnh query cần tính ma trận tương quan và qua Selector với toàn bộ $N$ ảnh.
- Chi phí bộ nhớ và thời gian tăng tuyến tính $\mathcal{O}(B \times N \times D)$, vượt quá dung lượng GPU phổ thông và làm giảm tốc độ suy luận nghiêm trọng.

## 3.2. Ý tưởng Cải tiến: Truy xuất Hai giai đoạn (Two-Stage Retrieval)
Nhóm đề xuất giải pháp Two-Stage kết hợp ưu điểm của cả hai trường phái:

```
Candidate Pool (N = 9.000)
       │
       ▼ [Giai đoạn 1: Lọc thô siêu nhanh (Coarse Stage)]
       │  - Sử dụng embedding tiền tính từ standard TinyVisionEncoder đã khóa.
       │  - Truy xuất k ứng viên gần nhất bằng tích vô hướng Cosine (k = 32 << N).
       │  - Chi phí cực thấp, tốc độ mili-giây.
       ▼
Filtered Top-k Candidates (k = 32)
       │
       ▼ [Giai đoạn 2: Tái xếp hạng thích ứng nhiệm vụ (Task-Aligned Reranking)]
       │  - Đưa Top-k vào TACS Selector có thể học.
       │  - Gumbel-Softmax chấm điểm Utility trong phạm vi 32 ứng viên.
       │  - Chọn ra 1 ứng viên tối ưu nhất cho Downstream Task.
       ▼
Selected Context c* ──► Classifier
```

## 3.3. So sánh độ phức tạp tính toán

| Tiêu chí | TACS Full | Two-Stage TACS | Kết quả |
|---|:---:|:---:|:---:|
| **Số candidate Selector chấm/query** | $N=9.000$ | $k=32$ | Giảm lý thuyết $9000/32 \approx 281$ lần |
| **Độ phức tạp Selector** | $\mathcal{O}(BND)$ | $\mathcal{O}(BkD)$ | Two-Stage tỷ lệ theo $k$, không theo $N$ |
| **End-to-end 5 epoch trên CPU** | 240,83 giây | 148,25 giây | Two-Stage nhanh hơn 1,62 lần |
| **Peak RAM đo được** | 859,11 MB | 875,07 MB | Không ghi nhận lợi thế RAM trong run CPU này |

Candidate của cả hai pipeline được mã hóa thành một embedding bank detach,
làm mới mỗi train/validation/test pass. Two-Stage chỉ truyền vị trí top-k trong
DataLoader rồi gom embedding từ bank chung; nhờ đó không còn mã hóa lại
$B \times k$ ảnh candidate ở từng batch. Môi trường không có CUDA nên báo
cáo không suy diễn mức tiết kiệm VRAM khi chưa có số đo GPU.

---

# CHƯƠNG 4: THIẾT KẾ THỰC NGHIỆM & PIPELINE DỮ LIỆU

## 4.1. Bộ dữ liệu CIFAR-10 & Chiến lược Phân chia
Tập dữ liệu CIFAR-10 gồm 60.000 ảnh màu $32 \times 32$ chia đều trên 10 lớp. Nhóm áp dụng thuật toán phân chia phân tầng (`stratified_split`, seed cố định 42) tạo ra 4 tập riêng biệt:

| Tập dữ liệu | Số lượng ảnh | Tỷ lệ (%) | Mục đích sử dụng | Transform áp dụng |
|---|:---:|:---:|---|---|
| **Query-Train** | $36.000$ | $72\%$ train | Huấn luyện bộ mã hóa và phân loại | Random Crop (32, pad 4), Random Flip, Normalize |
| **Candidate Pool** | $9.000$ | $18\%$ train | Kho mẫu tham chiếu làm ngữ cảnh | **Chỉ Normalize (Không biến dạng ngẫu nhiên)** |
| **Validation** | $5.000$ | $10\%$ train | Tinh chỉnh siêu tham số và Early Stopping | Chỉ Normalize |
| **Test Set** | $10.000$ | Độc lập | Đánh giá khách quan độ chính xác cuối cùng | Chỉ Normalize |

### Bảng phân bố chi tiết 10 lớp trên các tập:

| Lớp | Tên lớp | Query-Train | Candidate Pool | Validation | Test Set | Tổng |
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
| **Tổng** | **10 Lớp** | **36.000** | **9.000** | **5.000** | **10.000** | **60.000** |

## 4.2. Kiểm định chống Rò rỉ Dữ liệu (No-Leakage Proof)
Để đảm bảo tính trung thực khoa học, hệ thống áp dụng kiểm định tập hợp chặt chẽ:
$$\mathcal{I}_{\text{query}} \cap \mathcal{I}_{\text{candidate}} = \emptyset, \quad \mathcal{I}_{\text{query}} \cap \mathcal{I}_{\text{val}} = \emptyset, \quad \mathcal{I}_{\text{candidate}} \cap \mathcal{I}_{\text{val}} = \emptyset$$
$$\mathcal{I}_{\text{query}} \cup \mathcal{I}_{\text{candidate}} \cup \mathcal{I}_{\text{val}} = \{0, 1, \dots, 49.999\}$$
Tập kiểm thử ($10.000$ mẫu) hoàn toàn cách ly, không bao giờ được phép xuất hiện trong Candidate Pool.

## 4.3. Thiết lập các Baseline so sánh bắt buộc
Để đánh giá TACS và cải tiến Two-Stage, nhóm thiết kế 5 mô hình
thực nghiệm đối sánh trên cùng một kiến trúc backbone:
1. **Baseline 1 (No-Context):** Phân loại ảnh đơn thuần, vector context được đặt bằng vector 0.
2. **Baseline 2 (Random-Context):** Chọn ngẫu nhiên một ảnh trong Candidate Pool đưa vào Classifier.
3. **Baseline 3 (Similarity-Context):** Lấy ảnh có Cosine Similarity cao nhất trong không gian vector nhúng.
4. **Mô hình 4 (TACS Full):** Tái hiện nguyên bản TACS với Gumbel-Softmax và Policy Gradient.
5. **Mô hình 5 (Two-Stage TACS - Đề xuất cải tiến):** Lọc thô Top-32 bằng Cosine Similarity, sau đó dùng TACS Selector để chọn ứng viên tối ưu.

---

# CHƯƠNG 5: KẾT QUẢ TRIỂN KHAI & ĐÁNH GIÁ HỆ THỐNG

## 5.1. Kiểm thử và tính toàn vẹn thực nghiệm

Bộ test tự động hiện có 38 test: 37 test đạt và 1 integration test
CIFAR-10 được skip theo mặc định. Khi bật cờ chạy dữ liệu thật, integration
test này đạt 1/1. Phạm vi kiểm tra bao gồm:

- split phân tầng và không rò rỉ dữ liệu;
- shape, hard selection, gradient của TACS và hybrid loss;
- cache embedding và provenance của split/encoder/candidate pool;
- TACS Full chấm đủ candidate pool và Two-Stage dùng top-k thực;
- shared candidate embedding bank không nhân bản ảnh theo query;
- checkpoint, metrics và script tổng hợp năm mô hình.

Trainer không còn đường fallback sang mock data. Mọi run trong chương này ghi
`data.source=cifar10`, đủ 36.000 query-train, 9.000 candidate, 5.000 validation,
10.000 test và hash của file split.

## 5.2. Benchmark chính thức năm mô hình

Cấu hình chung: 5 epoch, batch size 64, seed 42, CPU, đánh giá test bằng
checkpoint có validation accuracy tốt nhất. TACS Full và Two-Stage trong bảng này
dùng hybrid loss mặc định $\lambda=1,0$.

| Mô hình | Val Acc | Test Acc | Latency (ms/query) | Throughput (q/s) | Peak RAM (MB) | Retrieval / Selector |
|---|---:|---:|---:|---:|---:|---:|
| No-context | 41,92% | 42,47% | 0,1731 | 5.909,64 | 850,20 | 0 / 0 |
| Random-context | 42,02% | 42,54% | 0,3845 | 2.617,11 | 877,41 | 0 / 1 |
| **Similarity-context** | **42,44%** | **42,67%** | 0,4483 | 2.391,47 | 851,26 | 9.000 / 1 |
| TACS Full | 40,76% | 42,05% | 0,7986 | 1.288,54 | 859,11 | 0 / 9.000 |
| Two-Stage TACS | 41,22% | 41,91% | 0,7240 | 1.382,70 | 875,07 | 9.000 / 32 |

![Biểu đồ so sánh năm mô hình](../outputs/final_5epoch/report/all_models_comparison.png)

Ở cấu hình mặc định, Similarity-context đạt accuracy cao nhất. TACS Full
và Two-Stage chưa vượt baseline này; vì vậy không thể kết luận task-aligned
selection cải thiện chất lượng chỉ từ run seed 42 mặc định.

## 5.3. Hiệu quả tối ưu shared embedding bank

Trước tối ưu, Two-Stage trả về 32 ảnh candidate cho từng query và mã
hóa lại chúng trong từng batch. Sau tối ưu, dataset chỉ trả về vị trí top-k;
trainer mã hóa candidate pool một lần cho mỗi pass và gom embedding theo chỉ số.

| Chỉ số pilot 1 epoch | Trước tối ưu | Sau tối ưu | Thay đổi |
|---|---:|---:|---:|
| End-to-end | 379,13 s | 52,13 s | Nhanh hơn 7,27 lần |
| Train throughput | 120,90 q/s | 1.775,16 q/s | Tăng 14,68 lần |
| Train model time | 285,02 s | 18,35 s | Nhanh hơn 15,54 lần |
| Peak RAM | 1.025,49 MB | 889,63 MB | Giảm 13,2% |
| Test accuracy | 30,26% | 30,99% | Không suy giảm trong pilot |

Kết quả 5 epoch cũng cho thấy Two-Stage hoàn tất trong 148,25 giây, nhanh
hơn TACS Full 240,83 giây khoảng 1,62 lần trên CPU.

## 5.4. Ablation temperature, policy weight và top-K

Giai đoạn sàng lọc chạy 3 epoch và chỉ dùng validation để chọn siêu tham
số, không xem test. Kết quả TACS:

| Temperature | Policy weight | Best validation 3 epoch |
|---:|---:|---:|
| 0,1 | 1,0 | 37,02% |
| **0,1** | **0,0** | **37,26%** |
| 0,1 | 0,1 | 37,20% |
| 0,5 | 0,0 | 37,24% |
| 1,0 | 0,0 | 37,16% |

Với Two-Stage, giữ temperature 0,1 và policy weight 0, top-K=32 đạt validation
37,94%, cao hơn K=8 (37,20%), K=16 (35,04%) và K=64 (37,18%).

Xác nhận 5 epoch tại seed 42:

| Mô hình | Policy weight | Validation | Test |
|---|---:|---:|---:|
| TACS mặc định | 1,0 | 40,76% | 42,05% |
| **TACS hybrid nhẹ** | **0,1** | **42,30%** | **43,32%** |
| TACS task-loss-only | 0,0 | 42,14% | 43,48% |
| Two-Stage mặc định | 1,0 | 41,22% | 41,91% |
| Two-Stage hybrid nhẹ | 0,1 | 41,42% | 41,61% |
| Two-Stage task-loss-only | 0,0 | 41,58% | 42,30% |

Giảm policy weight cải thiện TACS rõ tại seed 42. Tuy nhiên,
`policy_weight=0` loại policy-gradient term nên phải được ghi là
task-loss-only ablation, không phải hybrid TACS đầy đủ. Cấu hình 0,1 giữ
hybrid objective và chỉ thấp hơn task-loss-only 0,16 điểm test tại seed 42.

## 5.5. Đối chứng nhiều seed

TACS hybrid nhẹ và Similarity-context được chạy cùng ba seed huấn luyện;
split dữ liệu và standard encoder vẫn cố định.

| Seed | TACS hybrid test | Similarity test | TACS - Similarity |
|---:|---:|---:|---:|
| 7 | 41,67% | 41,97% | -0,30 |
| 42 | 43,32% | 42,67% | +0,65 |
| 123 | 43,51% | 44,52% | -1,01 |
| **Trung bình ± SD** | **42,83% ± 1,01** | **43,05% ± 1,32** | **-0,22 ± 0,83** |

TACS chỉ thắng ở một trong ba seed. Chênh lệch paired trung bình nghiêng về
Similarity-context 0,22 điểm; với cỡ mẫu ba seed, chưa có bằng chứng TACS
vượt baseline một cách ổn định hoặc có ý nghĩa thống kê.

## 5.6. Thảo luận

Kết quả cho thấy hai khía cạnh khác nhau:

1. **Hiệu quả hệ thống:** shared embedding bank giải quyết đúng nút thắt
   mã hóa lặp lại; Two-Stage nhanh hơn TACS Full mà vẫn chấm top-k thực.
2. **Hiệu quả mô hình:** TACS có thể đạt kết quả cao ở một seed sau tuning,
   nhưng phương sai giữa seed lớn và trung bình chưa vượt Similarity-context.
3. **Giới hạn phần cứng:** tất cả số đo trong báo cáo được thu trên CPU;
   peak VRAM không có giá trị và không nên suy diễn hiệu năng GPU.
4. **Giới hạn thống kê:** ba seed phù hợp để phát hiện độ nhạy, nhưng
   chưa đủ cho một kết luận ý nghĩa thống kê mạnh.

---

# CHƯƠNG 6: TIẾN ĐỘ THỰC HIỆN VÀ KHẢ NĂNG TÁI LẬP

Các mốc chính của đồ án đã hoàn thành:

```
[Tuần 1: M0 - M1] (HOÀN THÀNH)
  ├─ Chuẩn hóa môi trường & Test Suite
  ├─ Chia phân tầng CIFAR-10 & Candidate Pool
  └─ Tích hợp DataLoader - TACSModel

[Tuần 2: M1 - M2] (HOÀN THÀNH)
  ├─ Huấn luyện 3 Baseline (No-context, Random, Cosine-Similarity)
  ├─ Đo đạc độ chính xác phân loại Top-1 Accuracy
  └─ Chuẩn hóa checkpoint và cache retrieval

[Tuần 3: M3 - M4] (HOÀN THÀNH)
  ├─ Huấn luyện TACS Full và năm mô hình 5 epoch
  ├─ Triển khai Two-Stage TACS Reranker
  ├─ Đánh giá Accuracy, Latency, Throughput và peak RAM; VRAM ghi N/A trên CPU
  └─ Thực hiện Ablation Study (Nhiệt độ Gumbel tau, Hệ số lambda policy)

[Tuần 4: M5] (ĐANG HOÀN THIỆN)
  ├─ Tổng hợp benchmark, ablation và đối chứng nhiều seed
  ├─ Cập nhật báo cáo tổng thể
  └─ Còn lại: slide thuyết trình và demo nếu môn học yêu cầu
```

Mỗi run lưu `config.json`, `metrics.json`, `checkpoint_best.pt` và
`best_encoder.pt`. Script `scripts/analyze_all_models.py` kiểm tra schema, provenance,
split, số mẫu và sự hiện diện của test result trước khi sinh bảng/biểu đồ.

---

# CHƯƠNG 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

## 7.1. Kết luận
1. Đồ án đã xây dựng được pipeline TACS/Two-Stage hoàn chỉnh trên
   CIFAR-10 thật, có checkpoint, provenance và test tự động.
2. Shared candidate embedding bank là cải tiến hệ thống hiệu quả: pilot
   Two-Stage nhanh hơn 7,27 lần so với triển khai mã hóa lặp lại; run 5 epoch
   nhanh hơn TACS Full 1,62 lần.
3. Ở cấu hình mặc định seed 42, Similarity-context dẫn đầu với 42,67%
   test accuracy. Tuning policy weight giúp TACS đạt 43,32% tại seed 42.
4. Trên ba seed, TACS hybrid nhẹ đạt 42,83% ± 1,01, trong khi
   Similarity-context đạt 43,05% ± 1,32. Do đó, đồ án không tuyên bố
   TACS vượt baseline một cách ổn định.
5. Kết quả quan trọng nhất là phân tách rõ hiệu quả tính toán và hiệu
   quả accuracy: Two-Stage cải thiện tốc độ, nhưng chưa chứng minh được
   lợi thế accuracy trong thiết lập hiện tại.

## 7.2. Hướng phát triển
- Mở rộng từ 3 lên ít nhất 5 seed và dùng kiểm định paired trước khi
  kết luận về chênh lệch accuracy nhỏ.
- Phân tích phân phối reward, entropy của selector và tỷ lệ chọn context đúng
  lớp để giải thích độ nhạy seed.
- Khảo sát temperature annealing, reward baseline theo moving average và policy-weight
  schedule thay cho một hệ số cố định.
- Thử backbone mạnh hơn như ResNet-18, MobileNetV3 hoặc Vision Transformer nhỏ;
  TinyVisionEncoder hiện tại giới hạn trần accuracy của cả năm mô hình.
- Chạy benchmark CUDA thực để đo VRAM và xác nhận khả năng mở rộng
  candidate pool thay vì suy diễn từ số đo CPU.

---

# TÀI LIỆU THAM KHẢO

1. **Guo et al. (CVPR 2026):** *"Learning What Helps: Task-Aligned Context Selection for Vision Tasks"*.
2. **Krizhevsky, A. (2009):** *"Learning Multiple Layers of Features from Tiny Images"* (CIFAR-10 Dataset).
3. **Jang, E., Gu, S., & Poole, B. (2016):** *"Categorical Reparameterization with Gumbel-Softmax"*, ICLR 2017.
4. **Sutton, R. S., & Barto, A. G. (2018):** *"Reinforcement Learning: An Introduction"*, MIT Press.
5. **Radford et al. (2021):** *"Learning Transferable Visual Models From Natural Language Supervision"* (CLIP).
6. **Oquab et al. (2023):** *"DINOv2: Learning Robust Visual Features without Supervision"*.

---
*Báo cáo được khởi tạo và lưu trữ tại `docs/BAO_CAO_TONG_THE_DU_AN.md`.*
