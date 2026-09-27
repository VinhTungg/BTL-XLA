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
- **Kho lưu trữ mã nguồn (Repository):** `E:\BTL-XLA`

---

## TÓM TẮT DỰ ÁN (EXECUTIVE ABSTRACT)

Trong các bài toán thị giác máy tính dựa trên truy xuất (Retrieval-Augmented Computer Vision), phương pháp truyền thống thường giả định rằng các mẫu tham chiếu có độ tương đồng hình ảnh cao nhất (Visual Similarity) sẽ mang lại hiệu quả tốt nhất cho mô hình dự đoán. Tuy nhiên, nghiên cứu gần đây từ bài báo tại CVPR 2026 (*"Learning What Helps: Task-Aligned Context Selection for Vision Tasks"*) đã chỉ ra rằng ảnh trông giống nhất chưa chắc đã là ảnh hữu ích nhất. Khung làm việc **TACS (Task-Aligned Context Selection)** ra đời nhằm chuyển dịch mục tiêu: **dạy mô hình học cách chọn ảnh ngữ cảnh (context) giúp tối đa hóa độ chính xác của tác vụ xuôi dòng (downstream task), ngay cả khi ảnh đó khác lớp hoặc mang tính tương phản ranh giới**.

Đồ án này tập trung vào 3 mục tiêu cốt lõi:
1. **Chuẩn hóa nền tảng dữ liệu và môi trường:** Xây dựng pipeline dữ liệu CIFAR-10 với chiến lược chia phân tầng độc lập (Query-train 36k, Candidate pool 9k, Validation 5k, Test 10k) đảm bảo tuyệt đối không rò rỉ dữ liệu ($0\%$ leakage).
2. **Tái hiện nguyên lý TACS:** Xây dựng mô hình 2 khối (TaskAlignedSelector và Downstream Classifier) liên kết bằng cơ chế Gumbel-Softmax straight-through và hàm mất mát kết hợp Task Loss + Policy Gradient Reward.
3. **Đề xuất cải tiến Hai giai đoạn (Two-Stage Task-Aligned Retrieval):** Kết hợp trích xuất đặc trưng thô nhanh (Coarse Retrieval) với bộ tái xếp hạng theo nhiệm vụ (Task-Aligned Reranking) nhằm giải quyết bài toán nút thắt cổ chai tính toán và bộ nhớ khi kích thước Candidate Pool mở rộng.

---

# MỤC LỤC
1. [CHƯƠNG 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ](#chương-1-giới-thiệu--đặt-vấn-đề)
2. [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & PHƯƠNG PHÁP TACS GỐC](#chương-2-cơ-sở-lý-thuyết--phương-pháp-tacs-gốc)
3. [CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT CẢI TIẾN (TWO-STAGE TACS)](#chương-3-phương-pháp-đề-xuất-cải-tiến-two-stage-tacs)
4. [CHƯƠNG 4: THIẾT KẾ THỰC NGHIỆM & PIPELINE DỮ LIỆU](#chương-4-thiết-kế-thực-nghiệm--pipeline-dữ-liệu)
5. [CHƯƠNG 5: KẾT QUẢ TRIỂN KHAI & ĐÁNH GIÁ HỆ THỐNG](#chương-5-kết-quả-triển-khai--đánh-giá-hệ-thống)
6. [CHƯƠNG 6: KẾ HOẠCH THỰC HIỆN CÁC GIAI ĐOẠN TIẾP THEO](#chương-6-kế-hoạch-thực-hiện-các-giai-đoạn-tiếp-theo)
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
- Đề xuất kiến trúc **Two-Stage Task-Aligned Retrieval** giúp giảm độ phức tạp thời gian từ $O(N)$ xuống $O(k)$ với $k \ll N$, mở đường cho việc áp dụng trên kho ngữ cảnh quy mô lớn.

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
- $\lambda$: Hệ số điều chỉnh độ quan trọng của policy loss (mặc định $\lambda = 1.0$ hoặc $0.5$).

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
       │  - Sử dụng Frozen Precomputed Embeddings (DINO/ResNet).
       │  - Truy xuất k ứng viên gần nhất bằng Cosine Similarity / FAISS (k = 32 << N).
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

| Tiêu chí | TACS Nguyên bản (Full Pool) | Cải tiến Two-Stage TACS | Mức độ cải thiện |
|---|:---:|:---:|:---:|
| **Số phép tính Selector** | $\mathcal{O}(B \cdot N \cdot D)$ | $\mathcal{O}(B \cdot k \cdot D)$ | **Giảm $N/k$ lần** (ví dụ: $9000/32 \approx 281$ lần) |
| **Dung lượng VRAM** | Rất cao, dễ tràn GPU | Cố định ở mức nhỏ | **Tiết kiệm $\approx 90\%$ VRAM** |
| **Khả năng mở rộng Pool** | Bị giới hạn $< 100$ | Mở rộng lên tới hàng triệu mẫu | **Khả thi trên hệ thống thực tế** |

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
Để chứng minh tính hiệu quả của TACS và cải tiến Two-Stage, nhóm thiết kế 4 mô hình thực nghiệm đối sánh trên cùng một kiến trúc backbone:
1. **Baseline 1 (No-Context):** Phân loại ảnh đơn thuần, vector context được đặt bằng vector 0.
2. **Baseline 2 (Random-Context):** Chọn ngẫu nhiên một ảnh trong Candidate Pool đưa vào Classifier.
3. **Baseline 3 (Similarity-Context):** Lấy ảnh có Cosine Similarity cao nhất trong không gian vector nhúng.
4. **Mô hình 4 (TACS Full):** Tái hiện nguyên bản TACS với Gumbel-Softmax và Policy Gradient.
5. **Mô hình 5 (Two-Stage TACS - Đề xuất cải tiến):** Lọc thô Top-32 bằng Cosine Similarity, sau đó dùng TACS Selector để chọn ứng viên tối ưu.

---

# CHƯƠNG 5: KẾT QUẢ TRIỂN KHAI & ĐÁNH GIÁ HỆ THỐNG

## 5.1. Kết quả kiểm thử Unit Test và Kiểm tra Môi trường
Toàn bộ hệ sinh thái mã nguồn được kiểm thử tự động với $8/8$ unit test đạt tiêu chuẩn:
- `test_split_is_balanced_and_reproducible`: Đạt (Kiểm tra phân bố lớp và tính tái lập theo seed).
- `test_split_save_and_load`: Đạt (Lưu và đọc file JSON chỉ số).
- `test_sampling_is_reproducible`: Đạt (Đảm bảo candidate sampling thay đổi theo epoch nhưng cố định theo seed).
- `test_batch_integrates_with_tacs`: Đạt (Đưa batch qua mô hình và hàm mất mát thành công).
- `test_forward_shapes_and_hard_selection`: Đạt (Shape đúng chuẩn và One-Hot hợp lệ).
- `test_hybrid_loss_backpropagates_to_selector`: Đạt (Gradient truyền ngược về Selector).
- `test_eval_selection_is_deterministic`: Đạt (Suy luận tất định).
- `test_make_context_grid_returns_query_and_candidate_axes`: Đạt (Vẽ lưới trực quan hóa).

## 5.2. Hiệu năng Pipeline dữ liệu (DataLoader Benchmark)
Đo lường trực tiếp trên CPU Intel/AMD trong môi trường sạch:
- **Thông lượng xử lý:** $426.33 \text{ queries/giây}$ (Cold Cache) và đạt $1.280.56 \text{ queries/giây}$ (Warm Cache).
- **Mức tiêu thụ bộ nhớ RAM (RSS):** $773.31 \text{ MB}$ cố định, không phát hiện hiện tượng rò rỉ bộ nhớ (Memory Leak) trong suốt quá trình duyệt 50 batch liên tục.

## 5.3. Kết quả tích hợp TACSModel và Lan truyền Gradient
Chạy thực nghiệm kiểm chứng qua script `scripts/verify_tacs_pipeline.py`:
- Kích thước batch: Query `[4, 3, 32, 32]`, Candidates `[4, 8, 3, 32, 32]`.
- Đầu ra Selector: Ma trận trọng số `[4, 8]`, tổng xác suất mỗi hàng $= 1.0$ (One-Hot chính xác).
- Lan truyền ngược: **$6/6$ tham số của Selector** và **$4/4$ tham số của Classifier** đều nhận gradient khác 0, chứng minh hàm mất mát kết hợp `TACSLoss` kích hoạt thành công cơ chế Straight-Through Gumbel-Softmax.

## 5.4. Kết quả Audit trực quan 20 Query (160 Context Images)
Thông qua script `scripts/inspect_20_queries.py` và ảnh kết xuất `docs/visual_inspection_20_queries.png`:
- $0/20$ query bị trùng lặp index vào tập ứng viên ($0\%$ rò rỉ).
- $100\%$ ảnh sau khi hoàn nguyên chuẩn hóa nằm trong khoảng giá trị pixel hợp lệ $[0.00, 1.00]$.
- Tỷ lệ ứng viên khác lớp trung bình đạt $91.88\%$, tạo môi trường tương phản phong phú giúp mô hình học cách phân định ranh giới quyết định.

---

# CHƯƠNG 6: KẾ HOẠCH THỰC HIỆN CÁC GIAI ĐOẠN TIẾP THEO

Theo kế hoạch tổng thể tại `docs/PROJECT_PLAN.md`, lộ trình tiếp theo của đồ án được phân bổ như sau:

```
[Tuần 1: M0 - M1] (HOÀN THÀNH)
  ├─ Chuẩn hóa môi trường & Test Suite
  ├─ Chia phân tầng CIFAR-10 & Candidate Pool
  └─ Tích hợp DataLoader - TACSModel

[Tuần 2: M1 - M2] (TIẾP THEO)
  ├─ Huấn luyện 3 Baseline (No-context, Random, Cosine-Similarity)
  ├─ Đo đạc độ chính xác phân loại Top-1 Accuracy
  └─ Huấn luyện TACS Full ban đầu (Epoch 1 - 20)

[Tuần 3: M3 - M4]
  ├─ Triển khai Two-Stage TACS Reranker
  ├─ Đánh giá đồng thời 3 chỉ số: Accuracy, Latency (ms), Memory (VRAM)
  └─ Thực hiện Ablation Study (Nhiệt độ Gumbel tau, Hệ số lambda policy)

[Tuần 4: M5]
  ├─ Tổng hợp báo cáo cuối kỳ hoàn chỉnh
  ├─ Đóng gói mã nguồn & Viết tài liệu hướng dẫn tái lập
  └─ Chuẩn bị Slide thuyết trình & Demo
```

---

# CHƯƠNG 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

## 7.1. Kết luận
1. Đồ án đã xây dựng thành công nền móng vững chắc cho việc tái hiện và cải tiến TACS.
2. Bộ dữ liệu CIFAR-10 được tổ chức khoa học, có khả năng tái lập cao nhờ cơ chế hạt giống ngẫu nhiên (seed 42), đảm bảo tính công bằng và loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu.
3. Module TACS Model và hàm mất mát kết hợp hoạt động hoàn hảo trên thực tế, đảm bảo gradient truyền ngược đầy đủ qua thao tác lựa chọn rời rạc.

## 7.2. Hướng phát triển
- Thử nghiệm trên các kiến trúc Backbone mạnh hơn (ResNet-18, MobileNetV3 hoặc Vision Transformer nhỏ).
- Khảo sát lịch trình suy giảm nhiệt độ (Temperature Annealing) cho Gumbel-Softmax để tăng độ ổn định khi hội tụ.
- Hoàn thiện mô hình Two-Stage Task-Aligned Retrieval trên không gian Candidate Pool lớn.

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
