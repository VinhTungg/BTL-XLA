# Báo cáo BTL - TACS và phương pháp cải tiến

## 1. Mở đầu

Mô hình truy xuất ảnh thông thường xem độ tương tự thị giác là đại diện cho tính hữu ích. TACS thay đổi mục tiêu này: context được chọn theo mức cải thiện nhiệm vụ downstream.

## 2. Bài toán và mục tiêu

- Tái hiện nguyên lý TACS cho phân loại ảnh.
- So sánh công bằng với ba baseline.
- Cải thiện khả năng mở rộng candidate pool bằng truy xuất hai giai đoạn.

Mục tiêu trung tâm: **dạy mô hình chọn ảnh có ích cho quyết định, thay vì chỉ chọn ảnh trông giống ảnh đầu vào**.

## 3. Cơ sở lý thuyết

### 3.1 Retrieval theo độ tương tự

Trong truy xuất ảnh thông thường, mỗi ảnh được biểu diễn bởi một vector đặc trưng. Hệ thống lấy ảnh có cosine similarity cao nhất. Cách này trả lời *“ảnh nào giống nhất?”*, nhưng không đảm bảo trả lời *“ảnh nào giúp phân loại tốt hơn?”*.

Ví dụ, khi phân biệt hai loài chim rất giống nhau, một ảnh gần như trùng lặp có thể không cung cấp thêm thông tin. Ảnh của loài dễ nhầm lẫn nhưng có mỏ hoặc màu cánh khác biệt lại có thể giúp mô hình nhận ra ranh giới giữa hai lớp.

### 3.2 TACS là gì?

TACS là viết tắt của **Task-Aligned Context Selection**, hay *lựa chọn ngữ cảnh phù hợp với nhiệm vụ*. Khung này cho phép mô hình tự học xem ảnh tham chiếu nào thực sự cải thiện nhiệm vụ downstream.

TACS gồm hai khối:

1. **Selector:** nhận ảnh truy vấn và candidate pool, sau đó chấm utility score cho từng ứng viên.
2. **Downstream Task Network:** nhận cặp `query + context` và sinh dự đoán cuối cùng.

Luồng xử lý:

`Query + candidate pool → Selector chấm điểm → chọn context → Task Network dự đoán → loss/reward phản hồi cho Selector`.

Selector không học tìm ảnh giống nhất mà học tìm ảnh làm kết quả downstream tốt hơn. Context vì vậy có thể thuộc lớp khác, miễn là nó cung cấp sự tương phản hữu ích.

### 3.3 Gumbel-Softmax straight-through

Chọn một ảnh là thao tác rời rạc, nên `argmax` không cho gradient đi qua. Straight-through Gumbel-Softmax giữ lựa chọn one-hot ở lượt forward, nhưng dùng xấp xỉ mềm ở lượt backward. Nhờ đó, task loss có thể truyền gradient về Selector.

### 3.4 Policy gradient, reward và advantage

TACS xem Selector như một policy: hành động là chọn một candidate. Reward đo mức context làm thay đổi loss:

\[
r = L_{\text{không context}} - L_{\text{có context}}
\]

- `r > 0`: context làm loss giảm, lựa chọn hữu ích.
- `r = 0`: context hầu như không tạo khác biệt.
- `r < 0`: context làm dự đoán tệ hơn.

Reward được chuẩn hóa thành advantage trong batch nhằm giảm nhiễu. Policy gradient làm tăng xác suất của lựa chọn tốt và giảm xác suất của lựa chọn gây hại.

### 3.5 Hàm mất mát kết hợp

\[
L_{TACS} = L_{grad} + \lambda L_{policy}
\]

`L_grad` huấn luyện tác vụ và truyền gradient ổn định qua Gumbel-Softmax. `L_policy` đánh giá lựa chọn rời rạc theo reward. Hệ số `λ` cân bằng hai thành phần. Khi suy luận, mô hình chọn candidate có utility score cao nhất.

### 3.6 Ý nghĩa đối với dự án

Dự án không chỉ xây một công cụ tìm ảnh. Truy xuất trở thành một thành phần có thể học của quá trình ra quyết định. Chất lượng Selector phải được đánh giá bằng mức cải thiện downstream, không chỉ bằng similarity của cặp ảnh.

## 4. Phương pháp đề xuất

Two-stage task-aligned retrieval:

1. Frozen embedding lọc nhanh top-k từ pool.
2. TACS selector rerank top-k theo utility downstream.

Giả thuyết: giữ được phần lớn lợi ích của TACS trong khi giảm chi phí tính selector khi pool tăng.

## 5. Thiết kế thực nghiệm

Dataset sử dụng là CIFAR-10 gồm 50.000 ảnh train và 10.000 ảnh test. Tập train được chia phân tầng theo lớp bằng seed 42 thành 36.000 query train, 9.000 candidate pool và 5.000 validation. Candidate pool tách rời query train và validation để tránh rò rỉ dữ liệu. Với mỗi query, pipeline hiện lấy 8 candidates theo cơ chế ngẫu nhiên có seed và thay đổi theo epoch nhưng vẫn tái lập được.

Bảng phân bố chi tiết 10 lớp trên các tập dữ liệu:

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

Kiểm chứng rò rỉ dữ liệu (No-leakage validation):
- $I_{query} \cap I_{candidate} = \emptyset$ (kích thước tập giao = 0).
- $I_{query} \cap I_{val} = \emptyset$ (kích thước tập giao = 0).
- $I_{candidate} \cap I_{val} = \emptyset$ (kích thước tập giao = 0).
- $I_{query} \cup I_{candidate} \cup I_{val} = \{0, 1, \dots, 49.999\}$ (bao phủ chính xác 100% tập train ban đầu).
- Tập test $10.000$ mẫu hoàn toàn cách ly, không tham gia vào candidate pool hay huấn luyện.

Các thử nghiệm hiện chạy trên CPU. Backbone, batch size huấn luyện, learning rate và số epoch sẽ được chốt khi triển khai baseline huấn luyện thật.

## 6. Kết quả

### 6.1 Nghiệm thu pipeline tuần 1

Project được cài lại thành công trong một môi trường Python sạch. Toàn bộ 8 unit test đều đạt, bao gồm kiểm tra split, candidate sampling, kích thước tensor, forward pass, gradient từ hybrid loss về Selector và trực quan hóa. Smoke training chạy đủ 5 bước với loss hữu hạn.

Benchmark DataLoader trong môi trường sạch trên CPU xử lý 3.200 query thuộc 50 batch trong 7,506 giây, tương đương 426,33 query/giây. Tiến trình sử dụng 773,31 MB RAM sau khi nạp CIFAR-10 và không tăng RSS trong quá trình duyệt 50 batch. Một lần chạy sau khi cache hệ điều hành được làm nóng đạt 1.280,56 query/giây, cho thấy kết quả tốc độ phụ thuộc đáng kể vào trạng thái cache. Vì vậy, các so sánh hiệu năng sau này phải dùng cùng quy trình warm-up và lấy trung bình nhiều lần chạy.

### 6.2 Kiểm tra trực quan và tích hợp TACS (Người 3)

1. **Tích hợp DataLoader với TACSModel:**
   - Đã kiểm tra một batch thực tế `[B=4, C=3, H=32, W=32]` và candidates `[B=4, N=8, C=3, H=32, W=32]` đi qua `TACSModel`.
   - Selector cho ra ma trận trọng số one-hot hợp lệ (tổng trọng số trên 8 ứng viên bằng 1.0 cho mỗi query).
   - Hàm mất mát kết hợp `TACSLoss` tính toán đầy đủ `task_loss`, `policy_loss` và `reward`. Backward pass thành công, 100% tham số Selector (6/6 tensors) và Classifier (4/4 tensors) đều nhận gradient khác 0.
   - Ở chế độ suy luận (`eval()`), Selector đưa ra lựa chọn tất định (`deterministic`), đảm bảo tính tái lập khi đánh giá.

2. **Kiểm tra trực quan 20 query:**
   - Tiến hành audit trực quan 20 queries ngẫu nhiên với 160 ảnh candidate tương ứng qua `scripts/inspect_20_queries.py`.
   - Kết quả: 0/20 query bị trùng lặp index với candidate pool (bảo đảm tính cô lập dữ liệu). Toàn bộ 100% ảnh sau khi denormalize đều nằm trọn vẹn trong dải `[0.00, 1.00]`, không phát hiện hiện tượng bão hòa pixel, clipping dị thường hay giá trị NaN/Inf.
   - Tỷ lệ ứng viên cùng nhãn với query là 8.12% và khác nhãn là 91.88%, hoàn toàn phù hợp với kỳ vọng phân bố đều ngẫu nhiên trên 10 lớp (10% cùng lớp, 90% khác lớp).

Chưa có kết quả accuracy từ quá trình huấn luyện thật; đây là mục tiêu của giai đoạn baseline tiếp theo.

## 7. Thảo luận

1. **Ý nghĩa của ngữ cảnh khác lớp:**
   Trong truy xuất truyền thống, mục tiêu luôn là tìm kiếm ảnh cùng lớp hoặc có cosine similarity lớn nhất. Tuy nhiên, kết quả audit trực quan và lý thuyết TACS cho thấy tỷ lệ candidate khác lớp trong pool ngẫu nhiên lên đến ~90%. Điều này mở ra khả năng cho Selector học các mẫu tương phản ranh giới (boundary contrast), cung cấp thông tin loại trừ hữu ích cho downstream classifier.

2. **Bài toán nút thắt cổ chai tính toán (Computational Bottleneck):**
   Với candidate pool 9.000 mẫu, việc đưa toàn bộ pool qua Selector ở mỗi bước là bất khả thi trong điều kiện tài nguyên hạn chế. Do đó, định hướng phát triển **Two-Stage Task-Aligned Retrieval** (lọc thô $k=32$ bằng embedding đóng băng trước, sau đó rerank bằng TACS) là chìa khóa then chốt để áp dụng mô hình vào thực tế.

3. **Tính toàn vẹn dữ liệu:**
   Việc áp dụng seed cố định và kiểm tra tập hợp giao bằng rỗng ($\emptyset$) đảm bảo mọi so sánh hiệu năng giữa các baseline sau này đều diễn ra trên cùng một nền tảng dữ liệu công bằng, không bị sai lệch bởi rò rỉ thông tin.

## 8. Kết luận

Tuần 1 đã hoàn thành xuất sắc toàn bộ mục tiêu nền móng:
- Môi trường thực thi đồng nhất, có cơ chế fallback CPU an toàn.
- Dữ liệu CIFAR-10 được chuẩn hóa, phân tầng cân bằng 10 lớp và chứng minh toán học $0\%$ data leakage.
- Skeleton TACS và hàm mất mát kết hợp hoạt động hoàn hảo, gradient lan truyền ngược thành công về toàn bộ tham số Selector và Classifier.
- Hệ sinh thái kiểm thử đạt 100% (8/8 unit tests pass).

Dự án đủ điều kiện bước sang Tuần 2 để triển khai huấn luyện 3 baseline đối sánh (**No-context**, **Random-context**, **Similarity-context**) và mô hình TACS hoàn chỉnh.

*(Chi tiết báo cáo toàn diện của đồ án xem tại [BAO_CAO_TONG_THE_DU_AN.md](file:///e:/BTL-XLA/docs/BAO_CAO_TONG_THE_DU_AN.md) và báo cáo tiến độ Tuần 1 tại [BAO_CAO_TIEN_DO_TUAN_1.md](file:///e:/BTL-XLA/docs/BAO_CAO_TIEN_DO_TUAN_1.md)).*
