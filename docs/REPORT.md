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

Các thử nghiệm hiện chạy trên CPU. Backbone, batch size huấn luyện, learning rate và số epoch sẽ được chốt khi triển khai baseline huấn luyện thật.

## 6. Kết quả

### 6.1 Nghiệm thu pipeline tuần 1

Project được cài lại thành công trong một môi trường Python sạch. Toàn bộ 7 unit test đều đạt, bao gồm kiểm tra split, candidate sampling, kích thước tensor, forward pass và gradient từ hybrid loss về Selector. Smoke training chạy đủ 5 bước với loss hữu hạn.

Benchmark DataLoader trong môi trường sạch trên CPU xử lý 3.200 query thuộc 50 batch trong 7,506 giây, tương đương 426,33 query/giây. Tiến trình sử dụng 773,31 MB RAM sau khi nạp CIFAR-10 và không tăng RSS trong quá trình duyệt 50 batch. Một lần chạy sau khi cache hệ điều hành được làm nóng đạt 1.280,56 query/giây, cho thấy kết quả tốc độ phụ thuộc đáng kể vào trạng thái cache. Vì vậy, các so sánh hiệu năng sau này phải dùng cùng quy trình warm-up và lấy trung bình nhiều lần chạy.

Chưa có kết quả accuracy từ quá trình huấn luyện thật; đây là mục tiêu của giai đoạn baseline tiếp theo.

## 7. Thảo luận

## 8. Kết luận
