# BÁO CÁO PHÂN CÔNG CÔNG VIỆC TUẦN 2

## 1. Chủ đề

**Xây dựng, huấn luyện và so sánh ba baseline trên CIFAR-10**

- Nhân sự: 3 thành viên.
- Thời gian: 1 tuần.
- Khối lượng dự kiến: 14-15 giờ/người.
- Baseline bắt buộc: No-context, Random-context và Similarity-context.

Tuần 2 hoàn thành mốc M1 của dự án. Ba baseline phải dùng chung dữ liệu, backbone, seed, ngân sách huấn luyện và cách đánh giá để kết quả có thể so sánh công bằng.

## 2. Điều kiện bắt đầu

- Pipeline CIFAR-10 và split tuần 1 đã được giữ nguyên.
- Toàn bộ unit test hiện tại đạt.
- Ba thành viên chạy được project từ README.
- Người 2 và Người 3 hoàn tất phần trực quan hóa và kiểm tra dữ liệu còn lại của tuần 1 trước khi chạy huấn luyện chính thức.

## 3. Nguyên tắc bàn giao nối tiếp

Mỗi gói đi theo chuỗi bắt buộc:

`Người xây dựng → Người kiểm thử/tích hợp → Người chạy lại/nghiệm thu → Gói tiếp theo`

Người tiếp theo chỉ nhận việc khi người trước bàn giao đủ:

1. Source code và tên commit.
2. Lệnh chạy chính xác.
3. Test hoặc kiểm tra tối thiểu đã đạt.
4. File đầu ra cần dùng ở bước tiếp theo.
5. Ghi chú lỗi hoặc giới hạn còn tồn tại.

Nếu bàn giao thiếu một trong năm mục trên, công việc chưa được xem là hoàn thành và người sau chưa chạy thí nghiệm chính thức.

## 4. Cấu hình chung phải giữ cố định

| Thành phần | Quy ước tuần 2 |
|---|---|
| Dataset | CIFAR-10 |
| Split | 36.000 query train, 9.000 candidate, 5.000 validation, 10.000 test |
| Seed chính | 42 |
| Backbone | `TinyVisionEncoder` hiện tại |
| Số lớp | 10 |
| Candidate cho Random | 1 ảnh lấy từ candidate pool bằng seed cố định |
| Candidate cho Similarity | 1 ảnh có cosine similarity cao nhất |
| Chọn mô hình | Theo validation accuracy |
| Metric | Loss, accuracy, thời gian/epoch, query/giây và RAM |
| Kết quả lưu | Config, checkpoint tốt nhất và JSON metric |

Trong quá trình phát triển chỉ dùng train và validation. Test set chỉ được chạy sau khi cấu hình của cả ba baseline đã chốt.

## 5. Chuỗi công việc chi tiết

### Gói 1 - Hạ tầng thí nghiệm dùng chung

**Thời gian dự kiến:** 9 giờ công.

#### Bước 1.1 - Người 1 xây dựng (4 giờ)

- Tạo cấu hình chung cho seed, batch size, epoch, learning rate, device và đường dẫn output.
- Viết vòng lặp train/evaluate dùng lại được cho cả ba baseline.
- Ghi metric từng epoch và lưu checkpoint có validation accuracy tốt nhất.
- Tạo CLI để chọn `no_context`, `random` hoặc `similarity`.

**Bàn giao cho Người 2:** source code, một lệnh dry-run, file JSON metric mẫu và mô tả format checkpoint.

#### Bước 1.2 - Người 2 kiểm thử (3 giờ)

- Viết test cho parser/config, accuracy và lưu/đọc checkpoint.
- Kiểm tra cùng seed cho cùng kết quả ở một dry-run ngắn.
- Kiểm tra validation không tham gia cập nhật gradient.

**Bàn giao cho Người 3:** test đã đạt, danh sách lỗi đã sửa và lệnh chạy trên CPU.

#### Bước 1.3 - Người 3 tích hợp và nghiệm thu (2 giờ)

- Chạy lại từ môi trường sạch.
- Kiểm tra thư mục output, log và checkpoint được tạo đúng.
- Bổ sung hướng dẫn chạy baseline vào README.

**Điều kiện mở Gói 2:** dry-run train/evaluate chạy end-to-end và tạo được metric JSON hợp lệ.

### Gói 2 - No-context baseline

**Thời gian dự kiến:** 9 giờ công.

#### Bước 2.1 - Người 2 xây dựng và huấn luyện (4 giờ)

- Cài đặt nhánh chỉ dùng query embedding; context embedding được thay bằng vector 0.
- Huấn luyện trên query-training và chọn checkpoint theo validation accuracy.
- Lưu checkpoint encoder chuẩn để Similarity-context sử dụng về sau.

**Bàn giao cho Người 3:** checkpoint tốt nhất, config, metric JSON và log huấn luyện.

#### Bước 2.2 - Người 3 đánh giá (3 giờ)

- Load lại checkpoint trong tiến trình mới.
- Chạy validation, kiểm tra accuracy khớp file metric.
- Ghi thời gian/epoch, throughput và RAM.

**Bàn giao cho Người 1:** bảng kết quả và checkpoint đã được xác minh.

#### Bước 2.3 - Người 1 review và khóa encoder chuẩn (2 giờ)

- Review đường dữ liệu để chắc chắn không dùng candidate và test set.
- Kiểm tra checkpoint có thể trích xuất embedding.
- Ghi hash/tên checkpoint encoder chuẩn vào nhật ký quyết định.

**Điều kiện mở Gói 3:** No-context tái lập được và encoder chuẩn đã được khóa.

### Gói 3 - Random-context baseline

**Thời gian dự kiến:** 8 giờ công.

#### Bước 3.1 - Người 3 xây dựng (3,5 giờ)

- Dùng `ContextDataset` để lấy đúng một candidate ngẫu nhiên cho mỗi query.
- Bảo đảm candidate chỉ đến từ candidate pool.
- Tích hợp vào vòng lặp train/evaluate chung, không sao chép trainer.

**Bàn giao cho Người 1:** source code, lệnh dry-run và danh sách candidate index mẫu.

#### Bước 3.2 - Người 1 kiểm thử (2,5 giờ)

- Kiểm tra cùng seed cho cùng candidate; đổi seed cho candidate khác.
- Kiểm tra không có index ngoài pool hoặc giao với validation/test.
- Chạy test gradient và một epoch ngắn.

**Bàn giao cho Người 2:** test đã đạt và checkpoint dry-run.

#### Bước 3.3 - Người 2 huấn luyện và ghi kết quả (2 giờ)

- Chạy cấu hình chính thức với cùng ngân sách No-context.
- Lưu checkpoint, metric và tài nguyên sử dụng.
- So sánh sơ bộ với No-context nhưng chưa kết luận trên test set.

**Điều kiện mở Gói 4:** Random-context tái lập được và có validation metric đầy đủ.

### Gói 4 - Similarity-context baseline

**Thời gian dự kiến:** 11 giờ công.

#### Bước 4.1 - Người 1 xây dựng retrieval (4 giờ)

- Load encoder chuẩn đã khóa từ Gói 2.
- Tiền tính và lưu embedding của candidate pool.
- Với mỗi query, chuẩn hóa embedding và chọn candidate có cosine similarity cao nhất.
- Ghi candidate index và similarity score để có thể kiểm tra lại.

**Bàn giao cho Người 2:** file embedding/cache, API retrieval, lệnh tạo lại cache và ví dụ query→candidate.

#### Bước 4.2 - Người 2 kiểm thử (3 giờ)

- So sánh kết quả retrieval với phép tính cosine thủ công trên mẫu nhỏ.
- Kiểm tra cache gắn đúng split, checkpoint và embedding dimension.
- Kiểm tra không dùng validation/test làm candidate.
- Đo thời gian retrieval và RAM.

**Bàn giao cho Người 3:** test đã đạt, cache hợp lệ và báo cáo benchmark retrieval.

#### Bước 4.3 - Người 3 tích hợp và huấn luyện (4 giờ)

- Ghép candidate được truy xuất vào trainer chung.
- Huấn luyện với cùng config và ngân sách hai baseline trước.
- Lưu checkpoint, metric, latency và RAM.

**Điều kiện mở Gói 5:** Similarity-context chạy end-to-end và có validation metric tái lập được.

### Gói 5 - So sánh, test cuối và nghiệm thu

**Thời gian dự kiến:** 6 giờ công.

#### Bước 5.1 - Người 2 tổng hợp (2 giờ)

- Kiểm tra ba run dùng cùng split, seed, backbone và ngân sách.
- Tạo bảng No-context/Random/Similarity gồm accuracy, loss, thời gian và RAM.
- Sau khi cấu hình đã khóa, chạy test set đúng một lần cho mỗi baseline.

**Bàn giao cho Người 3:** bảng kết quả cuối và ba file metric gốc.

#### Bước 5.2 - Người 3 phân tích (2 giờ)

- Tạo biểu đồ loss/accuracy theo epoch.
- Chọn ví dụ Random và Similarity giúp hoặc làm hại dự đoán.
- Viết nhận xét dựa trên số liệu, không khẳng định vượt trội nếu chênh lệch nhỏ.

**Bàn giao cho Người 1:** hình, bảng và bản nháp phần kết quả.

#### Bước 5.3 - Người 1 nghiệm thu (2 giờ)

- Chạy lại một cấu hình từ README trong môi trường sạch.
- Chạy toàn bộ test và kiểm tra file kết quả có thể truy ngược về config/checkpoint.
- Cập nhật `REPORT.md`, `PROGRESS.md`, `DECISIONS.md` và chốt M1.

## 6. Luồng bàn giao toàn tuần

```text
Người 1: trainer/config
    ↓
Người 2: test hạ tầng → No-context + encoder chuẩn
    ↓
Người 3: xác minh No-context → Random-context
    ↓
Người 1: test Random → Similarity retrieval
    ↓
Người 2: test retrieval → tổng hợp ba baseline
    ↓
Người 3: biểu đồ và phân tích
    ↓
Người 1: chạy sạch, cập nhật tài liệu và nghiệm thu M1
```

## 7. Phân bổ thời gian

| Thành viên | Xây dựng | Test/tích hợp | Thí nghiệm/tài liệu | Tổng |
|---|---:|---:|---:|---:|
| Người 1 | 8 giờ | 4,5 giờ | 2 giờ | 14,5 giờ |
| Người 2 | 6 giờ | 6 giờ | 2 giờ | 14 giờ |
| Người 3 | 7,5 giờ | 5 giờ | 2 giờ | 14,5 giờ |

## 8. Sản phẩm bàn giao cuối tuần

- Trainer/evaluator dùng chung và cấu hình thí nghiệm tái lập được.
- No-context checkpoint và encoder chuẩn.
- Random-context baseline.
- Similarity-context baseline và candidate embedding cache.
- Unit test cho metric, checkpoint, random sampling và cosine retrieval.
- Bảng so sánh accuracy, loss, tốc độ và RAM.
- Biểu đồ học và ví dụ phân tích lỗi.
- README, báo cáo, tiến độ và nhật ký quyết định được cập nhật.

## 9. Tiêu chí hoàn thành tuần 2

- Cả ba baseline chạy end-to-end từ cùng một CLI.
- Mỗi baseline có config, checkpoint và metric JSON.
- Kết quả có thể tái lập bằng seed 42.
- Không có data leakage hoặc dùng test để chọn mô hình.
- Cả ba dùng cùng backbone và ngân sách huấn luyện.
- Một thành viên khác với người viết đã chạy lại từng baseline.
- M1 được đánh dấu hoàn thành trong `PROGRESS.md`.

Sau nghiệm thu, tuần 3 chuyển sang TACS reproduction: huấn luyện Selector bằng Gumbel-Softmax, policy reward và ablation hai thành phần loss.
