# BÁO CÁO PHÂN CÔNG CÔNG VIỆC TUẦN 1

## 1. Chủ đề

**Chuẩn hóa môi trường và xây dựng pipeline dữ liệu CIFAR-10**

- Nhân sự: 3 thành viên.
- Thời gian: 1 tuần.
- Khối lượng dự kiến: 12-15 giờ/người.

Mục tiêu tuần 1 là tạo nền móng dữ liệu và môi trường ổn định để bước sang huấn luyện baseline. Cả ba thành viên phải tham gia code, kiểm thử, review và tài liệu hóa.

## 2. Kết quả cần đạt

1. Môi trường Python/PyTorch thống nhất trên ba máy.
2. CIFAR-10 được tải và tiền xử lý đúng.
3. Query-training, candidate pool, validation và test được tách biệt.
4. Candidate sampling hoạt động và có thể tái lập.
5. Không có data leakage.
6. Một batch dữ liệu đi qua skeleton TACS thành công.
7. Có unit test, trực quan hóa và tài liệu hướng dẫn.

## 3. Quy ước dữ liệu

CIFAR-10 có 50.000 ảnh train, 10.000 ảnh test, 10 lớp và kích thước RGB 32x32.

| Tập | Số lượng dự kiến | Mục đích |
|---|---:|---|
| Query-training | 36.000 | Huấn luyện mô hình |
| Candidate pool | 9.000 | Cung cấp context |
| Validation | 5.000 | Chọn mô hình và siêu tham số |
| Test | 10.000 | Đánh giá cuối cùng |

Cấu hình ban đầu: `candidate_ratio=0.20`, `validation_size=5000`, `seed=42`, `num_candidates=8`.

## 4. Nguyên tắc phân công

- Mỗi gói công việc có người thực hiện, người review/chạy lại và người tài liệu hóa.
- Vai trò được luân phiên để cả ba đều chạm vào toàn bộ quy trình.
- Code chỉ hoàn thành khi một người khác chạy lại thành công.
- Mọi quyết định về split, seed, transform và tham số phải được ghi lại.

## 5. Phân công theo gói công việc

### Gói 1 - Chuẩn hóa môi trường

**Thời gian dự kiến:** 6-8 giờ công.

| Thành viên | Nhiệm vụ |
|---|---|
| Người 1 | Cập nhật dependency, thêm torchvision, viết cơ chế nhận CPU/GPU |
| Người 2 | Cài project từ đầu trên máy khác; kiểm tra Python, PyTorch, CUDA và GPU |
| Người 3 | Viết hướng dẫn cài đặt, lệnh kiểm tra và các lỗi thường gặp |

**Đầu ra:** File dependency, script kiểm tra môi trường và hướng dẫn cài đặt.

**Nghiệm thu:** Cả ba máy import được project, chạy được test và không phải sửa đường dẫn thủ công.

### Gói 2 - Xây dựng CIFAR-10 DataLoader

**Thời gian dự kiến:** 9-11 giờ công.

| Thành viên | Nhiệm vụ |
|---|---|
| Người 2 | Viết loader, transform train và transform validation/test |
| Người 3 | Viết test cho kích thước ảnh, label, batch và augmentation |
| Người 1 | Review API, tích hợp loader với skeleton TACS và bổ sung ví dụ sử dụng |

Transform train gồm random crop, horizontal flip, chuyển tensor và normalize. Validation/test không dùng augmentation ngẫu nhiên.

**Đầu ra:** Module CIFAR-10, DataLoader và test tương ứng.

**Nghiệm thu:** Batch có shape `[B,3,32,32]`, label trong `0-9` và đi qua encoder không lỗi.

### Gói 3 - Chia dữ liệu và tạo candidate pool

**Thời gian dự kiến:** 10-12 giờ công.

| Thành viên | Nhiệm vụ |
|---|---|
| Người 3 | Viết thuật toán chia dữ liệu phân tầng theo lớp và lưu index |
| Người 1 | Viết test chống trùng và rò rỉ giữa query, candidate, validation và test |
| Người 2 | Review cách chia, kiểm tra phân bố lớp và viết thống kê vào báo cáo |

**Đầu ra:** Các tập index cố định theo seed và bảng thống kê dữ liệu.

**Nghiệm thu:** Query-training, candidate pool và validation không giao nhau; test không tham gia huấn luyện.

### Gói 4 - Candidate sampler và tích hợp TACS

**Thời gian dự kiến:** 9-11 giờ công.

| Thành viên | Nhiệm vụ |
|---|---|
| Người 3 | Viết sampler lấy `N` candidates cho mỗi query; ban đầu `N=8` |
| Người 1 | Test index, shape, seed, query trùng candidate và batch cuối |
| Người 2 | Ghép sampler với DataLoader và đưa một batch qua TACSModel |

Kích thước tensor dự kiến:

```text
query:      [B, 3, 32, 32]
candidates: [B, N, 3, 32, 32]
target:     [B]
```

**Đầu ra:** Candidate sampler tái lập được và pipeline `DataLoader -> TACSModel`.

**Nghiệm thu:** Tensor đúng shape, candidate chỉ thuộc pool và một forward pass chạy thành công.

### Gói 5 - Kiểm thử và trực quan hóa

**Thời gian dự kiến:** 7-9 giờ công.

| Thành viên | Nhiệm vụ |
|---|---|
| Người 1 | Chạy unit test, kiểm tra gradient, tốc độ và bộ nhớ DataLoader |
| Người 2 | Viết script hiển thị query, candidates, label và index |
| Người 3 | Kiểm tra trực quan ít nhất 20 query và ghi nhận lỗi |

**Đầu ra:** Test suite, hình minh họa batch và danh sách lỗi.

**Nghiệm thu:** Tất cả test đạt và không phát hiện sai label, sai transform hoặc rò rỉ dữ liệu.

### Gói 6 - Review chéo và tài liệu hóa

**Thời gian dự kiến:** 6 giờ công.

| Thành viên | Nhiệm vụ |
|---|---|
| Người 1 | Chạy lại project từ môi trường sạch, xác minh README |
| Người 2 | Chạy toàn bộ test và tổng hợp thống kê dữ liệu |
| Người 3 | Cập nhật báo cáo, tiến độ và danh sách vấn đề |
| Cả nhóm | Mỗi người trình bày một module không do mình viết |

**Đầu ra:** README hoàn chỉnh, báo cáo dữ liệu và biên bản nghiệm thu.

## 6. Phân bổ thời gian theo thành viên

| Thành viên | Code | Test/review | Tài liệu/họp | Tổng |
|---|---:|---:|---:|---:|
| Người 1 | 5 giờ | 6 giờ | 3 giờ | 14 giờ |
| Người 2 | 7 giờ | 4 giờ | 3 giờ | 14 giờ |
| Người 3 | 7 giờ | 4 giờ | 3 giờ | 14 giờ |

## 7. Sản phẩm bàn giao cuối tuần

- Module CIFAR-10 và transform.
- Module chia dữ liệu.
- Candidate pool và candidate sampler.
- File cấu hình dữ liệu.
- Unit test chống data leakage.
- Script trực quan hóa batch.
- Pipeline tích hợp với skeleton TACS.
- README, báo cáo và bảng tiến độ cập nhật.

## 8. Điều kiện chuyển sang tuần 2

- Cả ba thành viên chạy được pipeline.
- Toàn bộ data test đạt.
- Không có giao nhau giữa các tập.
- Candidate sampling tái lập được.
- Một batch đi qua skeleton TACS thành công.
- Báo cáo mô tả rõ nguồn dữ liệu, transform và cách chia tập.

Sau khi nghiệm thu, tuần 2 sẽ triển khai ba baseline: **No-context**, **Random-context** và **Similarity-context**.
