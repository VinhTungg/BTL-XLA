# Tiến độ

Cập nhật: 2026-09-16

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Đọc và phân tích paper | Xong | Bản Anh là nguồn chuẩn |
| Kiểm kê source | Xong | Không có code TACS chính thức trong workspace |
| Phân tích dự án tham khảo | Xong | CLIP + Weaviate; không tái sử dụng cho lõi training |
| Skeleton TACS | Xong | Selector, classifier, hybrid loss |
| Unit test CPU | Xong | 7/7 test pass; kiểm tra split, sampling, forward và gradient |
| Giải thích TACS trong báo cáo | Xong | Có ví dụ, luồng xử lý, Gumbel và reward |
| Pipeline CIFAR-10 | Xong | Split 36.000/9.000/5.000/10.000; candidate sampling tái lập được |
| Benchmark DataLoader | Xong | Môi trường sạch: 426,33 query/s trên CPU; RSS 773,31 MB |
| Baseline huấn luyện thật | Chưa làm | Bước đầu của M2 |
| Reproduction | Chưa làm | M2 |
| Cải tiến two-stage | Chưa làm | M3 |

## Nghiệm thu Người 1 - Tuần 1

- Cài đặt thành công project trong môi trường `.venv-clean` từ hướng dẫn README.
- 7/7 unit test đạt trong 0,205 giây.
- Smoke training chạy đủ 5 bước; loss hữu hạn và gradient hoạt động.
- DataLoader xử lý 3.200 query trong 7,506 giây trên CPU.
- README đã được cập nhật theo trạng thái thực tế của project.

## Nghiệm thu Người 2 - Tuần 1

- Đã thêm script trực quan hóa query và candidates tại `src/tacs/visualize_batch.py`.
- Đã thêm unit test cho chức năng trực quan hóa tại `tests/test_visualize.py`.
- Unit test kiểm tra lưới gồm query và 8 candidates cho mỗi query, tổng cộng 18 axes với 2 query.
- Script hiển thị label query/candidate và lưu hình minh họa ra file PNG.
- Đã chạy test visualization: `1/1` test đạt.
- Phần kiểm thử dùng dữ liệu tổng hợp; việc tạo ảnh từ CIFAR-10 thật còn phụ thuộc dữ liệu CIFAR-10 đã tải đầy đủ.

## Việc tiếp theo

1. Người 2 hoàn thành trực quan hóa query và candidates.
2. Người 3 kiểm tra trực quan ít nhất 20 query và ghi nhận lỗi.
3. Cả nhóm review chéo, sau đó chạy no-context baseline.
