# Tiến độ

Cập nhật: 2026-09-11

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Đọc và phân tích paper | Xong | Bản Anh là nguồn chuẩn |
| Kiểm kê source | Xong | Không có code TACS chính thức trong workspace |
| Phân tích dự án tham khảo | Xong | CLIP + Weaviate; không tái sử dụng cho lõi training |
| Skeleton TACS | Xong | Selector, classifier, hybrid loss |
| Unit test CPU | Xong | 3/3 test pass; smoke train chạy end-to-end |
| Giải thích TACS trong báo cáo | Xong | Có ví dụ, luồng xử lý, Gumbel và reward |
| Dataset + baseline | Chưa làm | M1 |
| Reproduction | Chưa làm | M2 |
| Cải tiến two-stage | Chưa làm | M3 |

## Việc tiếp theo

1. Chốt GPU và deadline.
2. Thêm CIFAR loader/candidate pool và cấu hình thí nghiệm.
3. Chạy no-context baseline trước.
