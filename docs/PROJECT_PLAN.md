# Kế hoạch toàn dự án

## Phạm vi đã chốt

- MVP: phân loại ảnh, một context cho mỗi query.
- Dataset đầu tiên: CIFAR-10 hoặc CIFAR-100, tùy GPU.
- Baseline bắt buộc: no-context, random, cosine-similarity, TACS.
- Cải tiến chính dự kiến: **two-stage task-aligned retrieval** (lọc top-k rẻ, sau đó selector học rerank) để giảm chi phí khi pool lớn.

## Milestone

| Mốc | Sản phẩm | Tiêu chí hoàn thành |
|---|---|---|
| M0 - Khởi tạo | Skeleton, test, kế hoạch, khung báo cáo | Test CPU chạy qua |
| M1 - Dữ liệu & baseline | Loader, split, candidate pool, no/random/similarity | Có metric và file config tái lập |
| M2 - TACS reproduction | Gumbel + policy reward + train/eval | TACS chạy end-to-end, có ablation hai loss |
| M3 - Cải tiến | Two-stage reranking | So sánh accuracy, latency, memory với TACS |
| M4 - Thực nghiệm | Nhiều seed, bảng/biểu đồ, phân tích lỗi | Kết quả đủ cho báo cáo |
| M5 - Bàn giao | Báo cáo, slide/demo, README | Người khác chạy lại được |

## Quy tắc quản lý

- Mỗi thí nghiệm phải ghi config, seed, metric và thời gian chạy.
- Chỉ khẳng định cải tiến khi so sánh cùng backbone, split và budget.
- Mỗi thay đổi kiến trúc phải có ablation.
- `PROGRESS.md` và `REPORT.md` được cập nhật sau mỗi milestone.

## Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Không có code chính thức | Viết module nhỏ, bám công thức, test từng khối |
| Thiếu GPU | CPU smoke test; dùng backbone/dataset nhỏ trước |
| Pool lớn gây tốn bộ nhớ | Tiền tính embedding và two-stage retrieval |
| Reward nhiễu | Chuẩn hóa advantage, theo dõi entropy, thử EMA baseline |

