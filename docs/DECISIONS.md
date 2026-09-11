# Nhật ký quyết định

## ADR-001: Phân loại là MVP đầu tiên

- Trạng thái: chấp nhận tạm thời.
- Lý do: ít chi phí hơn segmentation, dễ xây baseline và ablation.
- Có thể thay đổi khi biết rõ yêu cầu giảng viên.

## ADR-002: Không fork dự án truy xuất tham khảo

- Dự án tham khảo tối ưu truy vấn CLIP/Weaviate, không có downstream task reward.
- Lõi huấn luyện TACS được viết mới; frontend có thể tham khảo ở M5.

## ADR-003: Cải tiến chính là two-stage retrieval

- Lọc top-k bằng embedding rẻ, sau đó rerank bằng task-aligned selector.
- Đánh giá đồng thời accuracy, latency và memory.

