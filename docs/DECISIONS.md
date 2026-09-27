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

## ADR-004: Khóa encoder từ No-context cho Similarity Retrieval

- Trạng thái: Đã chấp nhận và khóa trong Milestone M2.
- Quyết định: Sau khi huấn luyện Baseline 1 (No-context), trích xuất trọng số của `TinyVisionEncoder` tại `outputs/no_context/best_encoder.pt` làm encoder chuẩn.
- Lý do: Đảm bảo tính công bằng và nhất quán khi tiền tính toán embedding của Candidate Pool ($9.000$ mẫu) và tính toán Cosine Similarity cho Baseline 3, không bị data leakage sang test set.


