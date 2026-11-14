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

## ADR-005: Tập trung demo CIFAR-100, không mở dataset mới — 14/11/2026

**Đã bị thay thế bởi ADR-006.** Agent hiểu nhầm phạm vi người dùng; các dòng
dưới giữ để truy vết, không phải quyết định hiện hành.

- Theo yêu cầu mới của người dùng, dừng kế hoạch mở rộng Caltech-101/Pets.
  Không xóa dữ liệu, checkpoint hoặc kết quả CIFAR-100; không khởi chạy train thêm.
- Hoàn thiện giao diện offline từ các checkpoint advantage seed44 đã audit,
  mặc định microbatch; effective-batch là ablation chưa chứng minh tăng accuracy.
- Tách presentation ở `src/tacs/showcase/`, giữ nguyên code/bundle training cũ.
  Các can thiệp trong demo dùng cùng task weights, không giả làm baseline train riêng.
- Hướng dẫn vận hành và kiểm thử tại [SHOWCASE_GUIDE.md](SHOWCASE_GUIDE.md).

## ADR-006: Giữ CIFAR-100, thêm Caltech-101 để demo ảnh rõ — 14/11/2026

- Người dùng làm rõ: vẫn thêm dataset vì muốn demo bằng ảnh nguồn rõ nét hơn;
  "tập trung demo" không đồng nghĩa bỏ dataset mới. Bộ bổ sung đã chốt từ trước
  là **Caltech-101**, không tự đổi sang Oxford Pets.
- CIFAR-100/checkpoint/audit và demo đã làm được giữ nguyên, không train lại
  CIFAR-100 hoặc ghi đè kết quả. Caltech-101 là nhánh riêng phục vụ demo.
- Cần hỗ trợ loader/split/101 nhãn, checkpoint được huấn luyện riêng và lựa chọn
  dataset trong GUI. Không dùng bộ phân loại CIFAR-100 để giả làm Caltech-101.
- Tái sử dụng DINOv3 pretrained làm điểm khởi tạo; không resume epoch CIFAR.
  Không tự mở thêm vòng quét policy-weight/effective-batch/cosine.
- Trạng thái cập nhật: demo CIFAR-100 đã chạy. Caltech-101 có loader/split/trainer
  riêng và notebook Colab đã setup để người dùng tự chạy; chưa có checkpoint
  train hoặc GUI Caltech đã nghiệm thu. Không tự bắt đầu lượt GPU dài.
- Kế hoạch: [CALTECH101_DEMO_PLAN.md](CALTECH101_DEMO_PLAN.md).

## ADR-007: Caltech demo trên Colab, không thay đổi thí nghiệm CIFAR — 14/11/2026

- Người dùng yêu cầu setup đầy đủ rồi tự chạy, có thể mở cùng notebook/tài khoản
  trên máy khác. Đã lưu đủ7 code cells và hướng dẫn trên notebook Drive; chưa train.
- Tách `src/tacs/caltech/`, tái dùng model/loss/epoch/optimizer đã có. Không sửa
  các file `src/tacs/*.py` đang khóa source hash đối chứng CIFAR.
- Protocol `caltech101_demo_19query_6pool_5val_rest_test_v1`, split seed42,
  train seed44, No-context rồi Two-Stage,20epoch mỗi model. Không gọi là paper
  reproduction đầy đủ, không mở test để chọn model.
- Lưu full checkpoint/latest, optimizer, scheduler, AMP scaler, RNG sau mỗi epoch
  lên Drive; resume chỉ với source/config/split/reference/environment khớp.
- Chưa có accuracy Caltech được train; việc mở rộng dataset phục vụ demo không
  tự chứng minh cải tiến thuật toán.


