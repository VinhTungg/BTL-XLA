# Mục lục tài liệu

Luồng hiện hành: **CIFAR-100 và Caltech-101, demo bằng checkpoint thật**.
Tài liệu cũ vẫn được giữ để truy vết, không có nghĩa mọi kế hoạch đã hoàn thành.

## Demo và thuyết trình

- [Demo Caltech-101](CALTECH101_DEMO_GUIDE.md): lệnh chạy, ví dụ giúp/hại và hai loại No-context.
- [Demo CIFAR-100](SHOWCASE_GUIDE.md): checkpoint microbatch/effective-batch đã audit.
- [Kịch bản slide](SLIDE_SCRIPT.md): 15 slide chính, 6 slide dự phòng, lời nói và hỏi đáp.
- [Caltech full100](CALTECH101_FULL100_RESULTS.md): kết quả, split, source và hash artifact.
- [Cleanup project](PROJECT_CLEANUP.md): những gì đã dọn và bằng chứng cần giữ.

## Thực nghiệm CIFAR-100 và audit

| Chủ đề | Tài liệu |
|---|---|
| Chẩn đoán và sửa vai trò query/context | [CONTEXT_ROLE_FIX](CONTEXT_ROLE_FIX.md) |
| Audit gradient | [GRADIENT_AUDIT](GRADIENT_AUDIT.md) |
| Audit utility của selector | [SELECTOR_UTILITY_AUDIT](SELECTOR_UTILITY_AUDIT.md) |
| Pilot policy weight | [Thiết kế pilot](POLICY_WEIGHT_PILOT.md) |
| Policy weight 1,0 / 0,1 / 0,03 | [w1](POLICY_W1_SELECTOR_AUDIT.md), [w0,1](POLICY_W01_SELECTOR_AUDIT.md), [w0,03](POLICY_W003_SELECTOR_AUDIT.md) |
| Triển khai effective-batch | [Thiết kế kỹ thuật](EFFECTIVE_BATCH_IMPLEMENTATION.md) |
| Pilot chuẩn hóa advantage | [Protocol](EFFECTIVE_BATCH_PILOT.md), [hướng dẫn Colab](COLAB_ADVANTAGE_PILOT.md) |
| Kết quả paired pilot seed44 | [EFFECTIVE_BATCH_SEED44_RESULTS](EFFECTIVE_BATCH_SEED44_RESULTS.md) |
| Audit hai selector trên 5.000 validation | [EFFECTIVE_BATCH_SELECTOR_AUDIT](EFFECTIVE_BATCH_SELECTOR_AUDIT.md) |
| Thử cosine lúc inference | [EFFECTIVE_BATCH_COSINE_AUDIT](EFFECTIVE_BATCH_COSINE_AUDIT.md) |
| TACS seed42, bản triển khai lịch sử | [CIFAR100_TACS_SEED42_RESULTS](CIFAR100_TACS_SEED42_RESULTS.md) |

Không gộp khác version task network, split, seed, ngân sách epoch hoặc loại
đánh giá vào một bảng thắng/thua. Ví dụ demo không thay cho số liệu toàn tập.

## Cloud, kế hoạch và quyết định

- [Caltech Colab](CALTECH101_COLAB_GUIDE.md): thiết lập và resume lịch 100 epoch.
- [Quyết định kỹ thuật](DECISIONS.md): có lịch sử quyết định và các ADR thay thế nhau.
- [Roadmap tái hiện đến demo](ROADMAP_REPRODUCTION_TO_DEMO.md): kế hoạch lập 30/10/2026,
  không phải bản xác nhận mọi hạng mục đã làm xong.
- [Kế hoạch demo Caltech](CALTECH101_DEMO_PLAN.md): thiết kế trước khi thu hồi kết quả;
  dùng hướng dẫn/kết quả hiện hành ở phần đầu để biết trạng thái mới nhất.

## Tài liệu lịch sử CIFAR-10 và giai đoạn đầu

Các tài liệu sau ghi nhận công việc tại thời điểm cũ. Không lấy các con số
tiny/CIFAR-10, lịch 5 epoch hoặc tình trạng “đã xong” ở đó làm kết quả hiện tại
của CIFAR-100/DINOv3 hoặc Caltech-101:

- [Kế hoạch ban đầu](PROJECT_PLAN.md), [tiến độ ban đầu](PROGRESS.md).
- [Báo cáo kỹ thuật ban đầu](REPORT.md), [báo cáo tổng thể giai đoạn đầu](BAO_CAO_TONG_THE_DU_AN.md).
- [Phân công tuần 1](WEEK_1_ASSIGNMENT.md), [tuần 2](WEEK_2_ASSIGNMENT.md), [tuần 3](WEEK_3_ASSIGNMENT.md).
- [Báo cáo tuần 1](BAO_CAO_TIEN_DO_TUAN_1.md), [tuần 2](BAO_CAO_TIEN_DO_TUAN_2.md), [tuần 3](BAO_CAO_TIEN_DO_TUAN_3.md).
- [Ablation CIFAR-10 lịch sử](ABLATION_RESULTS.md).
- [Hướng dẫn demo policy-weight cũ](DEMO_GUIDE.md), đã có chỉ dẫn sang showcase hiện hành.

Source huấn luyện/demo, script xây notebook và các bản kết quả được giữ nguyên
trong cleanup này. Hai test xây notebook được cô lập đầu ra trong bộ nhớ.
Giữ đường dẫn cũ giúp không phá link hoặc provenance của thí nghiệm.
