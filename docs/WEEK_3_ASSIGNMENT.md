# KẾ HOẠCH HỢP NHẤT TUẦN 3 & TUẦN 4: VỀ ĐÍCH BÀI TẬP LỚN
## Đề tài: Tái hiện và Cải tiến Task-Aligned Context Selection (TACS) trên CIFAR-10
**Học phần:** Xử lý ảnh & Thị giác máy tính (BTL-XLA)  
**Thời gian thực hiện:** Giai đoạn nước rút (Hợp nhất Milestone M3 - M4 - M5)  
**Nhân sự tham gia:** Cả 3 thành viên (Người 1, Người 2, Người 3)

---

## 1. MỤC TIÊU CỐT LÕI CỦA GIAI ĐOẠN GỘP

1. **Tái hiện TACS Full (Milestone M3):** Huấn luyện mô hình TACS kết hợp mạng chọn mẫu (Selector) với Straight-Through Gumbel-Softmax và hàm mất mát tăng cường Policy Gradient (`TACSLoss`).
2. **Cải tiến Two-Stage TACS (Milestone M4):** Đề xuất giải pháp Hai giai đoạn:
   - *Giai đoạn 1 (Lọc thô):* Dùng Cosine Similarity lọc lấy Top-32 ảnh ứng viên gần nhất.
   - *Giai đoạn 2 (Tái xếp hạng):* Dùng TaskAlignedSelector chọn ra 1 ảnh tối ưu nhất từ Top-32 để đưa vào bộ phân loại.
3. **Đánh giá toàn diện 5 mô hình (Milestone M5):** Đối sánh toàn bộ 5 mô hình trên 3 trục:
   - **Độ chính xác (Accuracy %)**
   - **Tốc độ suy luận (Latency ms / Throughput q/s)**
   - **Bộ nhớ tiêu thụ (Peak RAM / VRAM MB)**
4. **Bàn giao đồ án:** Hoàn thiện Báo cáo tổng thể, Bộ biểu đồ trực quan, Slide thuyết trình và Kịch bản Demo.

---

## 2. BẢNG PHÂN CÔNG NHIỆM VỤ CHI TIẾT (3 THÀNH VIÊN)

| Thành viên | Vai trò phụ trách | Gói công việc cụ thể | Sản phẩm bàn giao (Deliverables) |
|---|---|---|---|
| **Người 1** | **Trưởng nhóm / Kỹ thuật & Thuật toán** *(Engineering & Algorithm)* | **Gói 1: Mở rộng Trainer & Two-Stage Pipeline**<br>1. Nâng cấp `src/tacs/train.py` hỗ trợ mode `--baseline tacs` sử dụng `TACSModel` và `TACSLoss`.<br>2. Xây dựng module Two-Stage TACS (tích hợp lọc Top-K Cosine + Selector) và hỗ trợ mode `--baseline two_stage`.<br>3. Tối ưu hóa bộ nhớ đệm (cache) để pipeline chạy mượt mà trên cả CPU và GPU. | - `src/tacs/train.py` (đã nâng cấp 2 mode mới)<br>- Module Two-Stage tích hợp<br>- Hướng dẫn CLI chạy train 2 mô hình mới |
| **Người 2** | **Kiểm thử & Thực nghiệm** *(Testing & Experiments)* | **Gói 2: Unit Test Suite & Chạy Huấn luyện**<br>1. Viết test tự động cho Two-Stage Pipeline và vòng lặp TACS trong `tests/`.<br>2. Chạy huấn luyện thực tế cho 2 mô hình: TACS Full và Two-Stage TACS.<br>3. Đo đạc chính xác 3 chỉ số: Accuracy, Latency (ms) và RAM chiếm dụng (MB) cho cả 5 mô hình.<br>4. Đóng gói các file trọng số `checkpoint_best.pt` và log `metrics.json`. | - Test suite mới đạt 100% pass<br>- Checkpoints: `outputs/tacs/` và `outputs/two_stage/`<br>- File log thông số `metrics.json` đầy đủ của 5 mô hình |
| **Người 3** | **Phân tích, Trực quan hóa & Báo cáo** *(Analysis & Documentation)* | **Gói 3: Biểu đồ đối sánh, Báo cáo & Slide**<br>1. Nâng cấp script phân tích thành `scripts/analyze_all_models.py` để vẽ biểu đồ tổng hợp 5 mô hình (Accuracy vs Latency vs RAM).<br>2. Cập nhật số liệu thực nghiệm vào Báo cáo tổng kết `docs/BAO_CAO_TONG_THE_DU_AN.md` (Chương 3 cải tiến & Chương 5 kết quả).<br>3. Thiết kế Slide thuyết trình PowerPoint/Canva và chuẩn bị kịch bản Demo trả lời vấn đáp. | - Script `scripts/analyze_all_models.py`<br>- Ảnh biểu đồ tổng hợp `docs/all_models_comparison.png`<br>- File Báo cáo tổng kết hoàn chỉnh<br>- Bộ Slide thuyết trình bảo vệ BTL |

---

## 3. QUY TRÌNH PHỐI HỢP & TIẾN ĐỘ THỰC HIỆN

Quy trình thực hiện theo chuỗi tiếp sức (Pipeline Workflow):

```
[Người 1: Code lõi TACS & Two-Stage]
                │
                ▼
[Người 2: Viết Test + Chạy Train thu thập Metrics 5 mô hình]
                │
                ▼
[Người 3: Vẽ biểu đồ so sánh + Hoàn thiện Báo cáo & Slide]
```

### Lộ trình thời gian đề xuất (3 - 5 ngày hoàn thành):
* **Ngày 1 (Xong Gói 1):** Người 1 hoàn thành nâng cấp `train.py` và module Two-Stage.
* **Ngày 2 (Xong Gói 2):** Người 2 kiểm thử test suite và chạy train thu thập trọn vẹn số liệu của cả 5 mô hình.
* **Ngày 3 (Xong Gói 3):** Người 3 chạy script vẽ biểu đồ, cập nhật báo cáo và chốt Slide bảo vệ.
* **Ngày 4:** Toàn nhóm họp thử (Mock Presentation), chạy demo kiểm tra trước ngày nộp.
