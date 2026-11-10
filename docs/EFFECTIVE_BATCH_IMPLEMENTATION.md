# Advantage trên effective batch: triển khai và kiểm chứng

Ngày 11/11/2026. Đây là kiểm chứng khác biệt chuẩn hóa, không phải kết quả
accuracy mới hoặc đóng góp nghiên cứu đã được chứng minh.

## Thuật toán

`--advantage-scope microbatch` là mặc định cũ. `effective_batch` áp dụng
cho TACS/Two-Stage: materialize các query đã augmentation trong cửa sổ;
đo reward dưới no_grad, lưu RNG torch/CUDA/Python/NumPy và registered buffers;
chuẩn hóa reward detach trên mọi mẫu thực có trong cửa sổ; replay từng
microbatch với RNG/buffer gốc và backward trọng số n_micro/n_window.
Không giữ 16 đồ thị cùng lúc. Kiểm tra action và reward replay, từ chối
optimizer step nếu khác ngoài tolerance. Khôi phục RNG/buffers như sau đúng
một lượt forward, không tăng BatchNorm statistics hai lần.

Lượt replay tăng chi phí forward; không hứa nhanh hơn hay accuracy cao hơn.
Một optimizer step vẫn dùng window hiện tại, kể cả window cuối thiếu 64 ảnh.
Replay kiểm soát trạng thái RNG/buffers đã đăng ký, không hỗ trợ model có
side effects không đăng ký hoặc đọc dữ liệu ngẫu nhiên ngoài các RNG này.

## Tương thích

Checkpoint cũ không có advantage_scope được hiểu là microbatch. Resume kiểm
tra cả scope và microbatch size; không resume chéo scope hoặc ghi đè output
scope khác, kể cả --no-resume. Không đổi kiến trúc hay mặc định của pilot cũ.
Config/metrics/checkpoint ghi scope; history ghi task/policy loss và replay
delta cho chế độ mới. Validation accuracy tính như cũ; validation hybrid loss
vẫn chuẩn hóa theo từng evaluation microbatch, không gọi đó là train-window loss.

## Kiểm thử phần mềm

Full suite tại bước triển khai: 119 test, 0 lỗi, 2 integration test opt-in bỏ qua.
Fixtures ngẫu nhiên chỉ dùng để kiểm tra phần mềm, không làm dữ liệu train
hoặc kết quả thực nghiệm. Tests mới: full batch loss/gradient, batch 64 chia
4×16, window thiếu mẫu, singleton reward, dropout/RNG, BatchNorm, NaN,
external advantage sai shape/có gradient, replay sai, output scope guard.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_effective_batch.py -v
```

Giữ batch query, augmentation, action noise, initialization, số epoch và
scheduler cố định giữa hai nhánh pilot. Không so run mới với pilot khác
source như thể chỉ khác normalization. Chỉ mở test sau khi khóa phương pháp.
