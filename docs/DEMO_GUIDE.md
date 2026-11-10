# Demo TACS bằng checkpoint thật

## Chạy trên Windows

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.demo
```

Không dựng server, không mở cổng mạng, không upload ảnh ra ngoài. GUI Tk
cho chọn ảnh validation hoặc mở ảnh riêng, hiển thị query/context, top-5 lớp,
probability, thời gian từng phần và lưu JSON theo đường dẫn người dùng chọn.
Đổi hàng trong bảng để xem context của selector, random hoặc frozen cosine;
hàng bỏ context không có ảnh phụ.

Mặc định dùng Two-Stage weight 1,0, seed 44, best checkpoint epoch 20 đã audit.
Chương trình xác minh hash checkpoint/metrics, encoder, split, strict state;
không tự tải pretrained weights, dataset hay checkpoint. Thiếu artifact sẽ
báo lỗi, không tạo mô hình/ảnh giả để demo.

Các file mặc định được giải nén tại `artifacts/policy_weight_pilot/`.
Lần đầu tạo 2 bank cho 9.000 ứng viên; lần sau dùng cache trong
`artifacts/demo_cache/`. Cache gắn với checkpoint, encoder, split, precision,
device, torch version và transform; không dùng nhầm giữa model/run.

## Ví dụ đã chạy thật

Các vị trí dưới đây là **validation position**, không phải chỉ số CIFAR gốc.
Đã kiểm tra bằng inference batch 1 trên RTX 3050, CUDA AMP, torch 2.14.1+cu126:

| Vị trí | Nhãn thật | Selector | Bỏ context | Ý nghĩa |
|---:|---|---|---|---|
| 0 | apple | apple | apple | Hai cách cùng đúng |
| 15 | apple | apple | pear | Context sửa một dự đoán sai |
| 158 | bear | beaver | bear | Context cũng có thể làm hại |

Các ví dụ chọn có chủ đích để giải thích, **không** dùng thay accuracy toàn
validation. Precision/batch/device khác có thể đổi trường hợp biên; trước
buổi bảo vệ chạy lại trên đúng máy và giữ cả ví dụ thất bại.

Artifact đã tạo: `artifacts/demo_validation_0.json`, `artifacts/demo_cases.json`.
Lượt đầu setup bank đo 102,35 giây; query 0 dự đoán apple. Không dùng một
query hoặc thời gian cold-start này để tuyên bố accuracy/benchmark tổng thể.
Đã kiểm tra Tk widgets bằng `--check-ui`; chưa nghiệm thu giao diện qua ảnh
chụp màn hình hoặc thao tác GUI hoàn chỉnh.

## Kịch bản trình bày 5–7 phút

1. Giải thích query là ảnh cần phân loại; context là ảnh được chọn trong pool
   train riêng, không lấy từ tập test hay query đang đánh giá.
2. Mở vị trí 0, bấm so sánh. Chỉ ra ba ảnh context khác nhau nhưng các hàng
   dùng cùng task weights; đây không phải bốn baseline train độc lập.
3. Mở vị trí 15 để thấy context có thể sửa sai; tiếp tục vị trí 158 để công khai
   giới hạn. Không coi context cùng lớp là điều kiện bắt buộc để hữu ích.
4. Nêu pilot ba policy weight: 1,0 / 0,1 / 0,03 chưa chứng minh cải thiện ổn
   định. Phân biệt validation pilot với kết quả test và benchmark của paper.
5. Trình bày giả thuyết chuẩn hóa advantage: batch hiệu dụng 64 không đồng
   nghĩa reward được chuẩn hóa trên 64 nếu loss vẫn xử lý từng nhóm 4.
   Replay đã được kiểm thử; hiệu quả accuracy còn phải train đối chứng.
6. Lưu JSON để chỉ ra checkpoint hash, nhãn dự đoán, selected context và timing.

Ảnh CIFAR-100 gốc 32×32 được phóng lớn nearest-neighbor; không hứa ảnh sắc
nét hơn. Ảnh riêng có thể ngoài phân phối, mô hình vẫn buộc chọn một trong
100 lớp. Probability không phải độ tin cậy đã calibration. Timing demo batch 1
gồm nhiều can thiệp, không phải latency benchmark của một baseline.

## Chạy kiểm tra không mở GUI

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.demo --check-only
.\.venv\Scripts\python.exe -X utf8 -m tacs.demo --check-ui
.\.venv\Scripts\python.exe -X utf8 -m tacs.demo --headless --validation-position 15 --output artifacts/demo_check_15.json
```

Output CLI phải mới, chưa tồn tại. Kiểm tra biên: vị trí ngoài 0..4999 bị từ
chối; checkpoint sai hash bị từ chối trước khi nạp tensor. Không sửa mật khẩu,
permission hoặc tải test để xử lý lỗi demo. Không xóa checkpoint khi xóa cache.
