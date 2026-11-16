# Demo Caltech-101 và CIFAR-100 — dùng checkpoint thật

Ngày 16/11/2026. Hai mô hình Caltech đã train 100 epoch và được thu hồi/audit.
Không cần train thêm để mở demo. Đây là giao diện offline Tk, không mở server,
không upload ảnh; `pretrained=False` khi khởi tạo inference rồi nạp weights thật,
không download/tráo mô hình nếu checkpoint thiếu.

## Mở và thao tác

Tại repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101
```

Hoặc launcher từ bất kỳ working directory:

```powershell
.\scripts\run_demo.ps1 -Dataset caltech101
```

Chờ **Sẵn sàng**. Dataset `caltech101`, preset `full100`; header ghi Two-Stage
best epoch54/val94,26% và No-context best epoch21/val95,64%. Chọn một ví dụ hoặc
validation position0..504, bấm **Chọn ảnh**, rồi **So sánh dự đoán**. Bấm từng
hàng kết quả để xem context thật, top-5 và SHA256 checkpoint tương ứng.

Chọn `cifar100` trong Dataset để quay về demo cũ. App nạp lại checkpoint,
xóa query/kết quả cũ, đổi class map, range validation/cache và ví dụ. Không giữ
nhãn Caltech để chấm ảnh CIFAR hay ngược lại. Trong lúc nạp/inference, các nút
đổi model bị khóa để tránh thay dataset giữa một dự đoán.

**Mở ảnh riêng…** cho phép PNG/JPEG/WebP/BMP, xử lý EXIF và RGB, tối đa16MP.
Ảnh riêng không có nhãn thật nên không tự chấm đúng/sai. Đây là phân loại đóng:
model buộc chọn một trong101lớp Caltech (hoặc100lớp CIFAR), không phát hiện unknown.
Preview Caltech đọc ảnh nguồn (ví dụ492×325,300×277), giữ tỷ lệ; model vẫn dùng
Resize256×256/ImageNet normalization, không tăng chi tiết bằng AI.

**Lưu bằng chứng JSON…**: chọn tên mới; có dataset/protocol, ảnh/nhãn/hình dạng,
RGB hash, vị trí validation và relative file, các predictions/top5/context/timing,
hash riêng từng model và cờ `weights_updated=false`, `test_evaluated=false`.
Không ghi đè JSON cũ. Không dùng timing một ảnh làm benchmark bài báo.

## Không nhầm hai loại No-context

| Hàng demo | Trọng số | Ý nghĩa |
|---|---|---|
| Selector đã học | Two-Stage best54 | Chọn từ top32 bằng selector đã học |
| Random trong top-32 | Cùng Two-Stage best54 | Một context random seed44, không phải baseline train riêng |
| Ảnh gần nhất (retrieval) | Cùng Two-Stage best54 | Lấy top1 frozen cosine, không phải baseline train riêng |
| Bỏ context (cùng weights) | Cùng Two-Stage best54 | Bỏ ảnh phụ để kiểm tra trên chính task đã train với context |
| No-context train riêng | **No-context best21** | Baseline độc lập train100epoch, có SHA256 khác |

Ví dụ context giúp hàng1 so với hàng4 **không chứng minh** Two-Stage thắng
baseline độc lập hàng5. Xác suất softmax chưa hiệu chỉnh, không phải độ chắc chắn
đáng tin tuyệt đối. Context cùng nhãn hoặc trông giống query cũng không bảo đảm giúp.

## Kịch bản demo 3–5 phút

1. Mở Caltech, nhấn **Cùng đúng ·0**, so sánh. Giải thích query,606ảnh pool
   train và việc model chọn ảnh phụ để phân loại. Chỉ ra ảnh gốc lớn hơn CIFAR32×32.
2. Nhấn **Context giúp ·178**: nhãn electric_guitar, selector dự đoán đúng;
   bỏ context trên cùng Two-Stage weights đoán mandolin. No-context train riêng
   cũng đúng, vì vậy ví dụ này chỉ cho thấy tác dụng của ảnh phụ trên cùng task.
3. Nhấn **Context gây sai ·169**: nhãn dolphin nhưng selector chọn context
   hawksbill rồi đoán hawksbill; bỏ context và No-context train riêng đều đoán
   dolphin. Đây là thất bại thật, không che đi khi bảo vệ.
4. Bấm hàng No-context train riêng: ảnh context phải biến mất, SHA256 đổi đúng
   baseline. Có thể chọn ảnh gần nhất để thấy không phải cứ selector học là tốt hơn.
5. Lưu JSON. Chuyển Dataset sang CIFAR-100, chờ nạp và chọn ví dụ cũ. Nêu rõ
   đây là dataset/checkpoint khác, không so trực tiếp phần trăm giữa hai dataset.

Các ví dụ trên đã tìm từ505validation và được chọn có chủ đích. Thêm ví dụ
đối chiếu hai model độc lập: vị trí27/airplanes chỉ No-context train riêng đúng;
vị trí73/brontosaurus Two-Stage đúng còn baseline đoán rooster. Không dùng
hai ảnh để thay cho kết quả toàn tập.

## Số liệu được phép trình bày

Cloud train: best validation No-context95,64%, Two-Stage94,26%; cả hai train100epoch.
Đã replay GUI engine trên toàn505ảnh validation ở RTX3050Laptop/CUDA AMP:

| Cách suy luận | Đúng/505 | Accuracy validation replay |
|---|---:|---:|
| Selector Two-Stage |476 |94,26% |
| Random trong top32 (một trial seed44) |474 |93,86% |
| Frozen retrieval top1 |477 |94,46% |
| Bỏ context, cùng task Two-Stage |470 |93,07% |
| No-context train độc lập |483 |95,64% |

Cloud score được so lại đúng khi làm tròn; replay batch1 local không phải train
mới/test benchmark. Khác batch/device/FP16 có thể làm logits khác nhẹ.
Two-Stage chưa thắng No-context và top1 retrieval trong lượt này; không nói cải
tiến tăng accuracy. Chi tiết protocol/hash: [CALTECH101_FULL100_RESULTS.md](CALTECH101_FULL100_RESULTS.md).

## Dữ liệu cần giữ và kiểm thử

Artifact tại `artifacts/caltech_demo/seed44_full100_v1/`, dữ liệu tại
`data/caltech101/101_ObjectCategories/`. ZIP gốc vẫn giữ. Cache riêng Caltech:
`artifacts/demo_cache/caltech101/`; kiểm tra identity theo checkpoint/split/
inventory/device/precision. Không mang chỉ source Git sang máy khác rồi giả định
checkpoint và ảnh Git-ignored cũng có sẵn. Máy demo có thể dùng CPU với
`--device cpu`, nhưng chưa nghiệm thu hiệu năng CPU.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101 --check-ui
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101 --headless --validation-position 178 --output artifacts/caltech_demo/retry_178.json
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
```

Replay toàn bộ validation (tốn thời gian, không cần chạy lại trước mỗi demo):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_caltech_demo.py --output artifacts/caltech_demo/rehearsal_retry
```

Đã đạt165tests,0lỗi,2integration opt-in skipped; replay505validation không
thay parameter versions, không mở test. File bằng chứng đã có tại
`artifacts/caltech_demo/rehearsal_validation/`, gồm summary và ví dụ đúng/sai;
GUI export thật `artifacts/caltech_demo/gui_validation_178.json` đã đọc lại xác minh.
Screenshots `gui_context_helps.png` / `gui_context_hurts.png` trong cùng thư mục.
Đã chuyển GUI từ Caltech sang CIFAR, xác nhận query/report cũ bị xóa, nạp lại
microbatch và dự đoán apple thành công; sau đó chuyển lại Caltech. Bố cục,
tiếng Việt, context/top-5 và hash riêng No-context đã được xem trực quan.
Phần mở ảnh riêng được unit-test grayscale/RGBA/ảnh hỏng/quá16MP; chưa kiểm thử
đầy đủ mọi định dạng qua native file chooser trong lượt này.

Hai trường hợp biên: Caltech position−1/505 phải bị từ chối trước nạp model;
đổi dataset phải xóa query/report/context cũ. Thiếu artifact/hash mismatch phải
báo lỗi, không fallback sang CIFAR hay mô hình chưa train.
