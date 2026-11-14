# Demo offline CIFAR-100 bằng checkpoint thật

## Phạm vi đã chốt — 14/11/2026

Phạm vi hiện hành: giữ CIFAR-100 và **thêm Caltech-101 để demo ảnh rõ**; không
thêm Oxford Pets. Agent đã hiểu nhầm "tập trung demo" là bỏ dataset mới, nay
đã sửa theo ADR-006. **Hướng dẫn dưới đây chỉ áp dụng cho demo CIFAR-100 đã
chạy được**, chưa phải demo Caltech-101. Kế hoạch bổ sung tại
[CALTECH101_DEMO_PLAN.md](CALTECH101_DEMO_PLAN.md).

Giữ toàn bộ CIFAR-100/checkpoint/audit. Giao diện trình bày riêng ở
`src/tacs/showcase/`, tái sử dụng `DemoEngine`; không thay source train,
thuật toán hoặc bundle đã dùng để so sánh microbatch/effective-batch.

## Mở demo trên Windows

Tại repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase
```

Hoặc launcher tự xác định project root:

```powershell
.\scripts\run_demo.ps1
```

Không cần đổi execution policy nếu PowerShell chặn script: dùng lệnh Python
ở trên. Không dựng server/mở port/upload ảnh; không tự tải dữ liệu hay weights.
Máy cần bộ `.venv` đã cài và các artifact thật có sẵn. Không sao chép cả project
sang máy khác rồi giả định checkpoint Git-ignored cũng đã được mang theo.

Mặc định `microbatch`, best epoch20, seed44, validation trainer87,92%.
Chọn `effective_batch` trong danh sách và bấm **Nạp checkpoint** để dùng best
epoch17, validation trainer87,80%. Đây là hai model khác nhau. Mỗi lần chỉ
giữ một model, tránh nạp đồng thời hai bộ vào GPU4GB.

Có thể chọn preset ngay khi mở:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --preset effective_batch
```

Nếu không có CUDA, `--device auto` dùng CPU; không hứa CPU đủ nhanh để bảo vệ.
Lần đầu mã hóa hai kho đặc trưng9.000ảnh (retrieval và selector), có thể mất
vài phút. Cache ở `artifacts/demo_cache/`, gắn checkpoint/encoder/split/device/
torch/precision. Các lần sau dùng lại cache; không xóa cache trước buổi demo.

## Thao tác

1. Chờ status **Sẵn sàng**, xem tên preset, epoch và thiết bị.
2. Nhập validation position0..4999 rồi bấm **Chọn ảnh**, hoặc chọn ví dụ nhanh.
   Position khác CIFAR ID; cả hai được hiển thị và lưu JSON.
3. Bấm **So sánh dự đoán**. Bảng hiển thị selector, ảnh gần nhất từ retrieval,
   một random context trong top-32 và không-context; kèm đúng/sai nếu biết nhãn.
4. Bấm từng hàng để xem context thật và top-5 xác suất. Chọn không-context thì
   không có ảnh phụ. Cùng nhãn context không bảo đảm dự đoán hữu ích.
5. Bấm **Lưu bằng chứng JSON…**, chọn **tên file mới**. Không ghi đè bằng chứng.
   JSON có preset/hash/epoch, input RGB hash, validation position/CIFAR ID/nhãn
   hoặc tên ảnh riêng, dự đoán/top-5/context/timing và random seed44.

Ảnh CIFAR gốc32×32 được phóng nearest-neighbor, không tạo thêm chi tiết. Ảnh
riêng giữ tỷ lệ khi preview, chuyển RGB/EXIF, giới hạn16MP, không upload ra ngoài.
Ảnh riêng không có nhãn thật: bảng ghi dấu gạch thay vì tự chấm đúng/sai.
Mô hình vẫn buộc chọn1trong100lớp, không phát hiện unknown/ảnh ngoài phân phối.

## Cách giải thích đúng khi bảo vệ

- **Các hàng dùng cùng task weights**, không phải bốn baseline train độc lập.
  Random demo là một context với seed44, không phải random mean3trials của audit.
- Best accuracy hiển thị ở header là số trainer ghi trên validation, không phải
  accuracy của ảnh đang chọn hoặc test accuracy. Tái đo audit có sai lệch nhỏ.
- Hai nhánh có best validation87,92%/87,80%; effective-batch chưa chứng minh
  tăng accuracy, cosine inference cũng không tăng. Không sửa câu chuyện nghiên
  cứu thành “cải tiến thắng” chỉ vì một ví dụ dự đoán đúng.
- Probability chưa calibration. Timing batch1 gồm nhiều can thiệp; không dùng
  tổng thời gian so sánh làm latency benchmark chuẩn hoặc so A100 trong paper.
- Ví dụ trình bày được chọn có chủ đích, không thay thế5.000query audit.

Xem số liệu tổng tại [EFFECTIVE_BATCH_COSINE_AUDIT.md](EFFECTIVE_BATCH_COSINE_AUDIT.md).

## Kịch bản 3–5 phút

1. Nêu phạm vi: CIFAR-100, TACS/Two-Stage, checkpoint thật, chỉ suy luận offline.
2. Mở vị trí0 và so sánh; giải thích query, kho context train, selector và task.
3. Thử ví dụ A120 và B111: nêu nhãn thật, chỉ ra trường hợp context giúp/hại
   theo kết quả thực tế trên máy, không hứa mọi preset có cùng hành vi.
4. Chọn ảnh gần nhất/random/không-context trong bảng; chỉ ra context có thể đổi
   nhưng dự đoán không nhất thiết đổi. Liên hệ bảng audit tổng, không cherry-pick.
5. Lưu JSON. Nếu cần, đổi checkpoint sang effective-batch và nhắc đây là model
   khác, không phải toggle phương pháp trên cùng task weights.

## Kiểm tra trước buổi demo

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --check-ui
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --headless --validation-position 120 --output artifacts/demo_rehearsal_120.json
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_showcase.py -v
```

Lệnh check-ui chỉ kiểm tra widgets, không thay kiểm tra trực quan/inference.
Headless output phải là tên mới. Không chạy smoke/headless song song GUI khi
GPU4GB gần hết bộ nhớ. Cắm sạc, mở sẵn app/cache trước bảo vệ.

Hai trường hợp biên: position−1/5000 phải bị từ chối; ảnh hỏng/ảnh quá16MP hoặc
output JSON đã có phải báo lỗi và giữ dữ liệu cũ. Thiếu/sai checkpoint hoặc hash
phải dừng, không tự tạo mô hình/dữ liệu giả và không tải weights thay thế.

## Tương thích bản demo trước

`python -m tacs.demo` vẫn giữ nguyên cho pilot policy-weight cũ. Không dùng
các vị trí minh họa cũ15/158 như thể đã kiểm chứng cho checkpoint advantage mới.

## Nghiệm thu thực tế — 14/11/2026

Đã chạy6case inference thật trên RTX3050Laptop/CUDAAMP/torch2.14.1+cu126,
3ảnh cho mỗi checkpoint, không train/test. Output riêng tại
`artifacts/demo_showcase_rehearsal/`, summary status complete:

| Checkpoint | Position / nhãn thật | Selector | Không context |
|---|---|---|---|
| microbatch | 0 / apple | apple | apple |
| microbatch | 120 / baby | baby | boy |
| microbatch | 111 / baby | girl | baby |
| effective_batch | 0 / apple | apple | apple |
| effective_batch | 120 / baby | baby | baby |
| effective_batch | 111 / baby | girl | girl |

Vì vậy A120 là ví dụ giúp và B111 là ví dụ hại **trên microbatch**, không phải
lời hứa cho mọi checkpoint. Random cùng ảnh/một trial cũng không thay mean audit.

Đã điều khiển GUI thật: chọn A120, bấm so sánh, xem hàng không-context (không
hiện ảnh phụ), lưu JSON qua Save As, đổi/nạp effective-batch rồi dự đoán lại.
File xuất GUI `artifacts/demo_showcase_gui_120.json` đã được đọc lại xác minh
query position120/targetbaby/presetmicrobatch, selected đúng và no-context sai.
Bằng chứng trực quan: `artifacts/demo_showcase_help.png` và
`artifacts/demo_showcase_effective.png`. Bố cục và tiếng Việt đã xem qua screenshot.
Ảnh riêng/RGBA/ảnh hỏng/quá16MP được kiểm tra ở unit tests; chưa kiểm thử đầy đủ
luồng mở ảnh riêng qua hộp thoại GUI trong lượt nghiệm thu này.

Hash summary: `f4052bd3dcbc2a58fb6b87f37e654f90d02139f134f4a8644087a1ea35b54953`.
Hash JSON GUI: `d709cbba4b63490d6cbb25fcf74e7d2f14bec3aa4680f198d600a71d2613a0f6`.
Training bundle vẫn là
`1d85d0bbb17201c08e38b7428f720fccfe2859cc94cd0b256fac1103c7c9dce5`.
Không sửa `inference.py`, `demo.py`, model/train/data hoặc checkpoint gốc.

Full suite sau cập nhật:145tests, không lỗi,2integration opt-in bỏ qua;
trong đó10tests mới cho preset guards, input/provenance, ảnh hỏng/quá lớn và
không ghi đè JSON. `--check-ui` và `git diff --check` đạt. Unit tests dùng
fixtures riêng trong thư mục tạm, không phải dữ liệu/thực nghiệm demo.

Có thể chạy lại rehearsal bằng tên output mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_demo_showcase.py --output artifacts/demo_showcase_rehearsal_retry
```

Đây là kiểm tra6ví dụ, không phải đánh giá accuracy mới.
