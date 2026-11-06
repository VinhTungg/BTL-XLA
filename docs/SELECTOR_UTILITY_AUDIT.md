# Kiểm tra ích lợi thực tế của selector role_v2

## Trạng thái ngày 06/11/2026

Đã viết công cụ `python -m tacs.selector_audit` và unit tests. **Chưa chạy
audit trên checkpoint role_v2 thật**: chưa tải thành công output Kaggle.
Brave chặn URL tải output Version 2 bằng `ERR_BLOCKED_BY_CLIENT`.
Không dùng checkpoint legacy trong `cifar100/` để thay cho pilot mới.

Nguồn cần thu thập, giữ riêng từng version:

- Version 1: https://www.kaggle.com/code/b23dccn227trnngdng/tacs-role-pilot-seed44/output?scriptVersionId=365804474
- Version 2: https://www.kaggle.com/code/b23dccn227trnngdng/tacs-role-pilot-seed44/output?scriptVersionId=366021012

Tải bằng **Output → Output actions (⋮) → Download output**. Nút Download
ở đầu notebook chỉ tải notebook, không đủ checkpoint. Đặt ZIP vào
`artifacts/role_v2/downloads/`; giải nén mỗi version vào thư mục mới riêng,
không ghi đè `cifar100/` hoặc output cũ. Giữ ZIP gốc để đối chiếu.

Đường dẫn chuẩn trong ví dụ dưới đây giả định Version 2 được giải nén vào
`artifacts/role_v2/kaggle_v2/`, chứa trực tiếp `TACS_selector_role_v2_pilot/`.
Nếu ZIP có thêm thư mục bao ngoài, cần sửa biến `$pilotRoot` cho đúng.

## Mục tiêu và giới hạn

Giữ nguyên checkpoint TACS/Two-Stage, chạy cùng ảnh validation qua cùng
task network với ba cách cấp context:

1. Context do selector chọn bằng argmax, đúng quy tắc evaluation của trainer.
2. Context ngẫu nhiên, mặc định 3 lần trên mỗi query. TACS lấy từ toàn pool;
   Two-Stage lấy ngẫu nhiên trong **chính top-k truy xuất của query đó**.
3. Không cấp context. Đây là can thiệp trên task network của run selector,
   **không phải** baseline No-context đã train độc lập.

Đo accuracy, cross-entropy, chênh lệch theo từng query, tần suất chọn context,
entropy của phân phối chọn và tỷ lệ chọn ảnh cùng nhãn. Nhãn candidate chỉ
dùng để mô tả sau khi đã chọn, không đi vào selector. CE reduction dương nghĩa
là context được chọn làm giảm lỗi so với đối chứng. Tập trung vào ít context
chưa đủ để kết luận selector collapse; cần đọc cùng ích lợi thực tế của chúng.

Công cụ không train, không sửa checkpoint, không đọc partition CIFAR test,
không download dữ liệu/model. Khởi tạo kiến trúc `pretrained=False` rồi nạp
toàn bộ trọng số bằng `strict=True`. Không chuyển đổi legacy thành role_v2.

Giữ đúng resize/normalization evaluation và split của trainer. Kiểm tra config
checkpoint khớp `metrics.json`, SHA256 split, số mẫu, architecture và provenance
encoder truy xuất. Metrics cũ chưa ghi hash encoder được dùng trong Two-Stage;
vì vậy phải lấy `best_encoder.pt` No-context từ **cùng Version 2**, không chỉ lấy
một file bất kỳ có schema hợp lệ. Audit ghi hash file thực tế để truy vết về sau.

Chưa đo gradient lúc train. Không có kết luận cải tiến vượt paper từ audit này.
Sửa role là sửa lỗi triển khai; lợi ích selector là câu hỏi riêng cần kiểm tra.

## Chạy sau khi có output thật

Từ thư mục gốc dự án, kiểm tra checkpoint trước, chưa inference:

```powershell
$pilotRoot = 'artifacts/role_v2/kaggle_v2/TACS_selector_role_v2_pilot/cifar100/dinov3_vits16/role_v2/seed_44'
.\.venv\Scripts\python.exe -X utf8 -m tacs.selector_audit --checkpoint "$pilotRoot/tacs/checkpoint_best.pt" --output artifacts/role_v2/audit_tacs_preflight --check-only --device cpu
.\.venv\Scripts\python.exe -X utf8 -m tacs.selector_audit --checkpoint "$pilotRoot/two_stage/checkpoint_best.pt" --standard-encoder "$pilotRoot/no_context/best_encoder.pt" --output artifacts/role_v2/audit_two_stage_preflight --check-only --device cpu
```

`--check-only` không tạo thư mục output; chỉ kiểm tra cấu trúc, trọng số, config
và split. Chưa xác minh inference, accuracy hoặc CIFAR trên đĩa.

Pilot kiểm tra nhanh trên 256 query lấy ngẫu nhiên có seed, không lấy prefix
có thể lệch lớp. Vẫn mã hóa đầy đủ 9.000 candidate để không thay thuật toán:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.selector_audit --checkpoint "$pilotRoot/tacs/checkpoint_best.pt" --output artifacts/role_v2/audit_tacs_256 --max-queries 256 --batch-size 4 --device cuda
.\.venv\Scripts\python.exe -X utf8 -m tacs.selector_audit --checkpoint "$pilotRoot/two_stage/checkpoint_best.pt" --standard-encoder "$pilotRoot/no_context/best_encoder.pt" --output artifacts/role_v2/audit_two_stage_256 --max-queries 256 --batch-size 4 --device cuda
```

Chạy toàn validation bằng `--max-queries 0` và **thư mục output mới**, ví dụ
`artifacts/role_v2/audit_tacs_full`. Chỉ full validation mới so lại accuracy
với checkpoint. Ngưỡng mặc định 0,001 (0,1 điểm phần trăm) là ngưỡng phát hiện
sai khác số học/protocol, không phải ngưỡng ý nghĩa thống kê. Exit code 2 nếu
sai khác vượt ngưỡng; phải điều tra trước khi dùng kết quả. Để khớp pilot,
ưu tiên batch 4 và CUDA AMP như lần train; CPU/FP32 có thể làm thay đổi argmax.

Mặc định `--data-root data` và `--split-file data/splits/cifar100_seed42.json`.
Seed train 44 **không có nghĩa** split phải đổi sang seed44: split của pilot
được cố định và xác minh bằng SHA256. Không tạo split mới khi thiếu file.

## Đọc kết quả và quyết định tiếp theo

- `manifest.json`: nguồn checkpoint, SHA256, cấu hình và môi trường.
- `queries.jsonl`: từng query, dự đoán, loss, context được chọn và đối chứng.
- `report.json`: chỉ được tạo khi audit xong. Không coi manifest/JSONL dở là
  kết quả hoàn chỉnh. Thư mục đã tồn tại bị từ chối, không tự ghi đè.

Nếu selected tốt hơn random trong cùng task network, tiếp tục xác nhận trên
toàn validation và nhiều seed. Nếu không hơn, xem CE reward và phân phối chọn
để chọn giả thuyết điều tra tiếp; chưa tăng epoch hoặc thay thuật toán hàng loạt.
Nếu selected không hơn bỏ context, chưa có bằng chứng context hữu ích với
checkpoint đó. Không chọn cấu hình bằng test. Sau khi khóa phương pháp mới
chạy benchmark công bằng và đánh giá test cuối cùng.

## Kiểm thử phần mềm

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_selector_audit.py -v
```

Unit fixtures chỉ kiểm tra logic, không phải dữ liệu train/accuracy thực nghiệm.
Hai trường hợp biên quan trọng: split hoặc checkpoint legacy phải bị từ chối;
Two-Stage phải ánh xạ chỉ số trong top-k về đúng candidate pool, kể cả top-k
chỉ còn một phần tử. GPU thiếu VRAM: chạy lại với batch 1 ở thư mục mới,
nhưng ghi nhận khác biệt batch/precision khi so tái lập accuracy.
