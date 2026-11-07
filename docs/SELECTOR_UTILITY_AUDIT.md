# Kiểm tra ích lợi thực tế của selector role_v2

## Tiếp nhận output ngày 07/11/2026

Đã nhận `results.zip` ở thư mục gốc, 2.649.484.182 byte, SHA256:
`6ac6e4a0f510af5e1cd12abf83f927984fcb04919c5bb41e5f3c70536c57063b`.
Đã kiểm tra đường dẫn archive, không cho path traversal/symlink, giải nén
84 tệp vào `artifacts/role_v2/kaggle_v2/`; giữ nguyên ZIP và output legacy.
Đây là output selector pilot (Version 2), không chứa Random Version 1.

`selector_pilot_comparison.json` xác nhận seed 44, đủ ba run 20 epoch,
`test_evaluated=false`. Best validation khớp kết quả đã đọc trên Kaggle:
No-context 0,8754; TACS 0,8732; Two-Stage 0,8742.
Split trong ZIP và split local cùng SHA256:
`0559402db013819b631d1f2e9df39dbd14d191df52e45b29d1eb53a4b8a7d295`.
Các số trên là số liệu trainer lưu, chưa phải kết quả paired audit.

## Kết quả audit thật — hoàn tất ngày 07/11/2026

Đã chạy trên **toàn bộ 5.000 ảnh validation**, cùng thứ tự query, toàn pool
9.000 ảnh, batch 4, CUDA AMP trên RTX 3050 Laptop, audit seed 44, 3 random
trials/query. Không train hoặc đánh giá test. Checkpoint TACS tốt nhất ở epoch
19, Two-Stage ở epoch 20. Nạp strict thành công; SHA256 checkpoint sau audit
không thay đổi. Hash `model.py`, `data.py`, `retrieval.py`, `train.py` local
trùng source trong ZIP Kaggle.

| Can thiệp trên cùng task weights | TACS | Two-Stage |
|---|---:|---:|
| Selector chọn context | **87,32%** | **87,42%** |
| Random trial 0 | 87,20% | 87,46% |
| Random trial 1 | 87,26% | 87,62% |
| Random trial 2 | 87,44% | 87,54% |
| Random trung bình 3 trials | 87,30% | 87,54% |
| Bỏ context, giữ nguyên task weights | 87,26% | 86,90% |
| Selector trừ random trung bình | +0,02 điểm % | −0,12 điểm % |
| Selector trừ bỏ context | +0,06 điểm % | +0,52 điểm % |

TACS random lấy từ toàn pool; Two-Stage random chỉ lấy trong cùng top-32 của
từng query. Không so hai cột như thể chúng dùng cùng task weights. Hàng bỏ
context **không phải baseline No-context train độc lập** (best validation của
baseline độc lập trong pilot là 87,54%). Ba random trials không phải ba seed
huấn luyện và không biến 5.000 query thành 15.000 mẫu độc lập.

Accuracy selector của cả hai run **khớp chính xác** accuracy lưu trong
checkpoint (delta = 0), không chỉ nằm trong tolerance 0,1 điểm phần trăm.
Thời gian audit full: TACS 275,50 giây; Two-Stage 412,40 giây. Trước đó đã chạy
TACS trên 256 query, 84,08 giây; dùng full validation trên đây cho kết luận.

### Context có tác dụng đến mức nào?

Với TACS, bỏ context làm đổi dự đoán 62/5.000 ảnh (1,24%). Context do selector
chọn sửa đúng 26 ảnh nhưng làm sai thêm 23 ảnh: lợi ích ròng **3 ảnh**. Mean
cross-entropy giảm 0,000971 so với bỏ context, nhưng tăng 0,000981 so với random
trung bình. Chưa có bằng chứng lợi thế selector đáng kể/ổn định.

Với Two-Stage, bỏ context làm đổi dự đoán 126/5.000 ảnh (2,52%). Context sửa
đúng 59 ảnh, làm sai thêm 33 ảnh: lợi ích ròng **26 ảnh**. Mean cross-entropy
giảm 0,007772 so với bỏ context, nhưng tăng 0,001209 so với random trong top-32.
Context có ích trong phép can thiệp này; phần rerank bằng selector chưa vượt
đối chứng random từ cùng tập truy xuất trên checkpoint đang xét.

### Phân bố chọn context

| Chỉ số trên 5.000 query | TACS | Two-Stage |
|---|---:|---:|
| Số context khác nhau được chọn | 30/9.000 | 1.385/9.000 |
| Context được chọn nhiều nhất | 19,66% lượt | 0,78% lượt |
| 10 context phổ biến nhất | 85,56% lượt | 5,78% lượt |
| Context cùng nhãn với query | 1,26% | 45,92% |
| Mean policy entropy | 7,158011 | 2,789374 |
| Mean top-1 probability | 0,007954 | 0,172224 |

TACS argmax tập trung mạnh vào một nhóm nhỏ context. Đây là dấu hiệu cần
điều tra, **chưa đủ để khẳng định toàn bộ phân phối xác suất đã collapse**:
entropy trung bình vẫn lớn, và top-1 probability trung bình chỉ khoảng 0,8%.
Entropy hai cột có số candidate khác nhau (9.000 và 32), không so trực tiếp
độ lớn như cùng không gian chọn. Context cùng nhãn là thống kê mô tả,
không phải mục tiêu bắt buộc của TACS; nhãn không được dùng để chọn context.

### Quyết định tiếp theo

Chưa tăng epoch, chưa tuyên bố cải tiến vượt paper. Ưu tiên kiểm tra cosine
top-1 trên cùng task weights để tách hiệu quả truy xuất khỏi hiệu quả rerank;
đồng thời kiểm tra quan hệ norm embedding/score với tần suất chọn ở TACS.
Nếu kiểm tra ủng hộ giả thuyết cụ thể mới thiết kế thay đổi selector/loss và
train pilot đối chứng cùng recipe. Không âm thầm đổi raw dot-product của bản
tái hiện gốc. Chỉ một seed huấn luyện nên chưa có kết luận nhiều seed hoặc
kiểm định lợi thế của một phương pháp mới. Chưa đo gradient lúc train.

Artifact local:

- `artifacts/role_v2/audit_tacs_full/report.json`, SHA256
  `a2b71d72ceb6512cac5b3dd76acf1de152274125ed1c4c1f7a8e236e6d4fa717`.
- `artifacts/role_v2/audit_two_stage_full/report.json`, SHA256
  `e27d90a5b22eb2ecc6bb37871325459d17e2de4b3e50ad93f330d9ec9eb8e06d`.
- Mỗi thư mục có `manifest.json` và `queries.jsonl` (5.000 dòng), để kiểm tra
  dự đoán và các cặp đúng/sai. Các artifact được giữ ngoài Git.
- TACS checkpoint SHA256:
  `800464d634a0b2c522220949c360b47f0a165c5a74f02b604472f052740d33ed`.
- Two-Stage checkpoint SHA256:
  `4b188f41df9fef4a38788147c36fc3fd8e5846972e8db4b6b1acd09ae18c49a0`.
- Encoder No-context của cùng output Version 2:
  `ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64`.

Unit tests audit: 12/12 đạt trong lần kiểm tra này. Không dùng fixture của
unit tests làm dữ liệu inference; audit trên CIFAR-100 thật tại `data/`.

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
