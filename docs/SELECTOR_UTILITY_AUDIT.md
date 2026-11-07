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

## Audit cosine và norm — hoàn tất ngày 07/11/2026

Source audit: commit `21ed393`. Hai lượt chạy đều hoàn thành trên 5.000 query,
pool 9.000, batch 4, CUDA AMP, seed 44 và cùng checkpoint/encoder của lượt
trước. Không train, không test, không cập nhật trọng số. 18 trường của các
nhánh cũ trên từng query (ID, target, score entropy/probability, context,
prediction, CE) khớp hoàn toàn với audit trước: 0 giá trị sai khác ở cả hai run.
SHA256 checkpoint sau chạy không đổi; accuracy gốc vẫn tái lập đúng 87,32%
và 87,42%. Thời gian TACS 488,25 giây, Two-Stage 568,51 giây.

### Accuracy khi chỉ thay cách chọn context

| Can thiệp, giữ nguyên task weights trong mỗi cột | TACS | Two-Stage |
|---|---:|---:|
| Selector gốc (dot-product, AMP) | 87,32% | 87,42% |
| Dot-product tính lại float32 trên cùng embedding | 87,28% | 87,44% |
| Cosine từ embedding selector đã học | 87,14% | 87,62% |
| Cosine top-1 từ encoder truy xuất cố định | 87,16% | **87,66%** |
| Random trung bình 3 lần | 87,30% | 87,54% |
| Bỏ context trên cùng task weights | 87,26% | 86,90% |

`selector_cosine` vẫn chọn trong cùng tập ứng viên của selector: toàn pool với
TACS, cùng top-32 với Two-Stage. `retrieval_top1` dùng encoder cố định của
No-context cùng pilot; đây **không phải một baseline Similarity được train
độc lập**. Không đổi tên can thiệp này thành kết quả mô hình mới hoặc kết quả
paper. Hàng bỏ context cũng không phải baseline No-context train độc lập.

TACS: cosine selector làm đúng thêm 14 ảnh nhưng làm sai thêm 23 ảnh so với
selector gốc, ròng −9 ảnh (−0,18 điểm %). Cosine encoder cố định ròng −8 ảnh
(−0,16 điểm %). Vì vậy thay sang cosine lúc inference **không cải thiện**
accuracy TACS ở checkpoint này; chưa kết luận về mô hình được train lại bằng
cosine ngay từ đầu.

Two-Stage: cosine selector đúng thêm 44 ảnh, sai thêm 34 ảnh, ròng +10 ảnh
(+0,20 điểm %). Cosine encoder cố định đúng thêm 48 ảnh, sai thêm 36 ảnh,
ròng +12 ảnh (+0,24 điểm %). Cross-entropy tương ứng: selector gốc 0,421209;
cosine selector 0,420321; cosine encoder cố định 0,418678. Cosine encoder cố
định cũng hơn random trung bình 0,12 điểm %, nhưng đây là một checkpoint/một
seed, chưa có kết luận về độ ổn định hoặc ý nghĩa thống kê qua nhiều lần train.

### Norm có liên quan đến việc chọn lặp không?

Norm là độ dài vector embedding. Dot-product phụ thuộc cả norm và hướng;
cosine chuẩn hóa norm để so hướng. Với một query cố định, norm query là hằng
số đối với mọi candidate; bỏ norm candidate có thể thay đổi thứ hạng.

| Chỉ số | TACS gốc | TACS cosine selector | Two-Stage gốc | Two-Stage cosine selector |
|---|---:|---:|---:|---:|
| Context khác nhau được chọn | 30 | 1.872 | 1.385 | 3.399 |
| Tỷ lệ lượt chọn vào top-10 context phổ biến | 85,56% | 3,32% | 5,78% | 1,34% |
| Norm trung bình context được chọn | 12,6542 | 11,6728 | 12,8506 | 12,6068 |
| Percentile norm trung bình trong tập ứng viên hợp lệ | 84,67% | 49,10% | 95,93% | 51,77% |

Norm trung bình khi chọn đều từ tập ứng viên: TACS 11,6897; Two-Stage 12,6048.
Pearson giữa norm và số lượt chọn: TACS 0,0526, Two-Stage 0,3562; nếu điều chỉnh
theo số lần candidate có mặt trong top-k, Two-Stage là 0,4091. Tương quan TACS
trên toàn pool yếu, nên **không khẳng định norm là nguyên nhân duy nhất**.
Can thiệp chuẩn hóa làm lựa chọn phân tán hơn rõ rệt, nhưng không tự động tạo
context hữu ích hơn cho task network: accuracy TACS vẫn giảm.

Cosine encoder cố định chọn context cùng nhãn ở 65,10% query trong cả hai run;
cosine selector đạt 3,06% ở TACS và 46,00% ở Two-Stage. Cùng nhãn là thống kê
mô tả, không phải mục tiêu bắt buộc của selector task-aligned và không được dùng
trong quá trình chọn. Cùng một context retrieval có thể tác động khác nhau
đến hai task network đã được train khác nhau.

### Đối chứng sai khác precision

Đổi dot-product sang float32 làm đổi context ở 19,68% query TACS và 9,68% query
Two-Stage, nhưng chỉ đổi dự đoán lớp ở 8 và 9 ảnh tương ứng. Float32 vẫn chỉ
chọn 26 context khác nhau ở TACS, top-10 chiếm 85,50% lượt. Vì vậy AMP rounding
gây nhạy cảm lựa chọn, nhưng **không giải thích hết** hiện tượng tập trung.
Đây là rescoring float32 của embedding hiện có, không phải re-encode toàn bộ
ảnh bằng FP32; CUDA matmul precision `high` được giữ như trainer.

### Quyết định sau audit

1. Giữ nguyên bản tái hiện dot-product; không sửa mặc định sang cosine.
2. Xác nhận đối chứng đơn giản frozen-cosine top-1 và biến thể cosine rerank
   của Two-Stage bằng protocol train/eval công bằng, nhiều seed trước khi
   tuyên bố cải thiện. Kết quả trên là inference ablation, không thay thế
   một baseline train độc lập. Chọn cosine top-1 đơn thuần cũng không phải
   đóng góp nghiên cứu mới của nhóm.
3. Với TACS, điều tra tiếp reward/advantage và gradient của selector trên
   batch thật trước khi tăng epoch hoặc đổi loss. Kiểm tra cả đường task-loss
   qua proxy và policy loss; không tự gọi có lỗi gradient chỉ từ accuracy.
4. Chỉ thiết kế/train pilot cải tiến sau khi có giả thuyết cụ thể. Không dùng
   test để chọn hướng. Chưa tự khởi chạy job train nào trong bước này.

Artifacts:

- `artifacts/role_v2/audit_tacs_cosine_full/report.json`, SHA256
  `ba013ec953bd7da07982262f65180669d1f38f2bb4036b77feff78cfe01d0f24`.
- `artifacts/role_v2/audit_two_stage_cosine_full/report.json`, SHA256
  `2d313322c3c9ef254d8b8298cefe60598dd9410c1bdccc43fa8133b5e6c025bb`.
- Mỗi thư mục chứa `queries.jsonl`, `manifest.json`, `candidate_norms.json`.
- 20/20 unit tests audit đạt, gồm kiểm tra điều chỉnh theo cơ hội xuất hiện
  trong top-k và không tính các can thiệp mới vào random trung bình.

## Kiểm tra gradient tiếp nối — 07/11/2026

Đã hoàn tất probe task/policy gradient trên cùng 64 validation query ở cả hai
checkpoint, với hai diagnostic loss scale 128 và 1024. Không đứt gradient ở
những batch được đo; policy/task L2 sau tích lũy khoảng 32× ở TACS và 44× ở
Two-Stage. Advantage chuẩn hóa theo batch 4 khác chuẩn hóa theo window 64 ở
25/64 dấu trong mỗi run. Đây là đầu mối cho ablation, không phải bằng chứng
đã tìm ra nguyên nhân duy nhất hoặc một cải tiến accuracy mới.
Chi tiết, giới hạn precision, artifacts và nhiệm vụ tiếp theo tại
[GRADIENT_AUDIT.md](GRADIENT_AUDIT.md). Không có optimizer step, không test,
không thay checkpoint hoặc loss training mặc định.

## Trạng thái lịch sử ngày 06/11/2026

Phần này là nhật ký lịch sử trước khi nhận ZIP; blocker tải output đã được
giải quyết ngày 07/11/2026 như phần trên.

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

### Mở rộng cosine và norm (chỉ inference)

Thêm `--cosine-diagnostics` cùng `--standard-encoder` của No-context trong
cùng pilot. Với TACS, encoder cố định chỉ tạo đối chứng; không thu hẹp pool
9.000 ảnh của selector gốc. Với Two-Stage, giữ nguyên top-32 và chọn ảnh đầu
tiên trong thứ tự cosine truy xuất làm đối chứng `retrieval_top1`.

Ba can thiệp mới trên cùng task weights:

- `retrieval_top1`: cosine top-1 từ encoder cố định, khác encoder selector.
- `selector_cosine`: chuẩn hóa chính embedding selector rồi chọn cosine
  cao nhất, trong cùng không gian ứng viên của selector gốc.
- `selector_dot_fp32`: tính lại dot-product ở float32 để đối chiếu ảnh hưởng
  làm tròn score AMP. Các embedding vẫn là embedding đã trích xuất theo AMP;
  không gọi đây là toàn bộ pipeline FP32. CUDA matmul precision vẫn là `high`.

Can thiệp không đổi loss, model, checkpoint hoặc recipe huấn luyện. Việc cosine
có làm tăng accuracy hay không phải được đo, không giả định từ norm lớn.
Report thêm norm quantiles, norm context được chọn, percentile norm trong tập
ứng viên hợp lệ và tương quan với tần suất chọn. Với Two-Stage, báo thêm tương
quan theo tỷ lệ chọn trên số lần candidate có mặt trong top-k để tránh bỏ qua
khác biệt cơ hội được chọn. Các tương quan là mô tả, không chứng minh nhân quả.

Ví dụ tái chạy (phải đặt output mới nếu thư mục này đã tồn tại):

```powershell
$pilotRoot = 'artifacts/role_v2/kaggle_v2/TACS_selector_role_v2_pilot/cifar100/dinov3_vits16/role_v2/seed_44'
.\.venv\Scripts\python.exe -X utf8 -m tacs.selector_audit --checkpoint "$pilotRoot/tacs/checkpoint_best.pt" --standard-encoder "$pilotRoot/no_context/best_encoder.pt" --output artifacts/role_v2/audit_tacs_cosine_recheck --max-queries 0 --batch-size 4 --device cuda --cosine-diagnostics
```

Đổi checkpoint thành `two_stage/checkpoint_best.pt` và output riêng để kiểm tra
Two-Stage. `candidate_norms.json` ghi norm của toàn bộ pool cùng chỉ số CIFAR
gốc; `queries.jsonl` ghi context/dự đoán/loss của các can thiệp.

### Unit tests

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_selector_audit.py -v
```

Unit fixtures chỉ kiểm tra logic, không phải dữ liệu train/accuracy thực nghiệm.
Hai trường hợp biên quan trọng: split hoặc checkpoint legacy phải bị từ chối;
Two-Stage phải ánh xạ chỉ số trong top-k về đúng candidate pool, kể cả top-k
chỉ còn một phần tử. GPU thiếu VRAM: chạy lại với batch 1 ở thư mục mới,
nhưng ghi nhận khác biệt batch/precision khi so tái lập accuracy.
