# Pilot đối chứng policy weight — 08/11/2026

## Mục tiêu và trạng thái

Kiểm tra giảm trọng số policy loss có giúp selector chọn context hữu ích hơn
hay không. Gradient audit gợi ý thử nghiệm này, **không chứng minh** giảm
trọng số sẽ tăng accuracy. Bộ notebook đã được tạo. Ngày 08/11/2026 đã khởi
chạy đối chứng Two-Stage weight 1,0 trên Kaggle. Ngày 09/11/2026 xác minh
đối chứng hoàn tất 20 epoch, best/final validation accuracy 87,34%; đã khởi
chạy weight 0,1 riêng biệt (chi tiết bên dưới). Cập nhật 10/11/2026: cả ba
trọng số Two-Stage đã có kết quả 20 epoch; weight 0,03 đã được tải về local.
Các mục theo ngày bên dưới là lịch sử, không phải trạng thái hiện tại.

Thiết kế gồm TACS / Two-Stage × policy_weight 1,0 / 0,1 / 0,03. Mỗi notebook
chỉ chạy một thí nghiệm, một GPU, không chạy cả sáu job liên tục. Nếu ngân
sách hạn chế, hoàn tất ba cấu hình Two-Stage trước để có một so sánh trọn vẹn.
Ba thành viên có thể chia nhau ba trọng số, nhưng phải dùng cùng loại GPU và
phiên bản thư viện; so manifest sau khi thu kết quả. Không coi ba trọng số là
ba seed độc lập.

## Các yếu tố giữ cố định

- CIFAR-100 thật; split `cifar100_seed42.json`; query_train 36.000,
  candidate_pool 9.000, validation 5.000. Không đánh giá test.
- DINOv3 ViT-S/16 pretrained; task network `role_v2`; training seed 44.
- 20 epoch, LR 1e-4, warm-up 5, weight decay 1e-4, min LR 0.
- Microbatch 4 × accumulation 16, AMP, 2 data workers; top-k 32,
  selector temperature 0,1, raw dot-product, advantage theo microbatch.
- Trong mỗi baseline, chỉ thay `policy_weight`. Không warm-start task hoặc
  selector từ pilot đã train; cả ba cấu hình bắt đầu từ cùng recipe khởi tạo.
- Two-Stage dùng cùng encoder No-context của selector pilot Version 2.
  SHA256 bắt buộc:
  `ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64`.

Không thay `train.py`, `model.py`, `losses.py`. Notebook nhúng source hiện tại,
split và runner, không dùng ZIP source cũ. `timm==1.0.30` được ghim theo môi
trường local đã kiểm tra; torch/torchvision dùng bản có sẵn của Kaggle và được
ghi vào manifest. Đây chưa phải môi trường container khóa toàn bộ dependency.
Seed giống nhau không đảm bảo mọi phép toán CUDA khớp bit tuyệt đối.

## File và cách chạy

Tạo lại notebook bằng:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/build_policy_weight_notebooks.py
```

Sáu file trong `artifacts/policy_weight_pilot/`:

| Baseline | Đối chứng 1,0 | Ứng viên 0,1 | Ứng viên 0,03 |
|---|---|---|---|
| TACS | `Policy_tacs_w1p0_seed44.ipynb` | `Policy_tacs_w0p1_seed44.ipynb` | `Policy_tacs_w0p03_seed44.ipynb` |
| Two-Stage | `Policy_two_stage_w1p0_seed44.ipynb` | `Policy_two_stage_w0p1_seed44.ipynb` | `Policy_two_stage_w0p03_seed44.ipynb` |

1. Import một file vào notebook Kaggle mới. Giữ nguyên ba cấu hình trong
   mỗi baseline, không tự đổi epoch/batch ở riêng một file.
2. Two-Stage: đính kèm dataset có `best_encoder.pt` trong thư mục No-context
   của selector role_v2 pilot V2. File local gốc:
   `artifacts/role_v2/kaggle_v2/TACS_selector_role_v2_pilot/cifar100/dinov3_vits16/role_v2/seed_44/no_context/best_encoder.pt`.
   Có thể upload riêng file này để tiết kiệm dung lượng; không cần upload
   cả `results.zip`. Notebook tìm file có đúng hash, không đoán theo tên.
   TACS không cần dataset checkpoint này. CIFAR-100 tải tự động.
3. Bật GPU và Internet, kiểm tra quota/thời gian phiên còn lại trước khi chạy.
   Mỗi notebook chỉ dùng GPU 0 ngay cả khi có hai GPU. Không hứa một run sẽ
   xong trong quota hiện tại; dùng thời gian epoch thực tế để ước lượng.
4. Dùng Save & Run All. Cell train phát log trực tiếp và lưu `train.log`.
5. Chỉ coi hoàn tất khi có `POLICY WEIGHT PILOT COMPLETE` và cell kiểm tra
   cuối thành công. Tải **toàn bộ thư mục** `policy_weight_<baseline>_<weight>_seed44`
   trong output. Không tạo ZIP trùng toàn bộ checkpoint trong working để tránh
   nhân đôi dung lượng. Notebook source/dataset tải về cũng chiếm working disk.

Runner từ chối encoder sai hash, output không có manifest nhưng chứa dữ liệu,
hoặc manifest không khớp source/split/recipe/môi trường. Bản chạy đối chứng
weight 1,0 được train mới; không lấy accuracy pilot cũ làm đối chứng tương đương
khi môi trường hoặc source có thể khác.

## Resume sau gián đoạn

Trong cùng phiên, chạy lại cell train để tiếp tục checkpoint hiện có. Nếu
phiên mới, phải đính kèm output cũ đã lưu thành dataset, đặt `RESUME_FROM` ở
cell cấu hình tới **thư mục chứa `experiment_manifest.json`**, rồi chạy lại.
Notebook copy toàn bộ thư mục sang working và kiểm tra manifest trước train.
Nếu copy đã tồn tại, đặt `RESUME_FROM=None` để dùng bản working; không ghi đè.

Giữ nguyên notebook/source, baseline, trọng số, epoch, môi trường và GPU.
Không lấy checkpoint của trọng số 1,0 để resume 0,1; không tăng epoch ở một
nhánh rồi đem so với các nhánh 20 epoch. Trainer ưu tiên latest checkpoint,
fallback best nếu latest không có, và khôi phục optimizer/scheduler/scaler
cùng trạng thái RNG PyTorch. Phần epoch chưa lưu phải chạy lại. Không cam kết
resume khớp bit tuyệt đối với chạy liên tục (worker/CUDA có thể không tất định).
Nếu mất cả latest và best nhưng còn dữ liệu run, runner dừng để tránh ghi đè.
Nếu Kaggle chưa lưu được output trước khi phiên mất, không thể phục hồi bằng
chỉ việc mở lại notebook; checkpoint chưa lưu ra ngoài sẽ không tự xuất hiện.

## Thu kết quả và quyết định

Mỗi run cần `experiment_manifest.json`, `result.json`, `train.log`, thư mục
run có `metrics.json`, `config.json`, `checkpoint_best.pt`,
`checkpoint_latest.pt`, `best_encoder.pt`. `result.json` ghi hash checkpoint,
metrics, source và best/final validation accuracy. Không phải chỉ có `.pt`
là chứng minh hoàn thành đủ 20 epoch.

Sau khi nhận đủ ba trọng số trong một baseline:

1. So manifest: chỉ policy weight/output được khác; kiểm tra môi trường và
   encoder đồng nhất. Nếu khác, ghi rõ confound, chưa gọi là đối chứng sạch.
2. So best/final validation accuracy, đường train/val, thời gian/bộ nhớ.
3. Chạy paired selector utility audit trên checkpoint tốt nhất của từng run:
   selected context so với random/no-context trên cùng query và task weights.
   Accuracy tăng nhưng selector không có utility chưa đủ để nói selector tốt hơn.
4. Chọn ứng viên theo validation; chạy nhiều training seed để xác nhận.
   Chỉ sau khi khóa phương án mới đánh giá test và cập nhật bảng báo cáo cuối.

Đây là cải tiến thử nghiệm của nhóm trên bản triển khai hiện tại, không được
gắn nhãn baseline gốc của tác giả hoặc tuyên bố vượt paper từ pilot một seed.

## Kiểm thử local

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_policy_weight_pilot.py -v
```

Tests kiểm tra recipe, notebook syntax, bundle source, output isolation và
manifest guard; không giả lập accuracy hay thay thế chạy GPU thật trên Kaggle.
Kiểm tra local ngày 08/11/2026: 10/10 tests của bộ pilot và 7/7 tests multiseed
đạt; sáu file notebook đọc được JSON, các code cell và source Python trong
bundle parse thành công. Chưa kiểm tra cài dependency/network hoặc train
end-to-end trong runtime Kaggle. Source bundle SHA256 của bộ file đã tạo:
`e427116799fe9e98986c7109dffd6e2421209d4ac8d80fbf3c54f1c99111ecda`.
Hai trường hợp biên chính: encoder sai hash phải dừng; resume từ manifest
khác trọng số/source phải dừng mà không thay dữ liệu cũ.

## Nhật ký khởi chạy cloud — 08/11/2026

- Notebook riêng tư: [TACS Policy TwoStage w1 seed44](https://www.kaggle.com/code/b23dccn227trnngdng/tacs-policy-twostage-w1-seed44?scriptVersionId=366787768).
- Version 1, scriptVersionId `366787768`, tên `TwoStage w1 seed44 20ep validation`.
- Save & Run All lúc khoảng 09:38 giờ Việt Nam; Internet bật, GPU T4 x2 được
  cấp nhưng notebook giới hạn dùng GPU 0. Quota UI trước khi chạy: còn 30 giờ.
- Đính kèm output `TACS_Role_Pilot_seed44` làm input chỉ đọc, không ghi đè
  notebook/output cũ. Không cần upload lại encoder local.
- Log 18,2 giây xác minh source SHA256 đúng bundle ghi trên; log 29,0 giây
  xác minh encoder SHA256 `ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64`.
- Log GPU: Tesla T4; môi trường quan sát torch `2.11.0+cu130`, torchvision
  `0.26.0+cu130`, timm `1.0.30`. Các đối chứng tiếp theo cần kiểm tra cùng
  môi trường, không lấy môi trường local thay thế cho môi trường cloud.
- Log 200,6 giây: real data `36000 9000 5000`. Đã sửa lỗi tài liệu ghi
  nhầm 35.000 query_train; split local cũng xác nhận 36.000. Không đổi split.
- Log 213,0 giây: tạo cache candidate lần đầu; đây không phải lỗi dừng job.
- Log 581,7 giây: `[TWO_STAGE] starting at epoch 1/20 on device 'cuda' with AMP FP16...`.
- Output root: `/kaggle/working/policy_weight_two_stage_1p0_seed44`.

Trạng thái cuối quan sát trong lần triển khai: **Running, đã vào vòng train**.
Chưa xác nhận hoàn tất epoch 1 hay đủ 20 epoch; chưa có accuracy cuối, chưa
thu checkpoint/output mới. Chỉ run weight 1,0 được khởi chạy; weight 0,1 và
0,03 chưa chạy. Chờ marker complete và artifacts trước khi cập nhật kết quả.

## Cập nhật xác minh và lượt tiếp theo — 09/11/2026

### Two-Stage weight 1,0 hoàn tất

- Kiểm tra trực tiếp Version 1 `366787768`: log `successful`, thời gian
  6 giờ 47 phút 45 giây; đủ 20/20 epoch, marker `POLICY WEIGHT PILOT COMPLETE`.
- Best/final validation accuracy: **87,34%**; `test_evaluated=false`.
- Cell cuối kiểm tra hash checkpoint thành công. Output Kaggle có
  `experiment_manifest.json`, `result.json`, `train.log`; chưa tải checkpoint
  về local trong lần xác minh này.
- Checkpoint SHA256:
  `307fa8f7e6f1df9456179272d2911515e935ad617da9a5ff628f2a81a1fd1ae2`.
- Metrics SHA256:
  `98934eb0d6856df3aadf7aad608c9ea9c03939b886c9b3f589722d510818bd2c`.
- Đây là đối chứng một seed trên validation, chưa chứng minh cải tiến và
  không phải kết quả test cuối cùng.

### Two-Stage weight 0,1 đã khởi chạy

- Notebook riêng tư: [TACS Policy TwoStage w0p1 seed44](https://www.kaggle.com/code/b23dccn227trnngdng/tacs-policy-twostage-w0p1-seed44?scriptVersionId=367137294).
- Version 1 `367137294`, tên `TwoStage w0p1 seed44 20ep validation`,
  Save & Run All lúc 09:38 giờ Việt Nam ngày 09/11/2026.
- Sao chép notebook đối chứng; thay mô tả, `POLICY_WEIGHT=0.1`, output và
  đường dẫn ví dụ resume. Source bundle không đổi; `RESUME_FROM=None`.
  Không nạp task/selector từ checkpoint weight 1,0; không sửa output cũ.
- Output riêng: `/kaggle/working/policy_weight_two_stage_0p1_seed44`.
- Input giữ `TACS_Role_Pilot_seed44`; Internet bật, GPU T4 x2 nhưng source
  chỉ dùng GPU 0. UI quota trước khi chạy: đã dùng 06:47 / 30 giờ.
- Editor hiển thị `Pin to original environment (2026-11-03)`; trang Logs
  hiển thị `Latest Container Image`. Không coi hai nhãn UI là bằng chứng
  môi trường hoàn toàn giống nhau. Log thực tế xác nhận torch `2.11.0+cu130`,
  torchvision `0.26.0+cu130`, khớp đối chứng; cần so manifest đầy đủ sau run.
- Log 14,7 giây xác minh source hash như trên; log 22,2 giây xác minh
  encoder hash gốc và `Run: two_stage 0.1`; log 22,4 giây: `GPU: Tesla T4`.
- Trạng thái cuối xác minh: **Running, đang tải CIFAR-100**. Chưa quan sát
  epoch 1 hay accuracy mới; chưa được coi là hoàn tất train.
- Ảnh bằng chứng: `artifacts/policy_weight_pilot/kaggle_w0p1_seed44_started.jpg`.
- Thao tác Copy & edit mở tab mới, dẫn tới tạo thêm một bản nháp không dùng
  tên `Fork of TACS Policy TwoStage w1 seed44 f4387c`. Không khởi chạy bản
  nháp này; chỉ version `367137294` được Save & Run All.

Tiếp theo: xác minh complete và thu toàn bộ output weight 0,1; sau đó chạy
weight 0,03 theo cùng recipe, so manifest và paired selector utility trước
khi kết luận. Không resume chéo giữa các trọng số.

## Chuyển weight 0,03 sang Brave — 09/11/2026

- Theo yêu cầu người dùng do tài khoản trên Edge hết giờ, tạo bản riêng trên
  tài khoản `b23dccn227trnngdng` bằng Copy & edit từ đối chứng weight 0,1.
  Không sửa hoặc xóa notebook/output trên Edge hay các lượt trước.
- Xác minh trực tiếp weight 0,1 Version `367137294` đã successful sau
  6 giờ 33 phút 38 giây, đủ epoch 20/20; report `status=complete`, best
  validation **87,38%**, final validation **87,36%**. Chưa tải output về local.
  Chênh best so với weight 1,0 chỉ 0,04 điểm phần trăm, chưa chứng minh cải tiến.
- Notebook mới: [TwoStage w0p03 seed44](https://www.kaggle.com/code/b23dccn227trnngdng/tacs-policy-twostage-w0p03-seed44?scriptVersionId=367326145).
  Version 1 `367326145`, Save & Run All, tên `TwoStage w0p03 seed44 20ep validation`.
- Đã đặt `POLICY_WEIGHT=0.03`, `RESUME_FROM=None`, output riêng
  `/kaggle/working/policy_weight_two_stage_0p03_seed44` và cập nhật markdown.
  Giữ nguyên source bundle, seed 44, 20 epoch, validation-only; input
  `TACS_Role_Pilot_seed44` cung cấp encoder gốc. Không resume từ weight 0,1.
- Trước submit: GPU T4 x2, Internet on, UI pin environment 2026-11-03,
  quota đã dùng 13:21 / 30 giờ. Source chỉ dùng GPU 0.
- Trạng thái cuối quan sát: **Queued**, chưa có log xác minh source/encoder
  trong runtime mới, chưa xác nhận vào epoch 1. Không gọi là train hoàn tất.
- Bằng chứng: `artifacts/policy_weight_pilot/brave_w0p03_seed44_submitted.jpg`.
  Theo dõi Logs của đúng version trên; không bấm Run All thêm để tránh job trùng.

## Thu kết quả weight 0,03 — 10/11/2026

Nguồn: archive người dùng tải về `results (1).zip` ở thư mục gốc dự án.
Không suy ra Kaggle version ID từ tên notebook: archive không cung cấp ID
đủ để gắn kết quả này với lượt submit/queued trước đó.

`result.json` báo complete; `train.log` ghi epoch 1 đến 20 và Done.
Config xác nhận Two-Stage, policy weight 0,03, seed 44, role_v2,
20 epoch, batch 4 × accumulation 16, image size 256, AMP,
LR 1e-4, warm-up 5, top-k 32, temperature 0,1; không đánh giá test.
Manifest ghi torch 2.11.0+cu130, torchvision 0.26.0+cu130,
timm 1.0.30, Tesla T4. Config có `resume=true`, nhưng log bắt đầu epoch 1;
không được dùng cờ này một mình để kết luận có warm-start từ thí nghiệm khác.

| Two-Stage policy weight | Best validation | Final validation | Bằng chứng |
|---|---:|---:|---|
| 1,0 | 87,34% | 87,34% | Xác minh Kaggle đã ghi ở trên |
| 0,1 | 87,38% | 87,36% | Xác minh Kaggle đã ghi ở trên |
| 0,03 | 87,28% | 87,28% | Archive local, result và log |

Weight 0,03 thấp hơn 0,1 đúng 0,10 điểm phần trăm (5/5.000 ảnh),
thấp hơn 1,0 đúng 0,06 điểm phần trăm (3/5.000 ảnh). Đây chỉ là chênh lệch
một seed trên validation; chưa chứng minh có khác biệt ổn định hay cải tiến.
Weight 0,1 hiện là ứng viên có best validation cao nhất, không phải kết luận
selector tốt nhất. Không dùng các con số này làm test accuracy hoặc so trực tiếp
với kết quả paper khác giao thức.

Đã tính lại và xác nhận khớp SHA256 checkpoint best, metrics, encoder gốc
và split trực tiếp từ archive. Metrics xác nhận executed_epochs=20,
resumed_from_epoch=null: lượt này không tiếp tục từ epoch của checkpoint cũ.
Chưa kiểm tra toàn bộ CRC archive hoặc nạp tensor checkpoint.

Giải nén vào `artifacts/policy_weight_pilot/kaggle_w0p03_seed44` bị dừng vì
`OSError: [Errno 28] No space left on device`. Thư mục này là bản giải nén
**không đầy đủ**, không dùng để resume/audit. ZIP gốc không thay đổi.
Lệnh dọn bản giải nén dở bị công cụ chặn; cần giải phóng dung lượng trước
khi giải nén lại. Không đánh dấu artifact local đã sẵn sàng để inference.

Checkpoint best SHA256 đã xác minh:
`890966a5057a5ff143902069374cdcd86d93df6bfdb7341c667e0ce847f92fd5`.
Metrics SHA256:
`85b49d0b1e8de883559a62c01c5ab62052a568c5fcb6c7b6bc24c577bc3d6c6e`.

Bước tiếp theo: thu đủ artifact weight 1,0 và 0,1 nếu chưa có local;
đối chiếu manifest, chạy paired selector utility trên cùng validation
(selected / random cùng top-k / no-context, cố định task weights), rồi
mới quyết định cấu hình cho thí nghiệm nhiều seed. Không train lại weight
0,03 chỉ vì accuracy thấp hơn 0,1 và chưa cần đánh giá test.

### Khôi phục sau khi giải phóng dung lượng — 10/11/2026

Người dùng đã giải phóng dung lượng; kiểm tra ổ D còn 8.422.510.592 byte
trước giải nén. Đã giải nén đầy đủ experiment weight 0,03, encoder tham chiếu
và split vào `artifacts/policy_weight_pilot/kaggle_w0p03_seed44`.
Trạng thái bản giải nén không đầy đủ ở mục trước đã được khắc phục.
Không giải nén lại bộ CIFAR hoặc source không cần thiết; không ghi đè ZIP.

Tính lại hash từ file đã giải nén: checkpoint best, metrics, encoder tham
chiếu và split đều khớp result/manifest. Chạy `tacs.selector_audit` với
`--check-only --device cpu` thành công: `checkpoint_and_split_preflight_passed`.
Checkpoint nạp bằng weights_only, model state nạp strict; checkpoint epoch 20,
val_accuracy 0,8728; encoder và split tương thích. Không cập nhật weights,
không inference dữ liệu và không đánh giá test trong bước này.

Bước kế tiếp là paired utility audit trên dữ liệu thật, không phải train
tiếp epoch 21. Có thể chạy 256 query validation để kiểm tra pipeline trước,
sau đó toàn bộ 5.000 query với cùng cấu hình giữa các trọng số. Môi trường
local torch 2.14.1+cu126 khác môi trường train 2.11.0+cu130; phải ghi nhận
khác biệt và kiểm tra tái lập accuracy trước khi diễn giải kết quả audit.
