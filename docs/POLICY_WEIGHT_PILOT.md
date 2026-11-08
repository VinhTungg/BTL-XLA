# Pilot đối chứng policy weight — 08/11/2026

## Mục tiêu và trạng thái

Kiểm tra giảm trọng số policy loss có giúp selector chọn context hữu ích hơn
hay không. Gradient audit gợi ý thử nghiệm này, **không chứng minh** giảm
trọng số sẽ tăng accuracy. Bộ notebook đã được tạo; chưa chạy train cloud,
chưa có kết quả accuracy cho các cấu hình mới.

Thiết kế gồm TACS / Two-Stage × policy_weight 1,0 / 0,1 / 0,03. Mỗi notebook
chỉ chạy một thí nghiệm, một GPU, không chạy cả sáu job liên tục. Nếu ngân
sách hạn chế, hoàn tất ba cấu hình Two-Stage trước để có một so sánh trọn vẹn.
Ba thành viên có thể chia nhau ba trọng số, nhưng phải dùng cùng loại GPU và
phiên bản thư viện; so manifest sau khi thu kết quả. Không coi ba trọng số là
ba seed độc lập.

## Các yếu tố giữ cố định

- CIFAR-100 thật; split `cifar100_seed42.json`; query_train 35.000,
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
