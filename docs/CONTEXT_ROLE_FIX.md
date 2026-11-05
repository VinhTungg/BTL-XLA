# Sửa vai trò query/context trong DINOv3 — role_v2

Ngày: 04/11/2026. Phạm vi: lỗi kiến trúc task network; không thay selector,
reward, policy loss, cách lấy top-k, split hay pretrained backbone.

## 1. Bằng chứng và nguyên nhân

Artifact người dùng cung cấp: `cifar100/dinov3_vits16/seed_44/`, năm run
đều có lịch sử 100 epoch. Test ở đây là checkpoint best-validation,
không phải mặc nhiên checkpoint epoch 100.

| Phương pháp legacy_v1 | Best validation (%) | Test (%) |
|---|---:|---:|
| No-context | 86,86 | 87,07 |
| Random | 39,28 | 39,26 |
| Similarity | 74,98 | 75,36 |
| TACS | 85,54 | 85,54 |
| Two-Stage | 75,94 | 75,68 |

Mạng DINOv3 cũ ghép patch query và context, dùng chung CLS và lặp lại cùng
tọa độ RoPE cho hai nhóm patch, nhưng không có dấu phân biệt vai trò ảnh.
Ở chế độ evaluation, hoán đổi hai ảnh vì vậy gần như không đổi logits (điểm
phân loại trước softmax), trong khi nhãn cần dự đoán là nhãn của query.

Đây là lỗi đối xứng có thể kiểm chứng bằng source và phép thử. Chênh lệch
augmentation query/context khi train có thể tạo dấu hiệu vai trò không còn
ở evaluation; đây là giả thuyết bổ sung, chưa được chứng minh là nguyên nhân
duy nhất của accuracy thấp. Không kết luận role_v2 sẽ tự động vượt No-context.

## 2. Thay đổi

- DINOv3 `role_v2` thêm một vector học được có 384 phần tử vào mỗi patch
  context, khởi tạo truncated normal với std 0,02. Patch query không đổi;
  khởi tạo vector mới không làm dịch chuỗi số ngẫu nhiên cho phần train sau đó.
- Tiny đã có query/context type embeddings nên không đổi kiến trúc.
- `legacy_v1` giữ cấu trúc tham số cũ để strict-load checkpoint cũ.
- Config/metrics/checkpoint ghi `task_network_version`. Config cũ thiếu
  trường này được hiểu là `legacy_v1`, không tự nâng cấp trọng số.
- CLI trainer và multiseed có `--task-network-version`. Mặc định run mới
  là `role_v2`; DINOv3 có namespace output riêng. Notebook dùng cùng version
  cho lệnh train, xem tiến độ và xác minh kết quả.
- Resume khác version bị chặn. Nếu chỉ định output cũ, cả `--no-resume`
  cũng không được ghi đè sang version khác. Resume thất bại không ghi đè
  `config.json` trước đó. Tổng hợp multiseed từ chối trộn version.

Đây là **bản sửa triển khai do nhóm lựa chọn**, chưa có cơ sở gọi là chi tiết
kiến trúc chính thức của tác giả. Không thay đổi kết quả lịch sử hoặc gọi
checkpoint legacy đã train 100 epoch là checkpoint role_v2.

## 3. Kiểm chứng trên checkpoint thật

Chạy CPU FP32, seed khởi tạo vector mới 44, checkpoint
`cifar100/dinov3_vits16/seed_44/random/checkpoint_best.pt`.
Dùng bốn query đầu trong validation và bốn context đầu trong candidate pool,
transform evaluation DINOv3 của project, không đánh giá test.

Giữ nguyên toàn bộ task weights cũ ở cả hai mạng; chỉ mạng role_v2 được thêm
vector mới trong RAM để chẩn đoán. Script không chuyển đổi hay ghi checkpoint.

| Kiểm tra | Kết quả |
|---|---:|
| Max chênh lệch logits khi đổi query/context, legacy | 0,0000033379 |
| Chênh lệch logits khi đổi query/context, role_v2, theo 4 cặp | 0,07950; 0,11336; 0,07111; 0,06782 |
| Max chênh lệch nhánh query-only giữa hai mạng | 0,0 |
| Tổng trị tuyệt đối gradient của role embedding | 0,02044885 |
| Tổng trị tuyệt đối gradient của context proxy | 0,02044885 |

Kết quả xác nhận mất đối xứng đổi vai trò, nhánh query-only không đổi và
gradient không bị cắt. **Không phải kết quả accuracy của role_v2 sau train.**

Unit test kiểm tra đối xứng, query-only, gradient, input sai shape, cùng ảnh
ở hai vai trò, version không hợp lệ và bảo vệ output/resume. Fixture DINOv3
không tải pretrained đặt LayerScale=0,1 để phép thử không bị triệt bởi
LayerScale=1e-5 của mạng chưa train; kiểm tra checkpoint thật bên trên không
thay LayerScale. Test tích hợp tùy chọn dùng hai ảnh CIFAR-100 thật, mô phỏng
ngắt trước epoch 2, rồi kiểm tra resume từ epoch 1 sang epoch 2 với optimizer
và checkpoint role_v2. Đó là kiểm tra kỹ thuật, không phải benchmark accuracy.

Kết quả kiểm tra cuối ngày 04/11/2026: **71/71 test đạt** khi bật cả hai
integration dữ liệu thật. Đã kiểm thử CPU FP32; chưa chạy benchmark GPU/AMP
cho role_v2. Notebook đã được kiểm tra JSON và cú pháp của cả 11 code cell.

## 4. Chạy lại kiểm tra

Từ thư mục gốc repo, PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_context_roles.py --checkpoint cifar100/dinov3_vits16/seed_44/random/checkpoint_best.pt

$env:TACS_RUN_REAL_CIFAR_TESTS = '1'
.\.venv\Scripts\python.exe -X utf8 -c "import torch, unittest; torch.set_num_threads(2); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

Không có dữ liệu thật thì bỏ biến môi trường để skip integration; unit tests
vẫn chạy không cần tải model. Diagnostic checkpoint yêu cầu sẵn CIFAR-100 và
split; không tự tải dataset hoặc dùng dữ liệu giả thay thế.

Hai trường hợp biên quan trọng: context khác batch/kích thước query phải báo
lỗi rõ; đưa checkpoint thiếu version vào run mới phải bị từ chối mà không
sửa artifact cũ.

## 5. Train tiếp như thế nào

Không cần xóa hoặc train lại ngay cả năm run. Giữ nguyên seed 44 legacy để
đối chiếu. Bước tiếp theo là pilot **Random legacy_v1 vs role_v2** cùng seed,
khởi tạo pretrained, split, batch, LR và số epoch. Đây là phép thử rẻ hơn
để tách ảnh hưởng của role embedding khỏi chất lượng selector TACS.

Ví dụ xem trước lệnh pilot 20 epoch trên GPU đủ chứa batch 64:

```powershell
foreach ($version in @('legacy_v1', 'role_v2')) {
    .\.venv\Scripts\python.exe -m tacs.multiseed --dataset cifar100 --backbone dinov3_vits16 --task-network-version $version --seeds 44 --baselines random --epochs 20 --batch-size 64 --lr 0.0001 --warmup-epochs 5 --num-workers 2 --device cuda --outputs-root outputs/context_role_pilot --no-eval-test --plan-only
}
```

Bỏ `--plan-only` khi muốn bắt đầu train. Pilot này dùng cosine 20 epoch;
không resume nó thành run chuẩn 100 epoch hoặc so trực tiếp với legacy
100 epoch để khẳng định cải tiến. Nếu validation ủng hộ bản sửa, chạy
thí nghiệm 100 epoch mới từ pretrained, sau đó kiểm tra TACS và Two-Stage
trong cùng protocol. Chỉ đánh giá test sau khi khóa cấu hình; chưa cần
train tất cả seed trong giai đoạn thử lỗi này.

Run multiseed mới có dạng:
`outputs/<dataset>/<backbone>/role_v2/seed_44/<baseline>/`.
Trên Kaggle/Colab phải cập nhật source hoặc bundle mới trước khi chạy;
chỉ đổi cell cấu hình trên source cũ không mang bản sửa này lên cloud.
Nếu muốn tiếp tục run legacy đang dở, chọn rõ `legacy_v1` và đường dẫn cũ.

## 6. Đính chính encoder truy xuất

Trong `BaselineModel`, `encoder` dùng truy xuất là một mạng riêng bị freeze;
task classifier train `task_network.backbone`. `best_encoder.pt` của
No-context xuất mạng riêng đó, **không phải backbone task đã fine-tune**.
Vì vậy không gọi Similarity hiện tại là `finetuned_similarity_legacy`
chỉ vì checkpoint được đặt dưới thư mục No-context. Dependency đường dẫn
No-context vẫn tồn tại; việc tách trực tiếp pretrained frozen encoder và
audit protocol paper là nhiệm vụ khác, không bị âm thầm thay trong bản sửa này.

## 7. Notebook Kaggle tự chứa source — 05/11/2026

Chạy `python scripts/build_kaggle_role_pilot.py` sau khi tạo bundle tại
`artifacts/role_v2/TACS_Colab_Bundle.zip`. Notebook sinh ra là
`artifacts/role_v2/TACS_Kaggle_Role_Pilot.ipynb`, nhúng bundle có SHA256,
không nhúng dữ liệu hoặc checkpoint. Trên Kaggle nhập bằng File → Import
Notebook, bật Internet và GPU T4 x2, rồi Save Version → Save & Run All.

Recipe riêng cho T4: Random, seed 44, 20 epoch mỗi version, batch vật lý 4,
tích lũy 16 bước, LR 1e-4, warm-up 5; chỉ validation, không test. Khi có hai
GPU thì mỗi version chạy trên một GPU; một GPU thì chạy tuần tự. Source mới
được giải nén riêng, không phụ thuộc dataset source legacy. Logs, best/latest
checkpoint và `pilot_comparison.json` lưu trong
`/kaggle/working/TACS_context_role_pilot`. Chỉ coi hoàn thành khi có dòng
`PILOT COMPLETE` và cả hai run đủ 20 epoch.

## 8. Kết quả pilot Random — xác minh ngày 05/11/2026

Kaggle `b23dccn227trnngdng/tacs-role-pilot-seed44`, Version 1,
scriptVersionId `365804474`: thành công, 15418,2 giây, output hiển thị 1,57 GB.
Đã đọc trực tiếp `TACS_context_role_pilot/pilot_comparison.json` trên Output:

| Version | Epoch | Best validation | Final validation |
|---|---:|---:|---:|
| legacy_v1 | 20 | 37,28% | 36,30% |
| role_v2 | 20 | 87,40% | 87,28% |

Seed 44, `test_evaluated=false`. Chênh lệch best validation +50,12 điểm
phần trăm. Đây là hiệu quả bản sửa triển khai trong pilot một seed,
không phải cải tiến vượt paper hoặc kết quả test. Chưa xác nhận tải toàn bộ
checkpoint về máy; artifacts gốc vẫn ở Kaggle Version 1.

## 9. Pilot selector role_v2

Sinh notebook bằng `python scripts/build_kaggle_role_pilot.py --selector-pilot`.
File: `artifacts/role_v2/TACS_Kaggle_Selector_Pilot.ipynb`.
Recipe giữ nguyên pilot Random; ba baseline No-context, TACS, Two-Stage,
seed 44, 20 epoch, validation only. TACS chạy GPU 0, No-context rồi Two-Stage
chạy GPU 1; nếu chỉ một GPU thì chạy tuần tự. Two-Stage chỉ bắt đầu sau khi
No-context cùng pilot hoàn tất và cung cấp encoder có provenance hợp lệ.
Không đổi nhãn encoder Random thành No-context, không warm-start task weights
từ Random, không sửa thuật toán selector. Encoder truy xuất của No-context
vẫn là pretrained frozen như phần 6, không gọi là fine-tuned encoder.

Output mới `/kaggle/working/TACS_selector_role_v2_pilot`, có log và report
riêng mỗi baseline; report chung `selector_pilot_comparison.json` chỉ ghi
khi cả ba run hoàn tất. Việc tạo notebook không có nghĩa đã chạy trên cloud.
Giữ Version 1 Random để đối chiếu; không resume cosine 20 epoch thành
benchmark 100 epoch. Trước benchmark cần kiểm tra nhiều seed và khóa recipe.

Khởi chạy trên Kaggle ngày 05/11/2026 lúc 23:31 (Asia/Saigon): notebook
`b23dccn227trnngdng/tacs-role-pilot-seed44`, **Version 2**, scriptVersionId
`366021012`, Save & Run All, GPU T4 x2, Internet bật. Cell train được thay
bằng `scripts/kaggle_selector_pilot_train.py`; giữ source bundle SHA256
`2bb4f915aa30cbb43dbeed7f18c5b5ecbfbcccacba1916e127b8d5b7edcad4c9`.
Đã thấy trạng thái Running; đây là xác nhận job đã được khởi chạy,
không phải xác nhận train hoàn tất hoặc accuracy. Version 1 Random được giữ.
