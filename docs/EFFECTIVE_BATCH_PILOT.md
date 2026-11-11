# Pilot chuẩn hóa advantage: microbatch vs effective batch

## Hai nhánh độc lập

| Trường | Đối chứng | Nhánh mới |
|---|---|---|
| advantage_scope | microbatch | effective_batch |
| Mẫu chuẩn hóa mỗi cửa sổ đầy đủ | 4 | 64 |
| Output root | advantage_microbatch_seed44 | advantage_effective_batch_seed44 |

Các yếu tố còn lại cố định: CIFAR-100 split seed42 (36.000/9.000/5.000),
Two-Stage role_v2, DINOv3 ViT-S/16 pretrained, training seed44, 20 epoch,
microbatch4 × accumulation16, LR1e-4, warm-up5, minLR0, WD1e-4,
policy_weight1, temperature0,1, top-k32, AMP, 2 workers; không đánh giá test.
Không nạp task/selector từ pilot cũ. Chỉ encoder truy xuất cố định được nạp
từ No-context gốc, SHA256:
`ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64`.

Mỗi notebook chỉ chạy một nhánh. Hai notebook nhúng cùng source bundle:
`1d85d0bbb17201c08e38b7428f720fccfe2859cc94cd0b256fac1103c7c9dce5`.
Source được kiểm tra hash trước giải nén; encoder/split/config/environment
được ghi manifest. Resume chỉ chấp nhận cùng manifest/source/environment,
có latest checkpoint; không resume từ weight khác hoặc scope khác.

## Notebook đã tạo

- `artifacts/effective_batch_pilot/Advantage_microbatch_seed44.ipynb`
- `artifacts/effective_batch_pilot/Advantage_effective_batch_seed44.ipynb`

Builder: `scripts/build_advantage_notebooks.py`; mặc định từ chối ghi đè
notebook đã có. Kiểm thử AST từng cell, bundle deterministic và chỉ một yếu
tố thay đổi đã đạt. Kaggle GPU/Internet cần bật; attach output
`TACS_Role_Pilot_seed44` để cung cấp encoder No-context khớp hash.

## Kiểm thử và so kết quả

Full suite: 130 tests, không lỗi, 2 integration opt-in bỏ qua. Smoke DINOv3
thật 64 mẫu đã đạt, xem `EFFECTIVE_BATCH_IMPLEMENTATION.md`. Smoke không thay
thế hai run 20 epoch đầy đủ và không chứng minh cải thiện accuracy.

Sau khi tải đủ output của cả hai nhánh:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_advantage_pilot.py --microbatch PATH_TO_MICROBATCH_ROOT --effective-batch PATH_TO_EFFECTIVE_ROOT
```

Comparator từ chối hash sai, manifest khác source/environment hoặc recipe
khác ngoài scope/output. Sau screening, dùng `tacs.selector_audit` với best
checkpoint để so selected/random/no-context, không chọn chỉ bằng accuracy.
Chỉ chạy nhiều seed nếu có tín hiệu đáng kiểm chứng; không mở test để chọn.

## Triển khai cloud — 11/11/2026

Colab tài khoản đang mở được kiểm tra trực tiếp: Resources hiển thị
`You are not subscribed`, trang gói hiển thị 0 compute units. Theo yêu cầu
người dùng, chuyển sang Kaggle, không mua/nâng cấp Colab.

Đã import nhánh microbatch qua UI (người dùng chọn file; agent bấm Import),
đặt tên TACS Advantage Microbatch seed44 và attach output pilot seed44 làm
input chỉ đọc. Internet bật, GPU T4 x2; source giới hạn CUDA_VISIBLE_DEVICES=0
để dùng một GPU. UI quota trước chạy: còn 10 giờ/tuần. Không hứa đủ hai nhánh.

Save & Run All Version 1: `Microbatch seed44 20ep validation`.
Kaggle run ID quan sát từ link Edit: `368151716`.
Viewer: https://www.kaggle.com/code/b23dccn227trnngdng/tacs-advantage-microbatch-seed44
Output dự kiến `/kaggle/working/advantage_microbatch_seed44`.

Trạng thái quan sát tại lúc tạo tài liệu: **Queued**, Latest Container Image,
GPU T4 x2, output 0 B. Chưa có runtime/source/encoder log hoặc epoch mới;
không gọi là đã train. Nhánh effective_batch chưa submit trên cloud.
Không bấm chạy lặp khi queued. Mọi thay đổi trạng thái phải được xác minh
trực tiếp trước khi cập nhật báo cáo.

## Kết quả microbatch — xác minh trên Kaggle 11/11/2026

Version 1, run `368151716` đã **successful**, runtime `24658.1 s`
(6 giờ 50 phút 58 giây), 54 file output, UI báo 1,24 GB.
Đã đọc trực tiếp `result.json`, `experiment_manifest.json` và `train.log`:

| Trường | Giá trị |
|---|---|
| Nhánh | `microbatch`, Two-Stage role_v2, seed44 |
| Epoch | 1–20, hoàn thành đủ 20 |
| Best validation | **87,92%**, epoch20 |
| Final validation | **87,92%** |
| Final train accuracy | 85,23% |
| Test | Chưa đánh giá (`test_evaluated=false`) |
| Tốc độ epoch cuối | 34,58 query/s |
| Môi trường | torch2.11.0+cu130, torchvision0.26.0+cu130, timm1.0.30, Tesla T4 |

Manifest khớp bundle/split/encoder đã định trước; không warm-start task/selector.
Source-tree SHA256:
`0b152ef319e324cb4af03de3875963b2458f6fff59d16604381b41495137efb0`.
Hash **do báo cáo cloud cung cấp**, chưa đồng nghĩa đã kiểm tra file local:

- Best checkpoint: `fbc50a26867d321a32892f262a9d447f45e3be95dd0a3a345053cf04ee8425d8`.
- Metrics: `45067f3951d47cf63f7d5647409eb446c42ed13358c20941c7ba2066f831d6e0`.

Đã yêu cầu Download output qua UI; file tải tạm xuất hiện trong Downloads.
Chưa gọi là đã thu hồi checkpoint cho tới khi ZIP tải đủ, đọc được và hash local
khớp. Bằng chứng UI: `artifacts/effective_batch_pilot/kaggle_microbatch_complete.jpg`.

Quota sau lượt chạy hiển thị `26:42 / 30 hrs`, tương ứng còn khoảng 3 giờ 18 phút
theo hạn mức đó. Draft session đang off; không khởi chạy lại microbatch và chưa
submit effective_batch. Nhánh mới cần hai forward pass nên không giả định đủ
quota để hoàn thành 20 epoch. Cần quota đủ hoặc kế hoạch chia chặng có lưu output
và resume cùng manifest trước khi submit.

### Bước tiếp theo

1. Tải đủ output; giữ `advantage_microbatch_seed44` với manifest, result, log,
   `run/config.json`, `metrics.json`, best/latest checkpoint và best encoder.
   Không xóa output cloud trước khi xác minh local.
2. Kiểm tra hash, history1–20 và nội dung checkpoint. Không dùng ZIP đang tải dở.
3. Chạy `Advantage_effective_batch_seed44.ipynb` từ **cùng bundle**, khởi tạo mới,
   output riêng. Không resume effective_batch từ checkpoint microbatch.
4. Dùng comparator ở trên rồi audit selected/random/no-context trên validation
   của cả hai best checkpoint. Chưa có kết quả effective_batch nên chưa kết luận
   chuẩn hóa advantage cải thiện accuracy; không so hai nhánh bằng test.
