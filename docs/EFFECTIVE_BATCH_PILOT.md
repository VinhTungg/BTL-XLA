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
