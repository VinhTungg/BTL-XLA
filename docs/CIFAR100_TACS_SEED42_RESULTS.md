# Kết quả TACS trên CIFAR-100 — seed 42

Ngày thu thập: 2026-10-30.

Nguồn chạy: [Kaggle Version 3](https://www.kaggle.com/code/b23dccn912nguynthuyn/tacs-cifar100-seed42-tacs-resume-86-100?scriptVersionId=363328910).
Run cuối tiếp tục từ checkpoint epoch 85 và hoàn thành đủ 100 epoch trên Tesla T4,
AMP FP16. Dữ liệu dùng split cố định 36.000 query train, 9.000 candidate pool,
5.000 validation và 10.000 test.

## Cấu hình

| Tham số | Giá trị |
|---|---:|
| Baseline | TACS |
| Backbone | DINOv3 ViT-S/16 |
| Seed | 42 |
| Epoch | 100 |
| Batch size | 32 |
| Gradient accumulation | 2 |
| Effective batch size | 64 |
| Learning rate | 0,0001 |
| Top-K candidates | 32 |
| Selector temperature | 0,1 |
| Policy weight | 1,0 |

## Kết quả

| Chỉ số | Giá trị |
|---|---:|
| Best validation accuracy | **85,40%** |
| Validation accuracy tại epoch 100 | 81,02% |
| Test accuracy | **85,66%** |
| Test loss | 0,5490 |
| Test throughput | 86,16 query/s |
| Test latency | 11,6059 ms/query |
| Peak RAM | 2.388,29 MB |
| Peak VRAM allocated | 9.116,61 MB |

`metrics.json` chứa đủ 100 bản ghi lịch sử và `checkpoint_latest.pt` đã được
nạp lại để xác nhận `epoch=100`, `best_val_accuracy=0.854`.

## Artifact cục bộ

Các file được giải nén vào
`outputs/cifar100/dinov3_vits16/seed_42/tacs/`:

| File | Kích thước (byte) | SHA-256 |
|---|---:|---|
| `best_encoder.pt` | 86.414.011 | `57020267F1BCA56EBBDA244DEA7224C58DCDDF9F65E49124E8B1FE72B771F738` |
| `checkpoint_best.pt` | 172.995.575 | `469FF105D570992655D42655ECF271E8A3836397754C6236BD68BC736151C41A` |
| `checkpoint_latest.pt` | 519.032.361 | `BB619192E2F0782F641C51DFC95368B4D33745BC7EB023129C1FD9A38BD02C72` |
| `config.json` | 957 | `037E8A6E5B3F567763305DC1560907F22471146D8AFDF6C826877E6D27895667` |
| `metrics.json` | 112.820 | `B083200B28C890FDAF3550BCC8A0BA3C76EBBE1BB63F8EE74628AEC57CCE2533` |

Thư mục `outputs/` nằm trong `.gitignore`, nên checkpoint lớn không được đưa
vào Git. Tài liệu này là manifest có thể commit để lưu cấu hình, kết quả và
checksum kiểm chứng artifact.
