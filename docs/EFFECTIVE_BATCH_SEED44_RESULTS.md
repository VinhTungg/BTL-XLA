# Nghiệm thu pilot effective-batch seed44

Ngày xác minh local: 12/11/2026. Đây là CIFAR-100 / DINOv3 ViT-S/16,
Two-Stage `role_v2`, 20 epoch, **validation-only**, không phải kết quả test
hay tái hiện toàn bộ bài báo.

## Thu hồi artifact

- ZIP gốc: `artifacts/advantage_effective_batch_seed44-20261112T142703Z-1-001.zip`.
- Dung lượng ZIP: 733.409.985 byte; 9 entry, toàn bộ vượt kiểm tra CRC.
- ZIP SHA256: `fddbf09d6a84ccc58df1e1c08d5127b7cd7ec0725df6d2507f87d4616156c6d4`.
- Giải nén vào thư mục mới, không ghi đè:
  `artifacts/effective_batch_pilot/colab_effective_batch_seed44/advantage_effective_batch_seed44`.
- Tổng dung lượng giải nén: 792.462.328 byte, bao gồm embedding cache.
- Giữ nguyên ZIP gốc và bản trên Drive; checkpoint không đưa vào Git.

| Artifact | SHA256 |
|---|---|
| `run/checkpoint_best.pt` | `5e7406aa36bfd16e20628f27c61f3238db3926b1afe12cc552ddb2618b845fa4` |
| `run/checkpoint_latest.pt` | `e82b112d29e18be6efa6f0c1cdfac79493af59bef23384075c7b3f8c0fe8e5b8` |
| `run/metrics.json` | `b5fb71bfb112cc46ca2f701e47c4ef99d0b3c956cca1def6c02dfc1b8d56f103` |
| `run/best_encoder.pt` | `202a201599695f00a70d15febed4227fd97bd5425ec55ed7eab35727e0c17362` |

Đã đọc best/latest/encoder bằng `torch.load(map_location='cpu', weights_only=True)`.
Tensor trọng số hữu hạn; best tại epoch17, accuracy0,878; latest tại epoch20,
accuracy0,8778. Latest có đủ history1–20 khớp metrics, optimizer/scheduler/scaler,
RNG CPU/CUDA. Đây là kiểm tra cấu trúc checkpoint, không chạy thêm epoch.
`config.json`, config trong metrics/manifest/best/latest khớp nhau.

## Định danh thí nghiệm

`compare_advantage_pilot.py` trả về `validated_screening_comparison` khi so
với đối chứng local tại
`artifacts/effective_batch_pilot/kaggle_microbatch_seed44/advantage_microbatch_seed44`.
Không sửa comparator hay artifact để làm khớp.

- Source tree: `0b152ef319e324cb4af03de3875963b2458f6fff59d16604381b41495137efb0`.
- Source bundle: `1d85d0bbb17201c08e38b7428f720fccfe2859cc94cd0b256fac1103c7c9dce5`.
- Split: `0559402db013819b631d1f2e9df39dbd14d191df52e45b29d1eb53a4b8a7d295`.
- Reference encoder: `ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64`.
- Environment hai manifest: torch2.11.0+cu130, torchvision0.26.0+cu130,
  timm1.0.30, Tesla T4.
- Source tree/bundle và split còn được tính lại từ checkout local và khớp manifest.
- Recipe khớp ngoại trừ `advantage_scope` và `output_dir` như thiết kế.
  Cùng seed44, microbatch4 × accumulation16, khởi tạo task/selector mới,
  policy weight1, top-k32, LR1e-4, warm-up5, AMP.
- Dữ liệu thật: query36.000 / pool9.000 / validation5.000; không đánh giá test.

## Kết quả sàng lọc

| Chỉ số | microbatch | effective_batch |
|---|---:|---:|
| Best validation | 87,92% | 87,80% |
| Epoch best | 20 | 17 |
| Final validation | 87,92% | 87,78% |
| History | 1–20 | 1–20 |
| Tổng thời gian train ghi trong metrics | 23.324,09 s | 41.941,80 s |
| Test | Không đánh giá | Không đánh giá |

Best delta effective minus micro = **−0,12 điểm phần trăm**, tương đương
net6 dự đoán đúng ít hơn trên5.000 ảnh, không có nghĩa chỉ6 ảnh dự đoán khác nhau.
Final delta = −0,14 điểm phần trăm. Thời gian ghi nhận tăng khoảng1,80lần.
Đây không phải benchmark tốc độ thuần: nền tảng Colab/Drive và Kaggle khác nhau,
effective-batch có thêm forward replay và nhiều lần resume.

`train.log` ghi bắt đầu epoch1, resume sau epoch8,9,14 rồi hoàn thành20.
Không giả định đường chạy resume giống hệt một lượt liên tục bit-for-bit.
Manifest khớp các trường được ghi không chứng minh toàn bộ driver/phần cứng phụ
và trạng thái runtime đều giống nhau.

**Quyết định hiện tại:** chưa có bằng chứng effective-batch cải thiện accuracy;
không chọn nó làm phương án thắng và chưa mở rộng train nhiều seed chỉ dựa vào
kết quả này. Một seed và chênh lệch nhỏ không đủ kết luận ưu thế thống kê.
Không dùng test để quyết định sửa tiếp.

## Selector audit: đã qua preflight, chưa inference

Đã chạy `tacs.selector_audit --check-only --device cpu` cho **cả hai best**:
`checkpoint_and_split_preflight_passed`. Model load strict thành công,
checkpoint/metrics/split và reference encoder khớp. Không train, không cập nhật
trọng số, không mở test, không inference ảnh; accuracy chưa được tái đo local.
Môi trường preflight local torch2.14.1+cu126 khác môi trường training; không
trình bày preflight này như một phép tái đo trên môi trường T4 gốc.

Reference encoder có sẵn, đã xác minh hash:
`artifacts/policy_weight_pilot/kaggle_w0p03_seed44/policy_source_e427116799fe/reference/no_context/best_encoder.pt`.

Bước tiếp theo là audit hai checkpoint trên cùng validation queries, auditseed44,
3 random trials, cùng chế độ precision/môi trường và cùng top-k32. So selected,
random trong cùng retrieval top-k và no-context dùng chính task weights của từng
checkpoint. No-context intervention không phải baseline No-context train riêng.
Ưu tiên full5.000 queries để tái kiểm tra validation; nếu chỉ screening256query
phải ghi rõ subset, không so accuracy subset với số fullvalidation trên.

Chưa tạo kết quả audit inference hoặc tuyên bố selector hữu ích. Không tự khởi
chạy một công việc GPU kéo dài trong bước nghiệm thu ZIP.

## Chạy lại kiểm tra

Chạy tại repository root (PowerShell):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_advantage_pilot.py --microbatch artifacts/effective_batch_pilot/kaggle_microbatch_seed44/advantage_microbatch_seed44 --effective-batch artifacts/effective_batch_pilot/colab_effective_batch_seed44/advantage_effective_batch_seed44
```

Preflight mẫu cho effective-batch (không tạo output và không inference):

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.selector_audit --checkpoint artifacts/effective_batch_pilot/colab_effective_batch_seed44/advantage_effective_batch_seed44/run/checkpoint_best.pt --standard-encoder artifacts/policy_weight_pilot/kaggle_w0p03_seed44/policy_source_e427116799fe/reference/no_context/best_encoder.pt --output artifacts/effective_batch_pilot/audit_effective_batch_seed44 --device cpu --check-only
```

Trường hợp biên: hash artifact sai phải bị comparator từ chối; checkpoint khác
split/reference encoder phải bị audit preflight từ chối, không ghi đè kết quả.
