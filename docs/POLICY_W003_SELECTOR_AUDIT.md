# Two-Stage policy weight 0,03 — kiểm tra selector

Ngày thực hiện: 10/11/2026. Chỉ inference validation CIFAR-100 thật;
không train, không thay đổi checkpoint, không đánh giá test.

## Nguồn và giao thức

- Archive: `results (1).zip`, experiment `policy_weight_two_stage_0p03_seed44`.
- DINOv3 ViT-S/16, role_v2, training seed 44, checkpoint best epoch 20.
- Checkpoint SHA256: `890966a5057a5ff143902069374cdcd86d93df6bfdb7341c667e0ce847f92fd5`.
- Metrics SHA256: `85b49d0b1e8de883559a62c01c5ab62052a568c5fcb6c7b6bc24c577bc3d6c6e`.
- Split SHA256: `0559402db013819b631d1f2e9df39dbd14d191df52e45b29d1eb53a4b8a7d295`.
- Encoder cố định SHA256: `ed59d392f6e78e7faeb6ffd9b6421ddf864725b610f10d986a179f417d84ec64`.
- RTX 3050 Laptop, CUDA AMP, batch 4, audit seed 44, random 3 trials/query
  trong cùng top-32 được truy xuất từ pool 9.000 ảnh.
- Local torch 2.14.1+cu126 khác torch train 2.11.0+cu130. Hash source local
  `model.py`, `data.py`, `retrieval.py`, `train.py`, `selector_audit.py`
  khớp source tương ứng trong archive.
- Mỗi can thiệp dùng cùng task weights. Bỏ context ở đây không phải mô hình
  No-context được train độc lập. Ba random trials không phải ba training seed.

## Kiểm tra nhanh 256 query

Chạy thành công trong 154,09 giây; chọn ngẫu nhiên 256 query theo audit seed,
không dùng các query này như một tập test mới.

| Can thiệp | Accuracy |
|---|---:|
| Selector | 87,109375% |
| Bỏ context | 88,281250% |
| Random trial 0 | 87,890625% |
| Random trial 1 | 88,671875% |
| Random trial 2 | 88,281250% |
| Random trung bình | 88,281250% |

Selector thấp hơn cả hai đối chứng 1,171875 điểm phần trăm trên mẫu nhỏ.
Không dùng chênh lệch này để kết luận toàn validation hoặc cải tiến.
Artifact: `artifacts/policy_weight_pilot/audit_w0p03_seed44_256/`.

## Kết quả toàn bộ validation

Hoàn tất 5.000/5.000 query duy nhất trong 391,91 giây. Thư mục kết quả:
`artifacts/policy_weight_pilot/audit_w0p03_seed44_full/`.

| Can thiệp trên cùng task weights | Số đúng / 5.000 | Accuracy |
|---|---:|---:|
| Selector | 4.365 | 87,30% |
| Bỏ context | 4.368 | 87,36% |
| Random trial 0 | 4.376 | 87,52% |
| Random trial 1 | 4.375 | 87,50% |
| Random trial 2 | 4.362 | 87,24% |
| Random trung bình 3 trials | — | 87,42% |

Accuracy selector cao hơn giá trị checkpoint 87,28% đúng 0,02 điểm phần trăm
(một ảnh đúng ròng). Kiểm tra tái lập đạt tolerance 0,1 điểm phần trăm,
**không khớp tuyệt đối**. Phiên bản torch/GPU khác là yếu tố có thể ảnh hưởng
số học; chưa thực hiện thí nghiệm riêng để xác định nguyên nhân chênh lệch.

Selector thấp hơn random trung bình 0,12 điểm phần trăm và thấp hơn bỏ
context 0,06 điểm phần trăm. So với bỏ context, đổi dự đoán 99 ảnh, sửa đúng
39 ảnh nhưng làm sai thêm 42 ảnh: ròng ít hơn 3 ảnh đúng. Đây là mô tả trên
một checkpoint, không phải chứng minh khác biệt có ý nghĩa thống kê.

Mean cross-entropy: selected 0,41105772; bỏ context 0,41763637. Context giảm
loss trung bình 0,00657866 nhưng không tăng accuracy. So random trung bình,
selected tăng loss 0,00004709; chưa có bằng chứng rerank học được tốt hơn
random trong cùng top-32. Không nhầm giảm loss với tăng số dự đoán đúng.

Selector dùng 1.199 context khác nhau; context phổ biến nhất chiếm 1,72%,
top-10 chiếm 7,56%; cùng nhãn query ở 48,62% lượt. Mean entropy 3,44511210,
mean xác suất top-1 0,04616060. Không suy ra collapse hay nguyên nhân thất bại
chỉ từ các thống kê này.

Đã kiểm tra lại SHA256 checkpoint sau inference: không đổi. Đếm lại dự đoán
trong queries.jsonl khớp accuracy report, đủ 5.000 chỉ số query không trùng.

- Report SHA256: `dfa6a14db199f66da0885bd9d5bf99592840ac5f63683b3a93c34324fbd8b769`.
- Queries SHA256: `26cc39551897e50200f3425642de7c844934ccb067bbc5c47e2f8dda96378a73`.

### Kết luận và bước kế tiếp

Weight 0,03 chưa cho thấy cải thiện accuracy hoặc lợi thế selector trên
checkpoint này. Không tiếp tục tăng epoch chỉ dựa trên giả thuyết giảm policy
weight. Cần thu artifact của hai đối chứng policy weight 1,0 và 0,1 đã chạy
trên Kaggle (không thay bằng pilot cũ khác recipe), đối chiếu manifest rồi
audit cùng giao thức. Sau đó mới chốt cấu hình chạy nhiều training seed.
Chưa đánh giá test và chưa đưa ra kết luận vượt paper.

## Cách chạy lại

Chạy từ thư mục gốc dự án; output phải là thư mục mới chưa tồn tại.

```powershell
$auditRoot = 'artifacts/policy_weight_pilot/kaggle_w0p03_seed44'
.\.venv\Scripts\python.exe -X utf8 -u -m tacs.selector_audit --checkpoint "$auditRoot/policy_weight_two_stage_0p03_seed44/cifar100/dinov3_vits16/role_v2/seed_44/two_stage/checkpoint_best.pt" --standard-encoder "$auditRoot/policy_source_e427116799fe/reference/no_context/best_encoder.pt" --split-file "$auditRoot/policy_source_e427116799fe/data/splits/cifar100_seed42.json" --data-root data --output artifacts/policy_weight_pilot/audit_w0p03_seed44_recheck --device cuda --max-queries 0 --random-trials 3 --batch-size 4
```

Kiểm thử công cụ: `python -m unittest discover -s tests -p test_selector_audit.py -v`:
20/20 test đạt. Hai trường hợp biên: checkpoint/split sai phải bị từ chối;
ánh xạ top-k về pool phải đúng kể cả chỉ có một candidate.
