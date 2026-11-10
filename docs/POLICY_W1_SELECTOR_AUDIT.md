# Two-Stage policy weight 1,0 — tiếp nhận và audit

Ngày: 10/11/2026. Archive người dùng cung cấp: `results (2).zip`,
1.284.354.003 byte. Chỉ giải nén checkpoint best, config, metrics,
manifest, result và train.log vào `artifacts/policy_weight_pilot/kaggle_w1p0_seed44/`.
Không giải nén latest, best_encoder, cache hoặc data/source trùng lặp.
ZIP gốc giữ nguyên; các file tải dở khác không bị thay đổi.

## Xác minh thí nghiệm

- Two-Stage, CIFAR-100 thật, DINOv3 ViT-S/16, role_v2, seed 44, 20 epoch.
- Result complete; log đến epoch 20 và Done. Best/final validation 87,34%.
- Metrics xác nhận executed_epochs=20, resumed_from_epoch=null.
- Đã tính lại và xác nhận khớp hash checkpoint và metrics từ file giải nén.
- Checkpoint SHA256: `307fa8f7e6f1df9456179272d2911515e935ad617da9a5ff628f2a81a1fd1ae2`.
- Metrics SHA256: `98934eb0d6856df3aadf7aad608c9ea9c03939b886c9b3f589722d510818bd2c`.
- So config với weight 0,03: chỉ khác policy_weight và output_dir.
- Manifest khớp source SHA256, split SHA256, encoder SHA256, initialization
  và environment của weight 0,03. Source local model.py, data.py, retrieval.py,
  train.py, selector_audit.py khớp source trong ZIP weight 1,0.
- Encoder tham chiếu và split dùng lại từ bản giải nén weight 0,03 vì cùng
  hash, không nạp weights của task/selector weight 0,03 vào mô hình weight 1,0.

## Kết quả full validation

Hoàn tất 5.000 query không trùng trong 1.038,93 giây (khoảng 17 phút 19
giây). Giai đoạn tạo đặc trưng chậm bất thường; không thay đổi giao thức
giữa lượt chạy. RTX 3050 Laptop, CUDA AMP, torch 2.14.1+cu126, batch 4,
audit seed 44, random 3 trials trong cùng top-32; không train hoặc dùng test.

| Can thiệp trên cùng task weights | Đúng / 5.000 | Accuracy |
|---|---:|---:|
| Selector | 4.368 | 87,36% |
| Bỏ context | 4.353 | 87,06% |
| Random trial 0 | 4.364 | 87,28% |
| Random trial 1 | 4.368 | 87,36% |
| Random trial 2 | 4.363 | 87,26% |
| Random trung bình | — | 87,30% |

Selector hơn random trung bình 0,06 điểm phần trăm; hơn bỏ context 0,30
điểm phần trăm. Bỏ context là can thiệp với cùng task weights, không phải
baseline No-context train riêng. Context thay đổi dự đoán 88 ảnh: sửa đúng
41, làm sai thêm 26, tăng ròng 15 ảnh đúng so với bỏ context.

Mean CE: selected 0,41642927, bỏ context 0,42623883. Selected giảm CE
0,00980956 so với bỏ context và 0,00083984 so với random trung bình.
Selector dùng 1.473 context khác nhau; top-1 context chiếm 1,02%, top-10
chiếm 7,62%; cùng nhãn query 45,54%. Mean entropy 0,85098748,
mean xác suất top-1 0,68979041; không suy ra nguyên nhân/hiệu quả chỉ từ entropy.

Accuracy tái lập 87,36% lệch checkpoint 87,34% đúng 0,02 điểm phần trăm,
một ảnh đúng ròng. Đạt tolerance 0,1 điểm phần trăm nhưng không khớp tuyệt
đối. Môi trường torch/GPU khác train; chưa cô lập nguyên nhân chênh lệch.
Đã kiểm tra lại hash checkpoint sau audit: không đổi. Đếm lại predictions
khớp report; đủ 5.000 query duy nhất. Query order và context random của cả
ba trials khớp từng ảnh với lượt weight 0,03.

Artifact: `artifacts/policy_weight_pilot/audit_w1p0_seed44_full/`.

- Report SHA256: `2596d8b4704c7cf4128b8437db6cf14282843340f3005246d6cf785af0e395ee`.
- Queries SHA256: `ea249be1e7428d60b0effc193cb6c85ff4ec2c22ff209ed7bef24c213d63c39c`.

## So với weight 0,03

| Chỉ số full validation | Weight 1,0 | Weight 0,03 |
|---|---:|---:|
| Accuracy selector | 87,36% | 87,30% |
| Random trung bình | 87,30% | 87,42% |
| Bỏ context, cùng weights của từng run | 87,06% | 87,36% |
| Selector trừ random | +0,06 điểm % | −0,12 điểm % |
| Selector trừ bỏ context | +0,30 điểm % | −0,06 điểm % |

Các hàng trong mỗi cột cố định task weights; hai cột là hai mô hình được
train riêng. Không coi random hoặc bỏ context giữa các cột là cùng mô hình.
Ba random trials không phải ba training seed. Weight 1,0 hiện có dấu hiệu
lợi ích selector nhẹ hơn đối chứng; chênh lệch nhỏ trên một seed không chứng
minh khác biệt ổn định. Chưa thấy giảm xuống 0,03 cải thiện accuracy.

Tiếp theo: nhận artifact đúng policy weight 0,1, đối chiếu manifest và chạy
audit cùng giao thức trước khi chốt hướng train nhiều seed. Chưa tăng epoch,
chưa đánh giá test, chưa tuyên bố cải tiến so với paper.

## Lệnh tái chạy

Output phải là thư mục mới chưa tồn tại. Chỉ dùng validation, không train/test.

```powershell
$auditRoot = 'artifacts/policy_weight_pilot'
.\.venv\Scripts\python.exe -X utf8 -u -m tacs.selector_audit --checkpoint "$auditRoot/kaggle_w1p0_seed44/policy_weight_two_stage_1p0_seed44/cifar100/dinov3_vits16/role_v2/seed_44/two_stage/checkpoint_best.pt" --standard-encoder "$auditRoot/kaggle_w0p03_seed44/policy_source_e427116799fe/reference/no_context/best_encoder.pt" --split-file "$auditRoot/kaggle_w0p03_seed44/policy_source_e427116799fe/data/splits/cifar100_seed42.json" --data-root data --output "$auditRoot/audit_w1p0_seed44_recheck" --device cuda --max-queries 0 --random-trials 3 --batch-size 4
```

Hai điều kiện biên: config/split sai phải bị từ chối; thư mục output đã tồn
tại không được ghi đè. Audit dùng local torch khác train; phải kiểm tra
validation_reproduction trước khi kết luận.
