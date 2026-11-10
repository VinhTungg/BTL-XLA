# Two-Stage policy weight 0,1 — tiếp nhận và audit

Ngày tiếp nhận: 10/11/2026. Archive `results (1).zip`, 1.284.366.504 byte,
hiện chứa `policy_weight_two_stage_0p1_seed44`. Tên ZIP này trước đó từng
chứa experiment weight 0,03: không dùng tên ZIP làm định danh thí nghiệm.
Artifact weight 0,03 đã giải nén và report của lượt trước vẫn được giữ riêng.

## Xác minh đầu vào

- CIFAR-100 thật, DINOv3 ViT-S/16, role_v2, Two-Stage, seed 44, 20 epoch.
- Result complete, best validation 87,38%, final validation 87,36%; không test.
- Metrics ghi executed_epochs=20, resumed_from_epoch=null.
- Checkpoint SHA256 đã tính lại khớp: `ccc8fdf63ea04edfbc44c222a1799e08680d4b52f4772c3432fa7fb14dddd39a`.
- Metrics SHA256 đã tính lại khớp: `502066212fd97be6b1dc30b4053fcab5706fb93d19822130d89b22ac0640c6cf`.
- Config so với weight 1,0 chỉ khác policy_weight và output_dir.
- Manifest khớp source, split, encoder tham chiếu, environment, initialization.
  Source model.py, data.py, retrieval.py, train.py, selector_audit.py local
  khớp source tương ứng trong archive.
- Chỉ giải nén checkpoint best và metadata/log cần thiết vào
  `artifacts/policy_weight_pilot/kaggle_w0p1_seed44/`; không nhân đôi dataset
  hoặc checkpoint latest. Dùng lại encoder/split có cùng SHA256 từ thư mục
  weight 0,03, không dùng weights task/selector của run khác.

## Lịch sử chạy lại sau lỗi dung lượng

Lượt đầu đã tạo xong các bank nhưng dừng trước inference query do
`OSError: [Errno 28] No space left on device`. Thư mục
`artifacts/policy_weight_pilot/audit_w0p1_seed44_full/` có manifest rỗng,
không phải kết quả hợp lệ. Không dùng artifact này để kết luận hoặc resume.

Người dùng đã dọn hai file tải dở được chỉ định. Trước khi chạy lại, xác nhận
cả hai không còn và ổ D trống 2.568.740.864 byte. Chạy lại ở thư mục riêng
`artifacts/policy_weight_pilot/audit_w0p1_seed44_full_retry/`, không ghi đè
thư mục lỗi, ZIP, checkpoint hoặc kết quả weight khác.

## Kết quả full validation — 11/11/2026

Lượt retry hoàn tất 5.000 query duy nhất, 527,55 giây. Best checkpoint ở
epoch 17, không phải checkpoint latest epoch 20. Inference trên RTX 3050
Laptop, CUDA AMP, batch 4, audit seed 44, random 3 trials trong cùng top-32.
Local torch 2.14.1+cu126 khác torch train 2.11.0+cu130; accuracy tái lập
87,38% **khớp chính xác** accuracy checkpoint, delta 0.

| Can thiệp trên cùng task weights | Đúng / 5.000 | Accuracy |
|---|---:|---:|
| Selector | 4.369 | 87,38% |
| Bỏ context | 4.357 | 87,14% |
| Random trial 0 | 4.373 | 87,46% |
| Random trial 1 | 4.364 | 87,28% |
| Random trial 2 | 4.377 | 87,54% |
| Random trung bình | — | 87,426667% |

Selector hơn bỏ context 0,24 điểm phần trăm (12 ảnh đúng ròng), nhưng
thấp hơn random trung bình 0,046667 điểm phần trăm. Context thay đổi nhãn
dự đoán ở 122 ảnh, sửa đúng 51 và làm sai thêm 39 so với bỏ context.
Đối chứng bỏ context dùng cùng task weights, không phải baseline train riêng.

Mean CE selected 0,41974034, bỏ context 0,42657715; giảm 0,00683681.
So random trung bình, selected giảm CE 0,00033979 dù accuracy thấp hơn:
giảm loss không đồng nghĩa tăng số dự đoán đúng.

Selector dùng 1.131 context; context phổ biến nhất 0,92%, top-10 7,64%,
cùng nhãn query 45,48%. Mean entropy 2,58418794, mean xác suất top-1
0,24554287. Đây là thống kê mô tả, không chứng minh nguyên nhân hoặc cải tiến.

Kiểm tra sau chạy: hash checkpoint không đổi; đếm lại predictions khớp
report; 5.000 query không trùng; query order và context random ở từng trial
khớp cả hai lượt weight 1,0 và 0,03. Không train hoặc đánh giá test.
20/20 unit tests công cụ audit đạt trong lượt kiểm tra này.

Artifact hợp lệ: `artifacts/policy_weight_pilot/audit_w0p1_seed44_full_retry/`.

- Report SHA256: `e5731c08d7994dce3cbd985c678aa8cd4ecb586fc5654c9da246fdd3794db43a`.
- Queries SHA256: `27d27f05f6e032495d21377672feba27af0e45edd6b0862f7e48c42a9247a1c9`.

## Tổng hợp ba policy weight (cùng training seed 44)

| Full validation audit | Weight 1,0 | Weight 0,1 | Weight 0,03 |
|---|---:|---:|---:|
| Selector accuracy | 87,36% | 87,38% | 87,30% |
| Random trung bình | 87,30% | 87,426667% | 87,42% |
| Bỏ context, cùng task weights của từng run | 87,06% | 87,14% | 87,36% |
| Selector trừ random (điểm %) | +0,060000 | −0,046667 | −0,120000 |
| Selector trừ bỏ context (điểm %) | +0,30 | +0,24 | −0,06 |

Trong mỗi cột, chỉ thay can thiệp context; giữa các cột là model được train
riêng. Không coi ba random trials là ba training seed. Weight 0,1 có accuracy
selector cao nhất nhưng hơn 1,0 chỉ một ảnh đúng trong audit local; selection
checkpoint của hai run và sai khác tái lập nhỏ cũng phải được ghi nhận.
Chưa có bằng chứng giảm policy weight cải thiện ổn định hay selector vượt
random một cách đáng kể. Không dùng pilot này để tuyên bố vượt paper.

Hoàn tất pilot ba weight Two-Stage. Không tự tăng epoch, mở test hoặc tiếp
tục quét weight. Nếu kiểm chứng tuning cần nhiều training seed; nếu cần phát
triển phương pháp mới, phải tách giả thuyết và đối chứng khỏi pilot này.

Lưu ý lưu trữ: trong lượt retry dung lượng D giảm mạnh; `git count-objects`
báo khoảng 1,93 GiB garbage ở ba tmp_obj và kho loose objects lớn. Không
xóa/sửa Git objects trong task này; cần xử lý dung lượng riêng trước job mới.

## Tái chạy

Chỉ inference validation; output phải mới, chưa tồn tại.

```powershell
$auditRoot = 'artifacts/policy_weight_pilot'
.\.venv\Scripts\python.exe -X utf8 -u -m tacs.selector_audit --checkpoint "$auditRoot/kaggle_w0p1_seed44/policy_weight_two_stage_0p1_seed44/cifar100/dinov3_vits16/role_v2/seed_44/two_stage/checkpoint_best.pt" --standard-encoder "$auditRoot/kaggle_w0p03_seed44/policy_source_e427116799fe/reference/no_context/best_encoder.pt" --split-file "$auditRoot/kaggle_w0p03_seed44/policy_source_e427116799fe/data/splits/cifar100_seed42.json" --data-root data --output "$auditRoot/audit_w0p1_seed44_recheck" --device cuda --max-queries 0 --random-trials 3 --batch-size 4
```

Hai điều kiện biên: checkpoint/split sai phải bị từ chối; output cũ không
được ghi đè. Không lấy chênh lệch accuracy nhỏ làm bằng chứng cải tiến ổn
định nếu chưa chạy nhiều training seed.
