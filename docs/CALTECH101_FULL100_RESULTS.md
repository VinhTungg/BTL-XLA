# Caltech-101 full100 seed44 — kết quả đã thu hồi

Ngày nghiệm thu: 16/11/2026. Đây là protocol demo của nhóm, **không phải điểm
test/paper reproduction đầy đủ**. Không có lượt train bổ sung trong nhiệm vụ
thu hồi và tích hợp demo này.

## Kết quả train cloud đã đối chiếu checkpoint

| Mô hình train độc lập | Đã train | Epoch checkpoint best | Best validation Acc | Validation epoch100 |
|---|---:|---:|---:|---:|
| No-context | 100 epoch | 21 | 95,64% | 94,26% |
| Two-Stage role_v2 | 100 epoch | 54 | 94,26% | 92,87% |

Two-Stage thấp hơn No-context **1,38 điểm phần trăm** ở best validation của lượt
seed44 này. Không được viết là cải tiến đã tăng accuracy. Lịch100epoch đã chạy
xong; checkpoint dùng trong demo là **best validation**, không phải tự động
epoch100. Vì validation chỉ505ảnh và dùng chọn checkpoint, kết quả này chưa
đủ cho kết luận tổng quát/ý nghĩa thống kê.

Validation cân bằng5ảnh/lớp nên Acc và mean-per-class accuracy bằng nhau trên
**validation này**. Số báo cáo train được làm tròn4chữ số thập phân. Chưa đánh
giá test; không gọi95,64% hoặc94,26% là test accuracy.

## Dữ liệu và recipe

-8.677ảnh thật/101lớp, loại BACKGROUND_Google, grayscale chuyển RGB.
- Protocol `caltech101_demo_19query_6pool_5val_rest_test_v1`, split seed42.
  Query1919, pool606, validation505, test5647; không trùng index/hash ảnh
  giữa các vai trò. Đã so kiểm kê ảnh local với inventory cloud.
- Seed train44, DINOv3 ViT-S/16 pretrained, role_v2,101logits,256×256.
- AdamW LR1e-4, WD1e-4, warmup5, cosine100; batch4, accumulation16, AMP.
  Two-Stage: frozen-DINO cosine top32, learned selector, temperature0.1,
  policy weight1.0, advantage microbatch. Không quét thêm hyperparameter.
- Cloud: Tesla T4, torch2.14.1+cu130, torchvision0.29.1+cu130, timm1.0.30,
  numpy2.4.6. Reference là frozen pretrained DINOv3, không phải task encoder
  đã fine-tune. Hai run có cùng source/split/inventory/environment.
- Tổng `history.seconds`: No-context4520,14s (~75,34phút), Two-Stage11229,68s
  (~187,16phút). Đây là tổng thời gian epoch ghi trong log, không gồm toàn bộ
  download/khởi tạo/retrieval setup/chờ quota; không dùng làm benchmark latencies.

## Gói đã thu hồi và kiểm tra

ZIP gốc `artifacts/seed44_full100_v1-20261116T133815Z-1-001.zip` được giữ nguyên.
Giải nén14file vào `artifacts/caltech_demo/seed44_full100_v1/`, tổng1.299.485.122bytes;
không ghi đè file khác, không xóa ZIP hoặc checkpoint. Audit chi tiết:
`artifacts/caltech_demo/audit_full100.json`.

| Định danh | SHA256 |
|---|---|
| ZIP gốc | `e28e3ca408de72df6bd2b26f001636a719f0d48b62070d4c531709f8e4ffa0be` |
| Source train (đối chiếu bundle nhúng notebook gốc) | `ad14eece024694224d7ae1eaa2544e532cd91038261ee2931eff4300406890a9` |
| Split cloud | `4110feb7775391e481fd6b0a2bcd5b841a6479e07e44a7293ed070407ea6a91c` |
| Inventory | `ce486e3a971bbb9f7cb5ae66d47893dcb4b56757292efdf9fda6125d78d554aa` |
| No-context best | `f2fa1b9f26c162eccf962e02679c08f76545349d69d99e8448b96c00faa9073d` |
| Two-Stage best | `27091fd7ca2405ece49d2fd4ce8b2d08f98c6e1fedad52876cc06ab9b25e5dc5` |
| Frozen reference | `44f6e65eda5c73620ef279f437851c7110dfba1b514d80d424245e8292e1c05f` |

Đã nạp best/latest bằng `weights_only=True`; latest có epoch100, đủ100history,
config đúng và state model/optimizer/scheduler/scaler/RNG. Hash metrics khớp
`result.json`; epoch và best validation trong best checkpoint khớp history.
Source được so với **notebook đã dùng train**, không lấy source UI mới thêm
sau train để giả thành source thực nghiệm cũ.

## Kiểm tra lại

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/recover_caltech_result.py --archive artifacts/seed44_full100_v1-20261116T133815Z-1-001.zip --report artifacts/caltech_demo/audit_full100_recheck.json
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_caltech_recovery.py -v
```

Recovery cho phép kiểm tra lại file trùng hash nhưng từ chối ghi đè file khác.
Luôn dùng tên report mới. Biên bắt buộc: ZIP traversal/entry lạ bị từ chối trước
giải nén; checkpoint/config/split/nhãn/test không khớp phải dừng, không fallback.

## Cách trình bày trung thực

“Nhóm mở rộng demo sang Caltech-101 để quan sát ảnh nguồn lớn hơn CIFAR-100.
Hai mô hình đã train100epoch từ pretrained, seed44, trên protocol demo riêng.
Best validation No-context95,64%, Two-Stage94,26%; chưa chứng minh cải tiến
tăng độ chính xác. Demo cho xem context giúp/hại ở từng ảnh; ví dụ cá biệt không
thay cho kết luận toàn tập. Chưa dùng test để chọn model.”
