# TACS Improvement Project

Tái hiện và cải tiến **Task-Aligned Context Selection (TACS)** cho bài toán phân loại ảnh.

TACS học chọn một ảnh tham chiếu làm context cho ảnh truy vấn. Context không nhất thiết là ảnh giống nhất; đó là ảnh giúp mô hình downstream dự đoán chính xác hơn. Xem giải thích trong `docs/REPORT.md`.

## Mục tiêu

1. Dựng baseline không dùng context, random context và similarity context.
2. Tái hiện selector học bằng Gumbel-Softmax + policy gradient.
3. Đề xuất và kiểm chứng một cải tiến có ablation rõ ràng.
4. Duy trì báo cáo và tiến độ song song với code.

## Chạy kiểm tra nhanh

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m tacs.train_smoke --steps 5
Remove-Item Env:PYTHONPATH
```

Hiện tại skeleton chỉ cần PyTorch. Dataset và backbone thật sẽ được thêm sau khi chốt tài nguyên GPU.

## Tài liệu quản lý

- `docs/PROJECT_PLAN.md`: phạm vi, milestone và tiêu chí hoàn thành.
- `docs/REPORT.md`: bản thảo báo cáo cập nhật trong quá trình làm.
- `docs/PROGRESS.md`: trạng thái và việc tiếp theo.
- `docs/DECISIONS.md`: nhật ký quyết định kỹ thuật.
