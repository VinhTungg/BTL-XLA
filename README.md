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
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python -m tacs.train_smoke --steps 5
Remove-Item Env:PYTHONPATH
```

## Kiểm tra môi trường

Chạy lệnh sau sau khi cài project để kiểm tra Python, PyTorch, CUDA và GPU:

```powershell
python -c "import sys, torch; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('PyTorch CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(f'GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Kết quả nghiệm thu trên máy số 2 (2026-09-15): Python 3.13.7, PyTorch 2.10.0+cpu, CUDA build `None`, CUDA không khả dụng và không phát hiện GPU. Unit test đạt 3/3; smoke training 5 bước chạy thành công trên CPU. `torchvision` hiện chưa được cài và cần bổ sung trước khi triển khai CIFAR-10.

Hiện tại skeleton chỉ cần PyTorch. Dataset và backbone thật sẽ được thêm sau khi chốt tài nguyên GPU.

## Tài liệu quản lý

- `docs/PROJECT_PLAN.md`: phạm vi, milestone và tiêu chí hoàn thành.
- `docs/REPORT.md`: bản thảo báo cáo cập nhật trong quá trình làm.
- `docs/PROGRESS.md`: trạng thái và việc tiếp theo.
- `docs/DECISIONS.md`: nhật ký quyết định kỹ thuật.
