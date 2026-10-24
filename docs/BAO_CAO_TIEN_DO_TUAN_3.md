# Báo cáo tiến độ Tuần 3 - TACS Full & Two-Stage Retrieval

## 1. Mục tiêu tuần 3

Tuần 3 tập trung vào việc hoàn thiện và chuẩn hóa triển khai Task-Aligned Context Selection (TACS) trong dự án CIFAR-10, đồng thời bổ sung mô hình cải tiến theo hướng Two-Stage Retrieval để tăng hiệu quả lựa chọn ngữ cảnh trong candidate pool lớn.

Các mục tiêu chính:
- Hoàn thiện mô hình TACS Full với selector và policy-gradient loss.
- Hỗ trợ pipeline huấn luyện cho các baseline và mô hình TACS mới.
- Triển khai hai giai đoạn lọc thô + rerank bằng cosine similarity + selector.
- Chuẩn hóa API và unit tests để đảm bảo pipeline chạy ổn định trên CPU.

---

## 2. Tình trạng thực hiện

### 2.1. Mô hình TACS Full

Dự án đã có sẵn các thành phần cốt lõi của TACS:
- `TACSModel` trong [src/tacs/model.py](../src/tacs/model.py)
- `TACSLoss` trong [src/tacs/losses.py](../src/tacs/losses.py)
- `TinyVisionEncoder` và `TaskAlignedSelector`

Mô hình này hoạt động theo nguyên tắc:
- Query embedding được trích xuất từ encoder.
- Selector đánh giá từng candidate trong pool và tính trọng số lựa chọn bằng Gumbel-Softmax.
- Context vector được tạo từ candidate được chọn.
- Classifier kết hợp query + context để phân loại.
- Hàm mất mát gồm task loss và policy loss dựa trên reward chênh lệch giữa no-context và context.

### 2.2. Two-Stage Retrieval

Phần cải tiến dựa trên ý tưởng:
1. Giai đoạn 1: lọc thô bằng cosine similarity để lấy top-k ứng viên gần nhất.
2. Giai đoạn 2: dùng selector TACS để rerank các ứng viên này, chọn ra candidate phù hợp nhất cho nhiệm vụ phân loại.

Các thành phần mới bao gồm:
- `retrieve_topk()` trong [src/tacs/retrieval.py](../src/tacs/retrieval.py)
- `TopKContextDataset` trong [src/tacs/retrieval.py](../src/tacs/retrieval.py)

Điểm mạnh của cấu trúc này là giảm chi phí tính toán khi candidate pool lớn, đồng thời vẫn giữ được khả năng chọn context phù hợp.

---

## 3. Cập nhật kĩ thuật đã thực hiện

### 3.1. Cập nhật trainer

File [src/tacs/train.py](../src/tacs/train.py) đã được mở rộng để hỗ trợ các mode sau:
- `no_context`
- `random`
- `similarity`
- `tacs`
- `two_stage`

Trainer mới có khả năng:
- dựng DataLoader tương ứng với từng baseline/mô hình,
- load checkpoint và pretrained encoder,
- chạy evaluation và lưu metrics JSON,
- lưu file checkpoint tốt nhất (`checkpoint_best.pt`) và encoder tốt nhất (`best_encoder.pt`).

### 3.2. Chuẩn hóa loss và accuracy cho TACS

Một điểm quan trọng khi tích hợp TACS vào trainer là xử lý sự khác biệt giữa:
- mô hình baseline: logits đơn giản, loss = `CrossEntropyLoss`
- mô hình TACS: loss là `TACSLossOutput` với thuộc tính `.total`

Vì vậy, trainer đã được cập nhật để:
- nếu mô hình trả về `TACSOutput`, lấy `output.context_logits` để tính accuracy,
- nếu loss là `TACSLossOutput`, lấy `loss.total` để backprop và tính giá trị mất mát.

### 3.3. Xuất API chuẩn hóa

File [src/tacs/__init__.py](../src/tacs/__init__.py) đã được bổ sung export cho:
- `TopKContextDataset`
- các thành phần chính của TACS và retrieval

Điều này giúp dễ dàng import trong test và các script cần dùng lại model.

---

## 4. Kiểm thử và smoke test đã thực hiện

### 4.1. Unit test

Đã chạy toàn bộ suite bằng PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Kết quả chạy gần nhất do thành viên xác nhận: **26/26 test đạt (`OK`)**, trong đó có 1 integration test CIFAR-10 thật được bật riêng. Nói đơn giản, các bài kiểm tra cho thấy những phần chính của chương trình chạy đúng; riêng TACS và Two-Stage cũng đã chạy thử được với một lượng nhỏ ảnh thật. Đây **chưa phải** kết quả huấn luyện đầy đủ hay so sánh hiệu năng mô hình.

Các test kiểm tra những phần như:
- top-k retrieval: xếp hạng, giới hạn theo số candidate và từ chối `top_k` không hợp lệ;
- `TopKContextDataset`: shape batch tương thích với TACS và điểm cosine được xếp giảm dần;
- vòng lặp train/evaluate xử lý `TACSOutput` và `TACSLoss`;
- CLI nhận `--topk-candidates` và từ chối giá trị không dương;
- smoke run của cả `tacs` và `two_stage`, xác nhận sinh metrics và checkpoint.

### 4.2. Smoke test CLI

Hai lệnh sau đã chạy hoàn tất trên CPU:

```powershell
$env:PYTHONPATH = "src"
python -m tacs.train --baseline tacs --dry-run --device cpu --output-dir outputs/smoke/tacs
python -m tacs.train --baseline two_stage --dry-run --device cpu --topk-candidates 32 --output-dir outputs/smoke/two_stage
```

Các file kết quả nằm trong `outputs/smoke/tacs/` và `outputs/smoke/two_stage/`. Metrics của lần chạy đã ghi nhận:

| Mode | Best validation accuracy | Train throughput | RSS RAM | Test |
|---|---:|---:|---:|---|
| TACS | 100% | 124.38 query/s | 614.97 MB | Chưa đánh giá |
| Two-Stage | 87.5% | 141.87 query/s | 659.68 MB | Chưa đánh giá |

**Các con số trên chỉ là kết quả smoke test với một epoch và tập mẫu nhỏ; không phải kết quả CIFAR-10 để so sánh hay đưa vào kết luận thực nghiệm.** `test_result` trong cả hai file metrics là `null`.

### 4.3. Lưu ý bắt buộc trước khi chạy dữ liệu thật

> [!WARNING]
> **Chưa chạy huấn luyện chính thức cho đến khi xử lý hoặc xác minh được nhánh fallback dữ liệu.** Trong [build_dataloaders](../src/tacs/train.py), nếu `build_cifar10(...)` phát sinh bất kỳ exception nào, trainer hiện bắt lỗi rồi âm thầm gọi `create_mock_datasets(...)`. Vì vậy, bỏ `--dry-run` không đảm bảo đang dùng CIFAR-10: lệnh vẫn có thể chạy xong và ghi `metrics.json`/checkpoint như bình thường, nhưng trên dữ liệu giả. Không dùng những artifact đó làm kết quả thực nghiệm CIFAR-10.

Trước khi train chính thức, Người 1 cần đổi fallback này thành lỗi rõ ràng cho lượt chạy thật (hoặc cung cấp chế độ mock tường minh, mặc định tắt); Người 2 cần xác nhận log/config chỉ ra dữ liệu CIFAR-10 thật, đúng split và số lượng mẫu. Nếu lần chạy báo lỗi khi nạp dữ liệu, phải dừng và sửa nguyên nhân, không báo cáo metric vừa sinh ra là kết quả CIFAR-10.

### 4.4. Kiểm thử tích hợp Tuần 3 trên CIFAR-10 thật

Kiểm thử [tests/test_week3_real_cifar10.py](../tests/test_week3_real_cifar10.py) dùng CIFAR-10 đã có trên máy (không tải lại), kiểm tra số lượng ảnh trong các tập dữ liệu, rồi chạy TACS và Two-Stage với 2 query thật cho mỗi lượt. Kết quả thành viên xác nhận: **1/1 test đạt (`OK`)**. Nói dễ hiểu: chương trình đã đọc được ảnh thật và hai pipeline chạy thử được.

```powershell
$env:PYTHONPATH = "src"
$env:TACS_RUN_REAL_CIFAR_TESTS = "1"
python -m unittest discover -s tests -p "test_week3_real_cifar10.py" -v
```

Đây chỉ là **chạy thử để kiểm tra chương trình hoạt động với dữ liệu thật**; không phải huấn luyện đầy đủ, không đánh giá chất lượng mô hình trên toàn bộ tập test và không phải benchmark. Test cũng tính embedding cho candidate pool gồm 9.000 ảnh. Cần CIFAR-10 đã giải nén trong `data/cifar-10-batches-py/` và split `data/splits/cifar10_seed42.json`. Có thể đặt `TACS_DATA_ROOT` và `TACS_SPLIT_FILE` để trỏ tới vị trí khác.

### 4.5. Hướng dẫn chạy

Đặt source path và chuẩn bị CIFAR-10:

```powershell
$env:PYTHONPATH = "src"
python -c "from tacs.data import build_cifar10; build_cifar10('data', 'data/splits/cifar10_seed42.json', download=True)"
```

Smoke test ngắn, không dùng làm kết quả thực nghiệm:

```powershell
python -m tacs.train --baseline tacs --dry-run --device cpu --output-dir outputs/smoke/tacs
python -m tacs.train --baseline two_stage --dry-run --device cpu --topk-candidates 32 --output-dir outputs/smoke/two_stage
```

Sau khi xác nhận dữ liệu thật, train `no_context` trước để tạo encoder chuẩn, rồi chạy TACS và Two-Stage:

```powershell
python -m tacs.train --baseline no_context --epochs 5 --batch-size 64 --device auto
python -m tacs.train --baseline tacs --epochs 5 --batch-size 64 --device auto --eval-test
python -m tacs.train --baseline two_stage --epochs 5 --batch-size 64 --device auto --topk-candidates 32 --eval-test
```

Kết quả được lưu lần lượt ở `outputs/no_context/`, `outputs/tacs/` và `outputs/two_stage/`, gồm metrics và checkpoint. Dùng `--device cpu` nếu không có GPU; thêm `--output-dir <đường-dẫn>` để đổi thư mục kết quả.

---

## 5. Phân tích kỹ thuật

### 5.1. Ưu điểm của TACS Full
- Chọn context có tính “task-aligned”, không chỉ dựa trên độ tương đồng hình ảnh.
- Cấu trúc selector + classifier cho phép học biểu diễn tốt hơn cho task cụ thể.
- Policy gradient giúp tối ưu selector theo reward của downstream classification.

### 5.2. Ưu điểm của Two-Stage
- Tiết kiệm chi phí khi candidate pool lớn.
- Dùng cosine similarity để lọc nhanh top-k trước khi đưa vào selector.
- Giảm thiểu khối lượng tính toán so với đánh giá toàn bộ candidate pool ở mỗi bước.

### 5.3. Hạn chế cần lưu ý
- TACS Full có chi phí tính toán cao hơn khi candidate pool lớn.
- Two-Stage cần có bộ encoder ổn định để kiếm top-k hiệu quả.
- Cần thêm thí nghiệm thực tế trên CIFAR-10 đầy đủ để so sánh chính xác accuracy, latency và bộ nhớ.

---

## 6. Kế hoạch tiếp theo

Các việc **chưa hoàn tất**:
1. Người 1 xử lý fallback mock trong trainer để lần chạy chính thức không thể âm thầm dùng dữ liệu giả; sau đó Người 2 xác nhận CIFAR-10/split và chạy huấn luyện, đánh giá thực tế cho `tacs` và `two_stage`.
2. Thu thập accuracy, latency và memory trên cùng cấu hình; không dùng metrics smoke test thay cho kết quả này.
3. Tổng hợp kết quả của đủ 5 mô hình: `no_context`, `random`, `similarity`, `tacs`, `two_stage`.
4. Người 3 cập nhật biểu đồ, báo cáo tổng kết, slide và kịch bản demo từ số liệu đã xác thực.
5. Nếu thí nghiệm phát hiện lỗi trong phần TACS/Two-Stage, Người 1 phối hợp sửa; Người 2 chạy lại test để xác nhận.

---

## 7. Kết luận

Phần triển khai TACS Full và pipeline Two-Stage đã được tích hợp. Theo kết quả chạy gần nhất, **26/26 test đạt**; trong đó integration test CIFAR-10 thật xác nhận chương trình đọc được dữ liệu và chạy thử train/evaluate cơ bản cho TACS/Two-Stage. Hiểu đơn giản: các phần code được kiểm tra đều chạy qua, nhưng đây chưa phải kết quả huấn luyện đầy đủ hay bằng chứng mô hình đạt độ chính xác tốt.

**Trạng thái bàn giao:** code đã sẵn sàng cho bước kiểm tra tiếp theo, nhưng trainer hiện chưa an toàn cho huấn luyện chính thức do có thể âm thầm fallback sang mock dataset. Cần xử lý/kiểm soát hành vi này trước khi chạy thực nghiệm. Tại thời điểm báo cáo, chưa có kết quả train/test CIFAR-10 đầy đủ cho `tacs` và `two_stage`; chưa thể kết luận accuracy, latency hay mức sử dụng bộ nhớ của chúng trên dữ liệu thật.
