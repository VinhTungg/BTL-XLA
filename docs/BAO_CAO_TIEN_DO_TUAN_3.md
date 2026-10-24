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

### 3.4. Sửa đúng định nghĩa TACS Full và phép so sánh

Mode `tacs` trước đây chỉ lấy ngẫu nhiên `topk_candidates`, vì vậy chưa thực sự
đánh giá toàn bộ candidate pool. Pipeline hiện tại đã tách rõ:

- **TACS Full:** selector chấm toàn bộ 9.000 candidate cho mỗi query thông qua
  một embedding bank dùng chung, được làm mới mỗi train/evaluation epoch.
- **Two-Stage:** standard encoder quét 9.000 candidate bằng cosine, giữ top-k,
  sau đó TACS selector chỉ đánh giá k candidate này. Top-k được truyền
  bằng vị trí trong pool và gom từ một embedding bank chung được làm mới
  mỗi train/evaluation epoch; không còn mã hóa lại `B*k` ảnh mỗi batch.

Embedding bank của cả TACS Full và Two-Stage được detach để tránh giữ
graph của 9.000 ảnh và không bị nhân bản theo batch. Cơ chế này giúp
hai mode dùng cùng cơ chế gradient; selector encoder vẫn được cập nhật qua nhánh
query. Đây là lựa chọn triển khai có giới hạn bộ nhớ và được ghi rõ trong metadata.

Temperature được dùng nhất quán cho gradient của Gumbel-Softmax relaxation;
validation/test luôn chọn argmax xác định. Policy log-probability vẫn dùng
`softmax(scores)` đúng với phân phối hard Gumbel-Max. Model của TACS Full và
Two-Stage được khởi tạo lại từ cùng seed sau bước chuẩn bị dữ liệu/retrieval.
`metrics.json` bổ sung strategy, số candidate được quét/chấm điểm, thời gian
setup/retrieval, latency end-to-end mỗi query và throughput validation.

Trainer hiện dùng metrics schema v2 và đo peak RSS bằng monitor trong suốt từng
pha setup/train/validation/test. Khi chạy CUDA, trainer đồng thời ghi peak VRAM
allocated và reserved từ PyTorch. Script `scripts/analyze_all_models.py` chỉ tạo
bảng/biểu đồ khi đủ năm artifact có cùng split, seed, epoch, batch size,
embedding dimension và kết quả test trên toàn bộ CIFAR-10; dry-run hoặc metrics
legacy bị từ chối ở chế độ báo cáo chính thức.

---

## 4. Kiểm thử và smoke test đã thực hiện

### 4.1. Unit test

Đã chạy toàn bộ suite bằng PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Kết quả chạy gần nhất: suite mặc định có **38 test, 37 đạt và 1 integration
test được skip theo thiết kế**. Khi bật riêng integration CIFAR-10 thật, bài đó
đạt **1/1 (`OK`)**. Đây **chưa phải** kết quả huấn luyện đầy đủ hay so sánh
hiệu năng mô hình.

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

**Các con số trên là artifact smoke cũ, được tạo trước khi sửa TACS thành
full-pool; không được dùng để so sánh hay đưa vào kết luận thực nghiệm.** Cần
chạy lại hai mode bằng code hiện tại; `test_result` trong các file cũ cũng là
`null`.

### 4.3. Kiểm soát nguồn dữ liệu trước khi chạy thật

> [!IMPORTANT]
> **Cập nhật 2026-10-24:** trainer đã loại bỏ hoàn toàn đường chạy mock. Lỗi nạp CIFAR-10 sẽ dừng lần chạy trước khi tạo output; `--dry-run` vẫn dùng dữ liệu thật. `metrics.json` ghi nguồn dữ liệu, kích thước tập và hash file split.

Trước khi train chính thức, vẫn phải xác nhận `data.source=cifar10`, đúng split và đủ số lượng mẫu trong `metrics.json`. Các artifact baseline cũ được sinh từ mock không được dùng làm kết quả CIFAR-10.

Standard encoder cho Similarity/Two-Stage giờ bắt buộc có metadata của lần train `no_context` trên CIFAR-10. Cache embedding được ràng buộc với hash split, candidate indices, encoder và `embedding_dim`; cache 100 mẫu cũ hoặc cache legacy sẽ bị tái tạo thay vì được dùng lại.

### 4.4. Kiểm thử tích hợp Tuần 3 trên CIFAR-10 thật

Kiểm thử [tests/test_week3_real_cifar10.py](../tests/test_week3_real_cifar10.py) dùng CIFAR-10 đã có trên máy (không tải lại), kiểm tra số lượng ảnh trong các tập dữ liệu, rồi chạy TACS và Two-Stage với 2 query thật cho mỗi lượt. Kết quả thành viên xác nhận: **1/1 test đạt (`OK`)**. Nói dễ hiểu: chương trình đã đọc được ảnh thật và hai pipeline chạy thử được.

```powershell
$env:PYTHONPATH = "src"
$env:TACS_RUN_REAL_CIFAR_TESTS = "1"
python -m unittest discover -s tests -p "test_week3_real_cifar10.py" -v
```

Đây chỉ là **chạy thử để kiểm tra chương trình hoạt động với dữ liệu thật**; không phải huấn luyện đầy đủ, không đánh giá chất lượng mô hình trên toàn bộ tập test và không phải benchmark. Test cũng tính embedding cho candidate pool gồm 9.000 ảnh. Cần CIFAR-10 đã giải nén trong `data/cifar-10-batches-py/` và split `data/splits/cifar10_seed42.json`. Có thể đặt `TACS_DATA_ROOT` và `TACS_SPLIT_FILE` để trỏ tới vị trí khác.

### 4.5. Benchmark Two-Stage sau tối ưu shared embedding bank

Benchmark 1 epoch ngày 2026-10-24 dùng CPU, batch size 64, top-k 32 và toàn bộ
36.000/5.000/10.000 mẫu train/validation/test:

| Chỉ số | Trước tối ưu | Sau tối ưu | Thay đổi |
|---|---:|---:|---:|
| End-to-end | 379,13 s | 52,13 s | Nhanh hơn 7,27x |
| Train throughput | 120,90 q/s | 1.775,16 q/s | Tăng 14,68x |
| Train model time | 285,02 s | 18,35 s | Nhanh hơn 15,54x |
| Peak RAM | 1.025,49 MB | 889,63 MB | Giảm 13,2% |
| Test accuracy | 30,26% | 30,99% | Không suy giảm |

### 4.6. Hướng dẫn chạy

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
1. Đường chạy mock trong trainer đã bị loại bỏ; tiếp theo cần xác nhận CIFAR-10/split qua metadata và chạy huấn luyện, đánh giá thực tế cho `tacs` và `two_stage`.
2. Thu thập accuracy, latency và memory trên cùng cấu hình; không dùng metrics smoke test thay cho kết quả này.
3. Tổng hợp kết quả của đủ 5 mô hình: `no_context`, `random`, `similarity`, `tacs`, `two_stage`.
4. Người 3 cập nhật biểu đồ, báo cáo tổng kết, slide và kịch bản demo từ số liệu đã xác thực.
5. Nếu thí nghiệm phát hiện lỗi trong phần TACS/Two-Stage, Người 1 phối hợp sửa; Người 2 chạy lại test để xác nhận.

---

## 7. Kết luận

Phần triển khai TACS Full và pipeline Two-Stage đã được tách đúng về candidate
scope. Suite mặc định đạt **37/37 test được chạy** và skip 1 integration theo
thiết kế; integration CIFAR-10 thật đạt **1/1** khi bật riêng. Các kết quả này
xác nhận đường chạy, không phải bằng chứng về độ chính xác sau huấn luyện đầy đủ.

**Trạng thái bàn giao:** trainer đã fail-fast khi CIFAR-10 lỗi và ghi provenance dữ liệu. Tuy nhiên, chưa có kết quả train/test CIFAR-10 đầy đủ cho `tacs` và `two_stage`; chưa thể kết luận accuracy, latency hay mức sử dụng bộ nhớ của chúng trên dữ liệu thật.
