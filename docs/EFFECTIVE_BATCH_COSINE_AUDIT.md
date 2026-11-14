# Chẩn đoán cosine trên hai checkpoint advantage seed44

Ngày nghiệm thu: 14/11/2026. Người dùng chạy hai audit; agent đọc và kiểm tra
artifact, không train hoặc chạy thêm GPU trong nhiệm vụ này. Đây là can thiệp
inference trên task weights cố định của từng checkpoint, không phải mô hình
similarity/cosine train độc lập hay bằng chứng vượt bài báo.

## Nghiệm thu

Hai thư mục tại `artifacts/effective_batch_pilot/`:

- `audit_cosine_microbatch_seed44/`.
- `audit_cosine_effective_batch_seed44/`.

Cả hai có status complete, full validation5.000query, pool9.000, top-k32,
audit seed44, 3random trials, batch4, CUDA AMP, torch2.14.1+cu126,
cosine_diagnostics=true, weights_updated=false, test_evaluated=false.
Thời gian audit lần lượt459,42s và454,69s; không phải latency demo hoặc train.

Đã xác minh đúng thứ tự/ID5.000query duy nhất trong split; nhãn query và mọi
context strategy khớp CIFAR-100 training partition thật. Không mở test partition.
Hash best checkpoint/metrics/split và config khớp training manifest. Manifest
audit khớp report. Tính lại toàn bộ results từ JSONL bằng summarize khớp report;
correct flags khớp prediction/target và CE hữu hạn. Pool indices/norms đủ9.000,
hữu hạn, dương; tính lại quantiles và mean norm context được chọn khớp report.

Các trường/prediction có sẵn ở audit không-cosine trước **khớp hoàn toàn** từng
query. Hai lượt mới có cùng query order, random pool positions và retrieval
top1 pool position; task weights hai checkpoint vẫn khác nhau.
Selected accuracy lệch checkpoint−0,06 và+0,02điểm %, trong tolerance0,1điểm %;
không gọi là tái đo khớp tuyệt đối. Môi trường audit khác môi trường train.

## Kết quả

| Chọn context khi inference | microbatch task weights | effective_batch task weights |
|---|---:|---:|
| Selector gốc, dot-product AMP | 87,86% | 87,82% |
| Selector dot FP32 trên features hiện có | 87,90% | 87,80% |
| Selector cosine trên features hiện có | 87,56% | 87,64% |
| Frozen retrieval cosine top-1 | 87,94% | 87,76% |
| Random trung bình3trials trong top-32 | 87,693333% | 87,72% |
| Không context, cùng task weights | 87,48% | 87,64% |

Selector cosine giảm0,30 và0,18điểm % so selector gốc: sửa đúng25/làm sai40
ở microbatch, sửa đúng23/làm sai32 ở effective-batch. Mean CE cũng xấu đi:
microbatch0,409153→0,411825; effective-batch0,411441→0,414486.
Đổi cosine riêng lúc inference chưa có ích; chưa kiểm tra cosine train từ đầu.

Frozen top-1 hơn selector gốc0,08điểm % ở microbatch (sửa đúng36/làm sai32),
nhưng thấp hơn0,06điểm % ở effective-batch (sửa đúng30/làm sai33).
Đây là−/+vài ảnh trên5.000query; chưa có kiểm định ý nghĩa hay nhiều seed.
Không tuyên bố frozen top-1 thắng ổn định hoặc learned reranking tốt hơn.
Cả hai dùng cùng reference retrieval encoder, nhưng task networks khác nhau,
nên không so hai cột như thể chỉ đổi một context selector trên cùng model.

Dot FP32 đổi context của583query(11,66%) và268query(5,36%), nhưng accuracy
chỉ tăng0,04điểm % / giảm0,02điểm %. Đây là rescoring float32 của features đã
tạo, không phải tái chạy toàn model FP32; không quy mọi khác biệt cho AMP
hoặc kết luận đổi precision sẽ tăng accuracy.

## Độ dài vector và sự đa dạng context

Norm là độ dài vector embedding; dot-product phụ thuộc cả độ dài và hướng,
cosine chuẩn hóa độ dài để so hướng. Đây là giải thích cơ chế, không đủ để
quy nguyên nhân accuracy của toàn hệ thống cho norm.

| Thống kê lựa chọn | microbatch gốc / cosine | effective_batch gốc / cosine |
|---|---:|---:|
| Context khác nhau | 1.108 / 3.453 | 1.283 / 3.372 |
| Tỷ lệ query dùng context phổ biến nhất | 1,14% / 0,14% | 0,92% / 0,18% |
| Percentile norm trung bình trong top-k hợp lệ | 99,509375% / 51,709375% | 98,945625% / 51,335625% |
| Mean selected norm | 18,407706 / 18,273225 | 14,501956 / 14,249627 |

Percentile trong bảng là tỷ lệ ứng viên hợp lệ có norm <= norm context được
chọn, bao gồm ties; không phải tỷ lệ query chọn đúng ảnh norm lớn nhất.
Selector gốc thiên mạnh về context norm cao trong top-k, nhưng cosine loại
ảnh hưởng này và tăng đa dạng **vẫn làm giảm accuracy**. Không coi norm bias
hay đa dạng thấp tự thân là bằng chứng lỗi cần sửa để chắc chắn tăng accuracy.
Report ghi tương quan norm với tần suất chọn sau điều chỉnh exposure là
0,4190/0,3829; đây là mô tả, không chứng minh quan hệ nhân quả.

Frozen retrieval top-1 chọn context cùng nhãn query65,10%, trong khi selector
gốc45,50%/46,80%. Chọn cùng nhãn nhiều hơn vẫn không đảm bảo task accuracy
cao hơn (effective-batch là phản ví dụ trong phép đo này).

Giới hạn nghiệm thu: artifact không lưu toàn bộ ma trận top-k5.000×32. Vì vậy
percentile/exposure-adjusted correlation được trích từ report, đã kiểm tra
công thức source nhưng **không tính lại độc lập các trường này từ artifact**.
Norm quantiles/mean-selected và toàn bộ accuracy/CE/selection counts đã được
tính lại từ norms/JSONL. Không tái inference chỉ để dựng lại top-k trong bước này.

## Quyết định tiếp theo

1. Giữ nguyên selector dot-product của bản tái hiện và checkpoint gốc.
   Không đổi mặc định sang cosine hoặc FP32 chỉ để chọn số validation đẹp hơn.
2. Khép lại nhánh effective-batch seed44 như ablation chưa chứng minh tăng
   accuracy; không tự mở thêm lượt train hoặc dùng test để chọn biến thể.
3. Chuyển sang hoàn thiện demo/bảng thực nghiệm, hiển thị rõ selected,
   retrieval top-1, random và no-context, gồm cả ví dụ context giúp và gây hại.
   Source đã có `src/tacs/demo.py`; chưa nghiệm thu demo với cặp checkpoint này.
4. Nếu vẫn cần phát triển thuật toán mới: phải khóa giả thuyết, metric và
   đối chứng trước pilot; kết quả hiện tại không đủ để chọn cosine làm hướng
   cải tiến thành công. Đây là quyết định nghiên cứu tiếp, không nằm trong
   nhiệm vụ đọc output. Chỉ mở test cuối sau khi khóa phương án.

## Hash bằng chứng

| Artifact | SHA256 |
|---|---|
| microbatch report.json | `92aa6d174665bd92dd9b7a99dca4f60acc27e7d7c9f0baba22e0c96019d139f0` |
| microbatch queries.jsonl | `34e9c49ab385067e2a77d8f55b884e38c2b6add9cc0abf49d23ee9816e644aa5` |
| microbatch candidate_norms.json | `78ca2b9e82bdfd2fd32c8582a58277fb3785f4b9196cef219a50375bf7f7540c` |
| effective-batch report.json | `25b72424fa7f0fc9f2cf4a41a14f951ab3ecfc42c2d880a41f724669cac6f26c` |
| effective-batch queries.jsonl | `3be73e432e6c7e1f5f831bee1cb166ab9675f0fb9b25adc4e110aa5ca88b69bd` |
| effective-batch candidate_norms.json | `31bf36a3ceecc4a57153664fb3371c22c2fa11f32f11a8dfd0eb567d48731e5f` |

Định danh checkpoint và training:
[EFFECTIVE_BATCH_SEED44_RESULTS.md](EFFECTIVE_BATCH_SEED44_RESULTS.md).
Audit cơ bản trước đó:
[EFFECTIVE_BATCH_SELECTOR_AUDIT.md](EFFECTIVE_BATCH_SELECTOR_AUDIT.md).

Kiểm tra phần mềm: `python -m unittest discover -s tests -p test_selector_audit.py -v`.
Đã chạy lại20/20tests đạt ngày14/11; `git diff --check` không lỗi.
Trường hợp biên: zero/nonfinite norms phải bị từ chối; strategy chẩn đoán thêm
không được lẫn vào random mean và làm thay đổi đối chứng3trials.
