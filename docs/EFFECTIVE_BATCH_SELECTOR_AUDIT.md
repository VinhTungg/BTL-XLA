# Audit selector: microbatch và effective-batch, seed44

Ngày nghiệm thu: 14/11/2026. Người dùng đã chạy audit; bước này chỉ đọc và
kiểm tra output, không chạy lại GPU, không train thêm và không đánh giá test.
Không gộp kết quả này với pilot policy-weight hoặc role_v2 cũ.

## Tính hợp lệ và provenance

Output tại `artifacts/effective_batch_pilot/`:

- `audit_full_microbatch_seed44/{report.json,manifest.json,queries.jsonl}`.
- `audit_full_effective_batch_seed44/{report.json,manifest.json,queries.jsonl}`.

Cả hai report có `status=complete`, full validation5.000 query, pool9.000,
32 ứng viên/query, seed audit44, 3 random trials, batch4, CUDA AMP,
torch2.14.1+cu126, `weights_updated=false`, `test_evaluated=false`.
Đây là môi trường audit local khác torch2.11.0+cu130/T4 của training;
không diễn giải sai lệch tái đo nhỏ như tác động của một lần train mới.

Đã kiểm tra trực tiếp:

1. Mỗi JSONL có đúng5.000 query ID duy nhất, đúng thứ tự validation trong split;
   query ID và nhãn khớp giữa hai audit, không lẫn train/pool/test queries.
2. Nhãn query và context được chọn khớp CIFAR-100 **training partition thật**
   (validation là subset của partition này); không mở CIFAR test partition.
3. Selected CIFAR ID khớp pool position trong split. Random pool positions hợp lệ;
   cả3 random positions của từng query khớp giữa hai audit, đủ5.000/5.000 query.
4. Tính lại toàn bộ `results` từ JSONL bằng `selector_audit.summarize`: khớp
   report sau chuẩn hóa tuple/list qua JSON. Kiểm tra correct flags khớp
   prediction/target và mọi cross-entropy hữu hạn.
5. Manifest là phần thông tin đầu report tương ứng; training config khớp
   manifest thí nghiệm. Hash best checkpoint và metrics tính lại khớp audit.
6. Split local và encoder tham chiếu được hash lại, khớp report/training manifest.

| File | SHA256 |
|---|---|
| microbatch report | `df128b8f7073d2a30d416b172c0af82599c3faf35a9634724a1b5f0fbcb9e113` |
| microbatch queries | `d252bdd9d8fc60f5423d089e87952c8a24d491786381a0548a7cdae45be1ac99` |
| effective-batch report | `630205360a791371721c9934c06e4b08bbcc85b2526b327bee8819ef5cd189b3` |
| effective-batch queries | `8d4e5e96dc7171669301c475e3c1c804803a44f541872d328827a54ef95abf02` |

Checkpoint best: microbatch epoch20,
`fbc50a26867d321a32892f262a9d447f45e3be95dd0a3a345053cf04ee8425d8`;
effective-batch epoch17,
`5e7406aa36bfd16e20628f27c61f3238db3926b1afe12cc552ddb2618b845fa4`.

## Accuracy và tái đo

| Chỉ số trên full validation | microbatch | effective_batch |
|---|---:|---:|
| Best accuracy ghi lúc train | 87,92% | 87,80% |
| Selector, đo lại trong audit | **87,86%** | **87,82%** |
| Random trial0 | 87,76% | 87,78% |
| Random trial1 | 87,46% | 87,60% |
| Random trial2 | 87,86% | 87,78% |
| Random trung bình3trials | 87,693333% | 87,720000% |
| Bỏ context, cùng task weights | 87,48% | 87,64% |
| Selector trừ random trung bình | +0,166667 điểm % | +0,100000 điểm % |
| Selector trừ bỏ context | +0,38 điểm % | +0,18 điểm % |
| Audit accuracy trừ checkpoint | −0,06 điểm % | +0,02 điểm % |
| Trong tolerance0,1điểm % | Có | Có |
| Thời gian audit | 322,76s | 322,09s |

Hai accuracy tái đo đều trong tolerance đã đặt, **không khớp tuyệt đối** số
checkpoint. Không sửa số train đã lưu thành số audit. Không biết chính xác
thành phần số học/môi trường nào tạo sai lệch nếu chưa có thử nghiệm đối chứng.

So hai model trên cùng query: effective-batch sửa đúng108 ảnh microbatch sai,
nhưng làm sai110 ảnh microbatch đúng; ròng−2ảnh, tương ứng−0,04điểm % audit.
Hai checkpoint/task weights khác nhau; đây không phải can thiệp chỉ đổi
selector trên một task network cố định.

Trong từng model, bỏ context làm đổi dự đoán111ảnh ở microbatch và121ảnh
ở effective-batch. Microbatch: context sửa đúng50, làm sai31, ròng+19ảnh.
Effective-batch: sửa đúng51, làm sai42, ròng+9ảnh. Số đổi dự đoán lớn hơn tổng
sửa đúng/làm sai vì có trường hợp sai sang một nhãn sai khác.

## Cross-entropy và phân bố chọn context

Cross-entropy (CE) đo mức phạt cho xác suất dự đoán; thấp hơn tốt hơn. CE
có thể giảm dù nhãn dự đoán top-1 không đổi, nên không thay thế accuracy.

| Chỉ số | microbatch | effective_batch |
|---|---:|---:|
| CE selector | 0,409153 | 0,411441 |
| CE bỏ context | 0,419278 | 0,421010 |
| Giảm CE so random trung bình | 0,001130 | 0,002492 |
| Query có CE tốt hơn random trung bình | 2.518/5.000 | 2.686/5.000 |
| Số context khác nhau đã chọn | 1.108 | 1.283 |
| Context phổ biến nhất | 1,14% query | 0,92% query |
| 10context phổ biến nhất | 7,68% query | 6,70% query |
| Context cùng nhãn query | 45,50% | 46,80% |
| Mean policy entropy | 2,862595 | 2,591596 |
| Mean top-1 probability | 0,200742 | 0,240365 |

Effective-batch có phân bố sử dụng context trên toàn validation đa dạng hơn,
nhưng policy trên từng query lại có entropy thấp hơn. Hai thống kê này không
mâu thuẫn: một policy có thể tự tin trên từng ảnh nhưng chọn ảnh context khác
nhau cho các query khác nhau. Không gọi sự đa dạng này là bằng chứng selector
tốt hơn, hết collapse, hoặc tác động nhân quả riêng của normalization.
Chỉ63/5.000query chọn cùng context giữa hai model; dù lựa chọn thay đổi nhiều,
accuracy tổng vẫn rất gần nhau. Cùng nhãn context là chẩn đoán, không phải mục
tiêu cần tối đa hóa hay chứng minh task utility.

## Kết luận và quyết định

Quan sát ở seed44: selector có lợi ích accuracy nhỏ so random/top-k và bỏ
context trên cả hai checkpoint. Tuy nhiên effective-batch không vượt microbatch
về accuracy lúc train lẫn tái đo audit; mức tăng so random cũng không lớn hơn.
Giảm CE so random của nhánh mới lớn hơn là tín hiệu mô tả, chưa đủ để chọn
phương án thắng. Không thực hiện kiểm định ý nghĩa hoặc xác nhận nhiều seed.
3random trials không phải3seed train và không tạo15.000query độc lập.

Giữ microbatch làm đối chứng hiện tại; lưu effective-batch như một **ablation
chưa cho thấy tăng accuracy**, không quảng bá là cải tiến đã thành công.
Không tăng epoch, quét weight tiếp hoặc mở test để tìm kết quả đẹp hơn.

Bước nghiên cứu ít tốn tài nguyên tiếp theo, nếu tiếp tục nhánh selector:
chạy `--cosine-diagnostics` trên **hai checkpoint mới này**, output riêng,
full validation và cùng môi trường. So thêm frozen retrieval top-1 với selector
dot/cosine để kiểm tra phần rerank có hơn cách lấy ảnh gần nhất đơn giản không.
Pilot role_v2 cũ đã có chẩn đoán cosine, nhưng không dùng kết quả checkpoint cũ
để thay cho phép kiểm tra checkpoint hiện tại. Đây vẫn chỉ là can thiệp inference,
không thay thuật toán tái hiện hoặc chứng minh một selector cosine train riêng.
Không tự khởi chạy chẩn đoán bổ sung trong bước đọc output này.

Đối với demo, hai checkpoint hiện tại đã có output hợp lệ để xây dựng phần
so sánh query/context/prediction; không cần đợi một lời hứa accuracy cao hơn.
Nhãn đóng góp, ablation không thành công và hạn chế một seed phải được trình bày
rõ; mô hình demo chưa được đánh giá test cuối trong protocol này.

## Kiểm tra lại

Dùng cùng CLI `python -m tacs.selector_audit` và cấu hình ở
[EFFECTIVE_BATCH_SEED44_RESULTS.md](EFFECTIVE_BATCH_SEED44_RESULTS.md), bỏ
`--check-only`, đặt `--device cuda --max-queries 0 --random-trials 3 --batch-size 4
--seed 44`. Chọn output mới, không ghi đè audit đã nghiệm thu.

Trường hợp biên cần giữ guard: thiếu query/trùng query phải bị coi là audit
không đầy đủ; checkpoint/metrics/split sai hash không được dùng để ghép cặp.

Kiểm thử liên quan chạy lại ngày14/11: `python -m unittest discover -s tests
-p test_selector_audit.py -v`, 20/20tests đạt. `git diff --check` không lỗi.
Không thay source/model/trọng số trong nhiệm vụ nghiệm thu này.

## Chẩn đoán cosine đã hoàn tất — 14/11/2026

Người dùng đã chạy hai audit mới đủ5.000query. Các dự đoán cũ khớp hoàn toàn;
selector cosine giảm accuracy0,30/0,18điểm % so selector gốc. Frozen retrieval
top-1 đạt87,94%/87,76%, gần selector87,86%/87,82%; chưa có thắng lợi ổn định
cho learned reranking hoặc cosine. Không cần chạy lại bước được đề xuất ở trên.
Xem [EFFECTIVE_BATCH_COSINE_AUDIT.md](EFFECTIVE_BATCH_COSINE_AUDIT.md) cho
kiểm tra artifact, norm diagnostics, paired counts và quyết định chuyển sang demo.
