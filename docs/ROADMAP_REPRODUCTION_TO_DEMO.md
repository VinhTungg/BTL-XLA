# Kế hoạch từ hiện tại đến tái hiện, cải tiến, demo và bảo vệ TACS

Ngày lập: **30/10/2026**. Nhóm: **3 người**. Phạm vi chính: **phân loại CIFAR-100 với DINOv3 ViT-S/16**.

Cập nhật 04/11/2026: đã phát hiện và sửa lỗi đối xứng query/context trong task
network DINOv3, có namespace `role_v2` và kiểm thử checkpoint thật; xem
[biên bản sửa và pilot đối chứng](CONTEXT_ROLE_FIX.md). Chưa có accuracy
benchmark role_v2. Các mục kế hoạch dưới đây không tự động được đánh dấu hoàn tất.

Đây là kế hoạch thực hiện tiếp theo, thay cho các mục “đã xong” trong tài liệu thí nghiệm CIFAR-10/tiny trước đây khi đánh giá mức sẵn sàng của bản CIFAR-100/DINOv3. Các nhiệm vụ phía dưới là **chưa hoàn thành**, trừ những artifact đã kiểm chứng tại mục 2. Việc viết kế hoạch không đồng nghĩa đã sửa trainer hay khởi chạy thí nghiệm.

## 1. Mục tiêu, phạm vi và cách hiểu “chuẩn”

Mục tiêu bàn giao là một chuỗi có thể kiểm chứng: **paper → đặc tả → source → cấu hình → checkpoint → kết quả → cải tiến → demo → báo cáo/thuyết trình**.

Không có cơ sở để hứa đạt chính xác 89,2% hay giống tác giả 100% khi paper thiếu chi tiết triển khai. “Chuẩn” trong kế hoạch này nghĩa là: bám mọi chi tiết được công bố; ghi riêng mọi giả định; kiểm tra thuật toán trước khi tốn GPU; so sánh công bằng; có thể chạy lại; trình bày trung thực cả thành công và giới hạn.

Ba sản phẩm cần hoàn thành:

1. **Tái hiện trên CIFAR-100:** đủ bốn baseline của paper và TACS, cùng điều kiện, ba seed; có bảng đối chiếu paper và giải thích sai khác.
2. **Phát triển của nhóm:** Two-Stage TACS, kiểm chứng đánh đổi accuracy, chi phí truy xuất, tổng thời gian và bộ nhớ. Không mặc định phương pháp sẽ nhanh hơn hoặc chính xác hơn.
3. **Demo và bảo vệ:** ứng dụng suy luận từ checkpoint thật, hiển thị query/context/prediction, so sánh phương pháp; bộ báo cáo, slide, video dự phòng và kịch bản ba người.

Không đặt việc tái hiện cả 18 dataset, phân đoạn y tế hoặc một ứng dụng nhận dạng ảnh bất kỳ làm điều kiện hoàn thành. Tên báo cáo phải ghi rõ **tái hiện nhánh phân loại trên CIFAR-100**, không nhận là đã tái hiện toàn bộ bài báo.

Mốc 89,2% là số tham chiếu, không phải ngưỡng đạt môn học và không phải lý do để điều chỉnh cấu hình theo test. Khoảng 88,2–90,2% từng được đề cập không phải tiêu chuẩn khoa học hay tiêu chuẩn do tác giả quy định.

## 2. Trạng thái thực tế lúc bắt đầu

| Hạng mục | Bằng chứng hiện có | Ý nghĩa cho bước tiếp theo |
|---|---|---|
| CIFAR-100 TACS seed 42 | 100 bản ghi epoch; config, metrics, best/latest checkpoint; manifest đã commit | Có kết quả V1 để giữ và đối chiếu |
| Test của V1 | **85,66%**, test loss 0,5490 | Thấp hơn mốc paper 3,54 điểm phần trăm; chưa biết nguyên nhân chính |
| Validation của V1 | Best 85,40%; epoch 100 là 81,02% | Kiểm tra dao động, checkpoint selection và selector; không suy đoán ngay là chỉ cần train thêm |
| Dữ liệu V1 | 36.000 query / 9.000 pool / 5.000 validation / 10.000 test | Có validation riêng; khác diễn giải 20% trên toàn bộ 50.000 train |
| Cấu hình V1 | 256×256, 100 epoch, AdamW, LR 1e-4, WD 1e-4, warm-up 5, batch 32×2, FP16 | Một phần khớp paper; LR/WD/warm-up là lựa chọn của nhóm |
| TACS code | Dot product, Gumbel, reward, policy loss, ghép patch token | Có nền tảng; đường gradient/cache cần audit |
| Similarity code | Lấy encoder riêng bị freeze từ checkpoint No-context, không phải task backbone fine-tuned | Cần tách dependency đường dẫn và audit provenance/protocol trước khi gắn nhãn baseline paper |
| Feature-Averaged | Chưa có mode riêng trong trainer/runner | Phải bổ sung để bảng tái hiện đủ baseline tác giả |
| Two-Stage | Đã có code và thử nghiệm cũ | Chưa phải bằng chứng cải tiến trên protocol CIFAR-100 mới |
| Tổng hợp kết quả | Có multiseed; script analyze_all_models vẫn khóa CIFAR-10 và 5 mode cũ | Cần hỗ trợ dataset/protocol/method mới |
| Demo | Chưa có ứng dụng demo trong source đã kiểm tra | Cần lớp inference, export artifact và UI |

Nguồn V1: [manifest kết quả](CIFAR100_TACS_SEED42_RESULTS.md), `outputs/cifar100/dinov3_vits16/seed_42/tacs/config.json` và `metrics.json`. Commit ghi nhận: `a07ec2f`.

Checkpoint V1 được giữ làm artifact lịch sử và có thể dùng dựng demo sơ bộ. Không trộn kết quả V1 vào mean của các run dùng protocol mới. Không ghi đè thư mục V1 khi triển khai V2.

## 3. Những gì paper thực sự quy định

Nguồn chuẩn: PDF tiếng Anh người dùng cung cấp, **Sec. 3.1–3.2, Fig. 2, Sec. 4.1–4.3, Tables 2/4/5**. Số trang PDF tương ứng phần chính: 3–7.

| Nội dung | Paper công bố | Cách đưa vào kế hoạch |
|---|---|---|
| Backbone | ViT-S/16, khởi tạo DINOv3 | Ghi chính xác checkpoint pretrained, revision và checksum |
| Phân loại | Ghép patch embeddings query/context trước transformer; fine-tune toàn mạng task | Kiểm tra tokens, CLS/register tokens và positional encoding/RoPE |
| Selector | Điểm tích vô hướng; softmax Eq. (2) | Không tự đổi thành cosine/temperature-scaled policy |
| Chọn context | Gumbel-Softmax straight-through khi train; argmax khi inference | Kiểm tra forward và gradient độc lập |
| Reward | CE không-context trừ CE có-context | Detach reward khỏi task network cho policy loss |
| Advantage | Chuẩn hóa reward trong batch | Làm rõ batch thực sự dùng chuẩn hóa |
| Objective | Task loss + λ × policy loss | τ=0,1; λ=1,0 mặc định |
| Optimizer / schedule | AdamW / cosine annealing / 100 epoch | Cố định schedule 100 epoch từ đầu |
| Batch / ảnh | Classification batch 64 / 256×256 | Ưu tiên batch vật lý 64; nếu tích lũy phải ghi sai khác |
| Augmentation | Horizontal/vertical flip, random resized crop | Ghi xác suất, scale, ratio, interpolation do paper không nêu hết |
| Pool | 20% training images, cố định trong train/evaluation; loại query-associated images | Với 50.000 official train: 10.000 pool theo diễn giải trực tiếp |
| Split / seed | Standard split; trung bình ba random seed | Chọn seed nhóm 42/43/44, không nói đây là seed tác giả |
| Hardware | Một A100 | Có thể dùng GPU khác nhưng phải ghi rõ và đo benchmark cùng thiết bị |

Mốc Table 2 trên CIFAR-100:

| Phương pháp tác giả | Accuracy (%) |
|---|---:|
| No-Context | 87,8 |
| Random Context Retrieval | 87,6 |
| Frozen DINO Similarity Retrieval | 88,4 |
| Feature-Averaged Context | 88,6 |
| TACS | 89,2 |

Đây là **bốn baseline đối sánh và một phương pháp TACS**, không phải năm phương pháp do nhóm tự đề xuất. Two-Stage là phương pháp thứ sáu trong bảng nghiên cứu của nhóm.

### 3.1. Các chỗ cần sửa lại cách diễn giải trước đây

1. **“Share a ViT-S/16 backbone” chưa chứng minh weight tying.** Paper nói hai module và hai đường tối ưu chia sẻ tham số Selector; chưa đủ kết luận Selector và Task Network phải là cùng một instance. Không sửa sang shared weights chỉ từ câu này. Tạm giữ hai mạng cùng kiến trúc, ghi giả định, tìm code chính thức; nếu có bằng chứng khác mới sửa.
2. **Pool 10.000 không tự động suy ra chính xác 40.000 query của tác giả.** Paper chưa nói rõ pool có bị loại hoàn toàn khỏi tập query train hay không. Nếu overlap, phải loại self-match/associated images khi truy xuất. Kế hoạch chọn disjoint pool/query làm giả định mặc định, không gán giả định đó cho tác giả.
3. **Batch 32 × accumulation 2 chưa tương đương đầy đủ batch 64.** Code chuẩn hóa reward ngay trong từng micro-batch; policy gradient khác chuẩn hóa trên 64 mẫu. Còn phải kiểm tra weighting của batch cuối, dropout, RNG và số optimizer steps.
4. **Có gradient không có nghĩa gradient đúng theo bản tham chiếu.** Zero-forward embedding proxy trong đường bank hiện tại chưa được chứng minh tương đương straight-through qua ảnh/patch context.
5. **Top-32 không hứa tăng tốc 10.000/32 lần.** TACS Full đã dùng bank và phép nhân ma trận rẻ; Two-Stage vẫn quét pool bằng cosine, có thể thêm encoder query và top-k overhead. Phải đo end-to-end.
6. **Demo không phải đợi mọi run xong mới bắt đầu.** Có thể dựng inference và UI bằng checkpoint V1 trong lúc GPU chạy; artifact cuối sẽ thay bằng phiên bản đã nghiệm thu.

### 3.2. Danh sách chưa xác định và quy tắc giải quyết

Chưa có bằng chứng đầy đủ về: LR, WD, warm-up, optimizer betas/eps, gradient clipping, checkpoint selection, seed cụ thể, pool/query overlap, weight tying, cách rút z từ các token trong Eq. (1), K và điểm ghép Feature-Averaged, candidate gradient/cache refresh, hai action của hai nhánh có độc lập không, chi tiết transform, AMP.

Thứ tự nguồn: **code/config chính thức có phiên bản → supplementary chính thức → paper CVPR → preprint để tham khảo → giả định được ghi trong quyết định kỹ thuật**. Không lấy code bên thứ ba làm bằng chứng chính thức.

Ngày 30/10/2026, kiểm tra trang CVF/arXiv và tìm kiếm theo tên bài chưa xác nhận được repository chính thức. Đây không phải kết luận rằng tác giả không công bố code. P01 dành tối đa một ngày công cho kiểm tra nguồn; sau đó ghi các lựa chọn còn thiếu để không treo cả dự án. Có thể chuẩn bị câu hỏi cho tác giả; gửi thư là việc riêng của thành viên nhóm, không phải điều kiện chặn demo.

## 4. Quy trình tổng thể và điều kiện chuyển giai đoạn

| Giai đoạn | Nhiệm vụ | Đầu ra bắt buộc | Điều kiện được sang bước sau |
|---|---|---|---|
| A. Kiểm kê và đặc tả | P00–P02 | Manifest V1, audit paper/code, protocol dữ liệu | Mỗi lựa chọn có nguồn hoặc nhãn giả định |
| B. Sửa và kiểm chứng | P03–P07 | Thuật toán tham chiếu, baselines, resume, smoke | Gradient/data/checkpoint tests đạt; không có lỗi chưa giải thích |
| C. Pilot và khóa cấu hình | P08 | Pilot, số giờ GPU, config khóa | Có budget, stability, config hash và quyết định go/no-go |
| D. Train nghiên cứu | P09–P11 | Main runs, cải tiến, ablations | Đủ artifact và run ledger; mỗi run đúng protocol |
| E. Đánh giá | P12–P13 | Test, mean±SD, latency, phân tích lỗi | Không trộn protocol; đủ raw predictions và provenance |
| F. Sản phẩm demo | P14–P16 | Inference, UI, bộ cài/offline | Demo trùng kết quả evaluator và chạy trên máy bảo vệ |
| G. Bảo vệ | P17–P18 | Báo cáo, slide, video, diễn tập | Người khác mở và chạy được; ba thành viên giải thích được |

Nhánh demo P14 bắt đầu sau khi nắm checkpoint schema; P15 làm song song với train. Chỉ bước thay artifact cuối, đo hiệu năng cuối và nghiệm thu phụ thuộc kết quả chính thức.

## 5. Protocol dữ liệu và đánh giá cần khóa

### 5.1. Hai chế độ dữ liệu, dùng đúng mục đích

**DEV để lựa chọn cấu hình:** từ official train 50.000, lấy pool 10.000 (100 ảnh/lớp), validation 5.000 (50 ảnh/lớp), query train 35.000 (350 ảnh/lớp), không giao nhau. Test 10.000 official giữ riêng. Đây là split do nhóm thiết kế để phát triển, không phải split được tác giả xác nhận.

**FINAL để đối chiếu gần mô tả paper:** sau khi khóa mọi lựa chọn, gộp query train DEV và validation DEV thành 40.000 query, giữ pool 10.000 như cũ; train 100 epoch từ cùng pretrained checkpoint, đánh giá checkpoint epoch 100 trên test. FINAL không còn validation độc lập, nên không gọi checkpoint là “best validation”. Việc dùng epoch 100 là quy ước đăng ký trước của nhóm do paper chưa nêu checkpoint selection.

Nếu code chính thức xác nhận một protocol khác, cập nhật trước khi khóa FINAL. Nếu dùng toàn 50.000 làm query với pool là subset, phải có self/associated-match masking và ghi rằng pool/query overlap có chủ đích. Không đồng thời chạy nhiều cách chia rồi chọn cách có test cao hơn.

Nếu thiếu GPU, có thể giữ DEV làm benchmark chính và chọn best validation; phải gọi đó là **tái hiện có validation holdout**, ghi rõ 35.000/10.000/5.000, không gọi là đúng hoàn toàn standard-split của tác giả. Các baseline và Two-Stage vẫn phải cùng protocol này. Nhánh đầy đủ mặc định của kế hoạch là DEV → FINAL.

### 5.2. Seed, tính công bằng và không rò rỉ dữ liệu

- Tách `split_seed`, `train_seed`, `eval_random_seed`; mặc định split_seed=42, train seeds=42/43/44. Pool chung giữa các seed để đo biến thiên huấn luyện; đây là lựa chọn nhóm do paper chưa nêu cách thay pool theo seed.
- Lưu danh sách indices, số mẫu/lớp và SHA-256; kiểm tra rời nhau theo vai trò ở chế độ disjoint.
- Tất cả phương pháp cùng query set, pool, transform, số epoch, schedule và quy tắc checkpoint. No-context dùng cùng query train với các phương pháp có context.
- Cùng pretrained weights và cùng cách khởi tạo task head cho từng seed; tránh để thứ tự tạo thêm selector tiêu thụ RNG làm khởi tạo task khác nhau ngoài ý muốn.
- Candidate labels chỉ phục vụ phân tích sau dự đoán và hiển thị demo, không đưa vào bộ chọn như đáp án.
- Validation/test không được thêm vào candidate pool. Resize/eval transform xác định, không augmentation ngẫu nhiên ở evaluation.
- Random baseline có mapping evaluation cố định theo query ID/seed; kết quả không đổi khi thay batch size/thứ tự DataLoader.
- Không chọn LR, K, epoch, seed trình diễn hay phương pháp cuối bằng test accuracy. Mọi lựa chọn dựa trên DEV và được khóa trước test FINAL.
- Test FINAL chỉ đánh giá theo danh sách run đã đăng ký; nếu phát hiện bug phải ghi lại run bị vô hiệu và lý do, không âm thầm thay số.

## 6. Công việc kỹ thuật trước đợt train lớn

### P00. Đóng gói kết quả cũ và lập sổ run

**Người 2 chủ trì, người 1 kiểm tra.** Dựa vào manifest V1 đã có, kiểm tra đủ file; bổ sung run ledger: run ID, protocol, method, seed, checkpoint epoch, source commit, dataset version, config hash, trạng thái, nơi chạy, nơi tải output. Chỉ kiểm kê kết quả Chrome/Brave/Edge còn lại, chưa coi là hoàn thành khi chưa xem metrics/checkpoint.

Giữ ít nhất hai bản artifact quan trọng: bản cloud đã lưu thành version và bản local. Không commit checkpoint lớn vào Git. Phân loại `legacy`, `development`, `final`, `invalid` để các script không cộng nhầm.

**Nghiệm thu:** truy ngược được 85,66% đến checkpoint/config; mỗi run có trạng thái rõ. **Commit:** `docs: inventory experiments and preserve legacy CIFAR-100 results`.

### P01. Lập ma trận paper → code

**Người 1 chủ trì, người 3 đọc chéo.** Tạo `docs/PAPER_ALIGNMENT_AUDIT.md` và cập nhật `docs/DECISIONS.md`. Mỗi hàng gồm trang/công thức, hành vi yêu cầu, file/function hiện tại, sai khác, mức độ chắc chắn, cách kiểm tra và quyết định.

Bao phủ Eq. (1)–(10), Fig. 2, pool, preprocess, token representation, weight sharing, cache, action sampling, baseline, evaluation. Đặc biệt tách “đã chứng minh khác” khỏi “chưa đủ thông tin”. Kiểm tra tác giả có code/supplementary hoặc cấu hình mới hay không; ghi URL, ngày truy cập và commit nếu có.

**Nghiệm thu:** không còn một quyết định lớn được ghi “đúng paper” chỉ vì phỏng đoán. **Commit:** `docs: audit TACS paper alignment and record unresolved assumptions`.

### P02. Bổ sung split và protocol manifest

**Người 2 chủ trì.** Sửa `src/tacs/data.py`, config và test để hỗ trợ DEV/FINAL, pool count rõ ràng, validation rỗng có chủ đích trong FINAL. Tạo split bằng script tái lập; metadata không chỉ dựa vào tên đường dẫn.

Thêm `protocol_id`, split hash, pool hash, split_seed; kiểm tra số lớp/mẫu, không giao nhau, không đưa test vào pool. No-context phải đọc đúng query set, không mặc định dùng tất cả 50.000 ảnh.

**Nghiệm thu:** sinh lại cùng seed ra cùng indices; DEV 35k/10k/5k và FINAL 40k/10k đúng 50k official train; tests phát hiện leakage và sai hash. **Commit:** `feat: add versioned CIFAR-100 development and final protocols`.

### P03. Kiểm chứng thuật toán và đường gradient

**Người 1 chủ trì.** Viết implementation tham chiếu quy mô nhỏ, dùng tensor kiểm thử và smoke trên ảnh thật, cho phép toàn bộ candidates đi qua đường straight-through đã đặc tả. Đây là unit test toán học, không phải mock data dùng báo cáo.

Các phép kiểm tra bắt buộc:

1. Utility score và policy log-prob đúng công thức; τ chỉ tác động đường relaxation như đặc tả đã khóa.
2. Train chọn one-hot; eval argmax xác định; chọn đúng global candidate ID sau top-k.
3. Reward đúng dấu, detach; policy loss không cập nhật task qua reward. Nếu dùng chung weights, phải phân biệt gradient theo từng đường thay vì kiểm tra mọi tham số task bằng zero.
4. Loss task thực sự truyền gradient về selector khi bật đường differentiable; policy-only vô hiệu hóa gradient relaxation đúng chỗ, nhưng task network vẫn học CE.
5. So sánh logits và gradient của đường optimized với reference trên pool nhỏ, cùng weights/noise/dtype. Không chỉ test `grad is not None`.
6. Đo gradient tới query/candidate branches; kiểm tra bank detach và refresh gây khác gì. Nếu chưa tương đương, ghi tên phương pháp là **xấp xỉ bank/proxy**, không đổi nhãn thành bản exact.
7. Kiểm tra CLS/register/patch tokens, chiều RoPE, thứ tự query/context và việc ghép context thực sự làm thay đổi prediction. Verify trường hợp query-only đúng no-context path.
8. Ghi rõ cách dùng CLS/mean/token cho selector và một hay hai sampled actions. Chốt theo code tác giả nếu tìm được; nếu không, ghi lý do chọn và hạn chế.

Ưu tiên sửa reference đúng trước, rồi giảm VRAM bằng chunking/recomputation/checkpointing phù hợp và đo sai khác. Nếu full-pool differentiable training vượt ngân sách, chọn một approximation được ghi rõ, đo ablation của nó trên DEV và dùng nhất quán cho TACS/Two-Stage. Khi đó báo cáo vẫn là bản tái hiện có xấp xỉ, không tuyên bố exact reproduction.

**Nghiệm thu:** có bảng pass/fail/tolerance và quyết định gradient mode; không chạy FINAL với lỗi gradient chưa giải thích. **Commit:** `fix: validate TACS selection and gradient semantics against a reference`.

### P04. Xử lý batch 64 và số học

**Người 1 chủ trì, người 2 benchmark.** Thử batch vật lý 64 trên GPU có sẵn. Nếu OOM, thử activation checkpointing và AMP phù hợp, kiểm tra số học trước khi dùng.

Nếu phải accumulation, triển khai/kiểm tra advantage trên logical batch 64 hoặc ghi rõ approximation micro-batch trong protocol. Việc gom reward phải giữ đúng action/noise và không làm sai scale của gradient; đối chiếu reference FP32, model xác định trên batch nhỏ. Batch cuối dùng đúng số mẫu, không overweight micro-batch ít mẫu.

Log effective batch, advantage batch, optimizer steps, gradient norm, loss task/policy, reward mean/std, selector entropy, embedding norm, AMP scale/skip và NaN/Inf. Không tự thêm clipping/normalize utility để “đẹp số” mà không ghi quyết định.

**Nghiệm thu:** batch policy được mô tả chính xác; smoke hữu hạn; phép so sánh batch/accumulation đạt tolerance hoặc sai khác được công khai. **Commit:** `fix: make reward normalization and accumulation explicit and testable`.

### P05. Đúng và đủ baseline của tác giả

**Người 1 chủ trì, người 3 kiểm chứng.** Giữ tên legacy riêng; mở rộng trainer/runner/analysis với các method ID mới (tên dự kiến, chưa phải CLI hiện có):

| Method ID đề xuất | Cách chọn/ghép context | Việc phải làm |
|---|---|---|
| `no_context` | Chỉ query | Khớp cùng task backbone/head/training budget |
| `random` | Một ứng viên ngẫu nhiên | Kiểm tra reproducibility, loại self-match nếu có |
| `frozen_dino_similarity` | Pretrained DINOv3 frozen, cosine nearest candidate | Không phụ thuộc `no_context/best_encoder.pt`; task model vẫn fine-tune |
| `feature_averaged` | Tổng hợp mềm top-k feature, trọng số theo Eq. (2), không RL | Chốt K, representation, encoder trainability và điểm đưa context vào task network |
| `tacs` | Selector học utility trên pool, hybrid loss | Dùng gradient mode đã nghiệm thu P03 |
| `two_stage` | Frozen coarse top-k → selector học rerank | Đề xuất nhóm; đặc tả tại mục 9 |

Feature-Averaged không phải trung bình xác suất của nhiều classifier, và không phải TACS với λ=0 một cách mặc định. Paper chưa đủ mô tả phần ghép feature; nếu thiếu code chính thức, viết rõ lựa chọn triển khai và xem đó là sai khác còn lại.

Đính chính sau kiểm tra source/artifact ngày 04/11: encoder xuất ở No-context là mạng truy xuất riêng bị freeze, không phải task backbone đã fine-tune. Giữ tên `similarity_legacy`; không gọi `finetuned_similarity_legacy` nếu chưa thực sự huấn luyện encoder truy xuất. Cần audit weights, transform, pool và task architecture trước khi gắn số cũ vào hàng Frozen DINO. Không bắt buộc train No-context trước Frozen DINO mới; dependency đó là của code cũ.

**Nghiệm thu:** mỗi mode chạy smoke, parameter trainability đúng, feature-averaged không gọi policy loss, evaluation nhất quán. **Commit:** `feat: implement paper baselines with frozen DINO retrieval`.

### P06. Resume, schedule và cloud notebook

**Người 2 chủ trì.** Giữ tổng schedule **100 epoch ngay khi bắt đầu**, bổ sung cơ chế `stop_after_epoch`/time budget độc lập với tổng epoch. Tên option là đề xuất cần triển khai, không dùng ngay với CLI cũ.

Checkpoint latest cần model, optimizer, scheduler, AMP scaler, epoch/global step, lịch sử, best state theo DEV, RNG Python/NumPy/Torch/CUDA, sampler state phù hợp; lưu atomic và checksum. Resume kiểm tra protocol/split/pool/model/pretrained/config hash, không chỉ một số field cấu hình. Thay đường dẫn cloud được phép nếu nội dung và hash khớp.

Nghiệm thu bằng thử nghiệm ngắn A chạy liên tục và B dừng/resume từ cùng khởi tạo: so LR, optimizer steps, lựa chọn context và model metrics trong tolerance định trước. Multi-worker/GPU có thể không bitwise-identical; ghi giới hạn. Không fallback âm thầm từ best checkpoint thiếu optimizer sang “resume chính xác”.

Notebook Kaggle gồm: xác định bundle/dataset version → cài có log → import/preflight → kiểm tra GPU/disk → kiểm tra hashes → copy checkpoint vào nơi ghi được → plan → train từng đoạn → xác minh checkpoint → lưu output version. Tự tìm artifact bằng manifest, tránh đoán đường dẫn hoặc tự chọn file tên giống nhau.

**Nghiệm thu:** session mới đọc output session trước và tiếp tục đúng epoch/LR; log live có epoch, batch, elapsed/ETA, loss/acc; dừng có kiểm soát trước time limit. **Commit:** `fix: preserve training schedules and reproducibility across cloud resumes`.

### P07. Smoke end-to-end và audit kết quả V1

**Người 2 chủ trì.** Test toán học trên CPU; smoke 1–2 epoch trên tập con CIFAR-100 thật cho mọi mode. Integration dùng DINOv3 để phát hiện lỗi không thấy trên model tiny. Chạy bất kỳ test ảnh thật nào bằng cơ chế opt-in; không dùng kết quả smoke làm accuracy cuối.

Nạp lại V1 best checkpoint bằng source tương ứng để kiểm tra evaluation tái lập, selected IDs và cache đúng model. Xem lịch sử từ best validation tới epoch 100, reward, encoder norm/entropy nếu có; kiểm tra scaler/scheduler và candidate bank. Nếu log thiếu, ghi chưa đủ dữ liệu, không kết luận nguyên nhân chính xác.

**Nghiệm thu:** artifacts hợp lệ, predictions xác định ở eval, mọi gradient/data/resume gate đạt. **Commit:** `test: verify paper-protocol training and artifact integrity`.

## 7. Pilot và khóa cấu hình trước khi chạy dài

### P08. Pilot có giới hạn ngân sách

**Người 1 và 2.** Dùng DEV, không xem test. Tổng schedule là 100, dừng sớm bằng cơ chế riêng; không dùng cosine 10 epoch rồi đổi lên 100 để gọi là cùng thí nghiệm.

Trình tự mặc định:

1. Đo 2–3 epoch No-context và TACS sau warm-up kỹ thuật để ước tính thời gian, VRAM, thời gian bank và validation. Lần này ưu tiên độ ổn định, chưa chọn mô hình theo accuracy sớm.
2. Nếu có recipe chính thức, dùng recipe đó. Nếu chưa có: khảo sát LR **3e-5, 1e-4, 3e-4**, giữ WD=1e-4 và warm-up=5 theo khởi điểm nhóm. Đây là grid đề xuất, không phải thông số tác giả.
3. Chạy No-context 10 epoch cho mỗi LR, dùng cùng seed/initialization. Loại NaN/Inf/divergence; xem cả loss curve, không chỉ accuracy của một epoch.
4. Hai lựa chọn ổn nhất tiếp tục DEV đến 20 epoch; thử TACS với hai lựa chọn đó (cùng τ=0,1; λ=1,0). Không giảm λ để làm đẹp kết quả tái hiện mặc định.
5. Nếu tất cả không ổn, quay lại audit/số học. Nếu ổn nhưng ngang nhau, ưu tiên cấu hình đơn giản và ổn định; không mở vô hạn grid WD/warm-up. Tối đa một vòng bổ sung có lý do từ log.
6. Pilot ngắn có thể đánh giá sai cấu hình dài hạn. Cho No-context và TACS cấu hình đã chọn hoàn thành DEV 100 epoch như kiểm tra hội tụ trước ma trận FINAL. Đây là hai run phát triển, không cộng vào ba seed FINAL.

Mục tiêu pilot: loss hữu hạn; selector có gradient đúng; reward không collapse vô lý; không lỗi cache; resume đúng; budget khả thi. Không đặt “phải đạt X% ở epoch 10” khi chưa có learning curve của tác giả.

Khóa `configs/...`/manifest với: protocol, method definition, seeds, pretrained revision/hash, image transform, gradient mode, τ/λ, logical/micro batch, advantage batch, LR/WD/warm-up/betas/eps/min-LR, AMP/dtype, epoch100, checkpoint rule, K rule, code commit và dependency lock. Đây là file cần bổ sung, không giả định đã tồn tại.

**Nghiệm thu:** hai người đọc config và ký nhận trong run ledger; có dự toán GPU; không còn thay lựa chọn dựa trên test. **Commit:** `chore: freeze the CIFAR-100 experimental protocol after development pilots`.

## 8. Ma trận train chính thức

### P09. Tái hiện năm phương pháp trên FINAL

**Người 2 điều phối; cả ba người vận hành seed được giao.**

Mỗi phương pháp dùng seeds **42, 43, 44**, 100 epoch, epoch-100 checkpoint, từ pretrained và task-head initialization đã khóa. Không lấy checkpoint cũ 100 epoch fine-tune tiếp rồi gọi là một run 100 epoch từ đầu.

| Thứ tự ưu tiên | Phương pháp | Số run FINAL | Vai trò |
|---:|---|---:|---|
| 1 | No-context | 3 | Kiểm tra nền tảng fine-tuning |
| 2 | Frozen DINO Similarity | 3 | Mốc đối sánh retrieval đúng paper |
| 3 | TACS Full | 3 | Kết quả tái hiện chính |
| 4 | Random Context | 3 | Kiểm soát lợi ích do thêm ảnh bất kỳ |
| 5 | Feature-Averaged | 3 | Kiểm soát soft aggregation, không RL |
| 6 | Two-Stage với K đã khóa | 3 | Cải tiến nhóm, sau P10 |

Tổng đầy đủ: **18 run × 100 = 1.800 model-epochs**, chưa tính DEV, ablation và thử lỗi. Các mode có chi phí khác nhau, model-epoch không phải một đơn vị GPU-time đồng nhất.

Ba người nhận một seed mỗi người; mỗi người chạy tất cả method của seed đó. Dùng cùng GPU model nếu có thể; nếu GPU khác nhau, log riêng và đo lại inference của toàn bộ checkpoint trên cùng một máy. Không gán cố định TACS vào GPU mạnh và baseline vào GPU yếu rồi so thời gian train trực tiếp.

**Nghiệm thu P09:** năm phương pháp tái hiện có đủ ba seed theo kế hoạch đầy đủ, mỗi run có epoch 100 và manifest đúng config đã khóa; mọi run thiếu hoặc lỗi được ghi rõ. **Commit:** `docs: register completed paper-baseline and TACS training runs`.

### 8.1. Điều kiện mở rộng và phương án thiếu GPU

Sau DEV hội tụ và một cặp FINAL No-context/TACS đầu tiên, chỉ kiểm tra tính hợp lệ kỹ thuật và hoàn thành lịch; không tune theo test. Có thể chạy benchmark/demo với artifact đã có trong khi chờ các seed còn lại.

Nếu deadline/GPU không đủ, thứ tự cắt giảm được công bố trước:

1. Giữ **No-context, Frozen DINO, TACS, Two-Stage × 3 seed = 12 run**.
2. Random và Feature-Averaged chạy seed42 trước: tổng **14 run**, đánh dấu `n=1` cho hai hàng; không gọi là tái hiện đầy đủ Table 2 ba seed.
3. Nếu vẫn thiếu: bốn phương pháp chính ×1 seed =4 run, cộng hai baseline còn lại nếu đủ. Đây là demo và nghiên cứu thăm dò, không có bằng chứng độ ổn định đa seed.
4. Không giảm epoch một phương pháp rồi so công bằng với phương pháp khác 100 epoch; mọi so sánh rút gọn phải cùng budget và ghi rõ.

Không cần train thêm dataset mới chỉ để có demo. Dataset thứ hai là mở rộng tùy chọn sau khi sản phẩm chính hoàn thành.

## 9. Cải tiến của nhóm: Two-Stage TACS

### P10. Giả thuyết và thiết kế

**Người 1 thiết kế, người 2 đo đạc, người 3 phân tích và trình bày.**

Giả thuyết cần kiểm chứng: **giới hạn vùng tìm kiếm của selector bằng một bộ lọc rẻ có thể giảm chi phí mà giữ phần lớn lợi ích của context**.

Pipeline dự kiến:

```text
Query → Frozen DINO coarse embedding → cosine với pool cố định
      → lấy Top-K IDs → selector TACS rerank K ứng viên
      → ảnh context thắng → task network phân loại query/context
```

Giữ task architecture, losses, training budget và gradient/cache mode của bản TACS đã khóa; chỉ thay tập ứng viên được selector xét. Coarse encoder mặc định pretrained frozen để không cần train thêm encoder từ No-context; lựa chọn này được đối chiếu công bằng với Frozen DINO baseline. Dùng fine-tuned coarse encoder là ablation riêng nếu có lý do và ngân sách.

K khảo sát trên DEV: **16, 32, 64, 128**; thêm K=N làm kiểm tra tính đúng thứ tự/masking và tương đương full selection khi đầu vào, action noise và model giống nhau. K=N vẫn có overhead coarse retrieval, không kỳ vọng runtime bằng Full.

Đo trước khi train lớn:

- Tỷ lệ context do Full chọn nằm trong shortlist Top-K, gọi là **coverage của lựa chọn Full**, không gọi đó là ground-truth relevance.
- Accuracy DEV khi áp shortlist vào checkpoint Full: chỉ là khảo sát pruning tại inference, không phải kết quả mô hình Two-Stage đã train.
- Coarse encoder time, quét cosine, top-k, selector time, tải context, task forward và tổng thời gian.
- Độ nhạy với pool size trên các subset DEV có cùng quy tắc; không trộn accuracy các pool size vào bảng chính.

### 9.1. Chọn K và tiêu chí thành công

1. Profile cả bốn K bằng checkpoint DEV Full; chọn tối đa hai K có trade-off tốt để train pilot 20 epoch trên DEV từ cùng initialization.
2. Chọn K theo DEV accuracy và latency; nếu cần, hoàn tất một run DEV 100 epoch của K được chọn để xác nhận hội tụ. Khóa K trước test FINAL.
3. Train Two-Stage từ pretrained cùng điều kiện FINAL, ba seed. Không chỉ đổi K trên checkpoint Full rồi tuyên bố đã train cải tiến.

Mục tiêu nội bộ đề xuất: accuracy giảm không quá **0,5 điểm phần trăm**, latency end-to-end batch1 giảm ít nhất **20%** (speedup ≥1,25×), cùng thiết bị/dtype và điều kiện bank. Đây là mục tiêu thiết kế, không phải kết quả đã đạt; báo cáo cả mean, per-seed và độ biến thiên.

Nếu chỉ bước selector nhanh hơn nhưng tổng thời gian chậm hơn, kết luận đúng là giảm phạm vi scoring, chưa chứng minh tăng tốc hệ thống. Nếu accuracy giảm >0,5 điểm, ghi trade-off; có thể xem mức 1 điểm như phương án triển khai do người dùng lựa chọn, không đổi tiêu chí sau khi nhìn test.

**Rủi ro cụ thể:** bank 10.000×384 chỉ khoảng 15,36 MB ở FP32 (14,65 MiB); dot product Full khá rẻ. Two-Stage có thể không có lợi khi thêm query encoder và top-k. Nếu profile DEV xác nhận điều này, dừng mở rộng vô ích; vẫn báo cáo kết quả âm. Chỉ phát triển ANN/query embedding dùng chung/cơ chế thích ứng thành một biến thể mới nếu DEV chứng minh nút thắt thực tế và còn budget; không thay đổi âm thầm định nghĩa Two-Stage đã khóa.

**Nghiệm thu:** có lựa chọn K có lý do, runtime breakdown và ba checkpoint FINAL hợp lệ; kết luận có thể là có hoặc chưa có cải thiện. **Commit code:** `feat: add measurable coarse-to-fine TACS retrieval`; **commit kết quả:** `docs: report Two-Stage accuracy and efficiency trade-offs`.

### P11. Ablation để chứng minh đóng góp

**Người 2 chủ trì, người 1 kiểm tra cấu hình ablation.** Tối thiểu trên DEV seed 42, cùng budget với control DEV tương ứng:

| Ablation | Thay đổi duy nhất | Câu hỏi trả lời |
|---|---|---|
| Full hybrid | τ=0,1; λ=1,0 | Mốc gốc |
| Không policy | λ=0, giữ differentiable path đã đặc tả | Policy có đóng góp không? |
| Policy-only selector | Ngắt relaxation tới selector, giữ policy và task CE | Gradient differentiable có đóng góp không? |
| K sweep | 16/32/64/128 | Accuracy/coverage/latency đổi ra sao? |
| Bank/proxy approximation | Chỉ khi dùng approximation; đối chiếu reference trên quy mô khả thi | Mức sai khác do tối ưu hệ thống? |

Full hybrid và Two-Stage K đã chọn được tái dùng làm control đúng protocol/budget, không train trùng không cần thiết. Feature-Averaged và “λ=0” là hai phương pháp khác nhau nếu kiến trúc/hard-soft selection khác nhau; không dùng chung tên.

Hai ablation loss nếu đưa vào kết luận hội tụ nên chạy đủ 100 epoch; nếu chỉ 20 epoch, gắn nhãn screening và giới hạn kết luận. Có thể mở rộng ba seed sau phần chính nếu budget cho phép; seed 42 đơn lẻ không chứng minh ý nghĩa thống kê.

**Nghiệm thu:** mỗi hàng đổi đúng một yếu tố, có config và control; không lấy kết quả tiny/CIFAR-10 cũ làm số cho DINOv3/CIFAR-100. **Commit:** `docs: add controlled TACS objective and shortlist ablations`.

## 10. Ngân sách GPU, thời gian và lưu trữ

### 10.1. Dự toán từ log có thật

V1 có tổng `train_end_to_end_time_s` khoảng **99.066,2 giây**, validation khoảng **10.323,1 giây**. Tổng train+validation khoảng **30,39 giờ** cho 100 epoch trên run T4 đã thu thập; chưa bao gồm đầy đủ thời gian chờ quota, upload/download và setup. Các epoch dao động, và V2 có thể thay đổi chi phí đáng kể.

Chỉ để dự phòng ngân sách, nếu dùng 30,39 giờ làm một đơn vị run tham chiếu:

| Kịch bản | Ước lượng tương đương V1 | Giới hạn diễn giải |
|---|---:|---|
| TACS ×3 seed | 91,2 GPU-giờ | V2 chưa được đo |
| 18 run chính | 547,0 GPU-giờ | Không phải forecast chính xác vì baseline rẻ hơn/TACS reference có thể đắt hơn |
| 18 run chia 3 GPU liên tục | Khoảng 7,6 ngày máy chạy lý tưởng | Không tính DEV, ablation, quota, lỗi và thời gian con người |

Sau P08 phải thay bằng công thức đo riêng từng method:

`GPU-hours = tổng_m [số_run_m × (setup_m + 100 × train_epoch_m + eval_m)] / 3600`

DEV cộng validation mỗi epoch; FINAL dùng quy tắc evaluation đã khóa. Đưa dự phòng **20–30%** cho chạy lại kỹ thuật, I/O và dao động; thời gian chờ quota ghi riêng. Tổng calendar dựa vào GPU-hours khả dụng thực tế mỗi ngày, không chỉ số tài khoản.

### 10.2. Vận hành cloud và từng phiên

- Mỗi thành viên dùng tài nguyên/tài khoản mình được phép dùng; phân công theo seed để tránh trùng run. Ghi notebook ID và version, không dùng tên trình duyệt làm danh tính thí nghiệm.
- Trước phiên: kiểm tra quota/time limit/disk hiện tại trên nền tảng; không dựa vào một con số giới hạn cố định trong kế hoạch.
- Ước lượng số epoch/phiên từ thời gian đo được, chừa 15–30 phút để lưu output. Nếu nền tảng thay giới hạn thì điều chỉnh stop_after, giữ schedule100.
- Chỉ đọc dataset/input; ghi vào working/output. Copy latest sang nơi writable rồi resume, không giải nén bundle đè checkpoint đang chạy.
- Sau phiên: kiểm tra epoch/log/config hash/checksum, lưu version output có trạng thái thành công, thử nạp checkpoint bằng process mới. Có file best chưa đủ chứng minh xong100 epoch.
- Chuyển cloud/GPU: dùng đúng bundle/dependencies hoặc ghi thay đổi; restore optimizer/scheduler/scaler/RNG, xác minh LR đầu phiên. Không đổi baseline/seed/split để resume chung một thư mục.
- Pilot không auto-eval test ở cuối mỗi đoạn; runner hiện có default `eval_test=True` cần được sửa/điều khiển đúng chế độ.

### 10.3. Output và dung lượng

V1 ba file `.pt` khoảng **778,44 MB thập phân**, chưa tính data, cache, nhiều version hoặc dung lượng giải nén. 18 bộ tương tự khoảng14,01 GB; đây chỉ là phép tính dung lượng tham khảo, không đảm bảo vừa một hạn mức20 GB vì version/source/data/temp có thể cộng thêm.

Mỗi run giữ latest để resume và artifact cuối theo protocol; DEV giữ best + latest, FINAL giữ epoch100 + latest nếu khác. Sau xác minh và sao lưu mới dọn bản trùng theo mục tiêu cụ thể, không xóa rộng toàn output. Cân nhắc không xuất `best_encoder.pt` thừa cho mode không cần.

Checkpoint dùng demo chỉ cần weights và metadata; optimizer/scaler vẫn lưu ở archive nghiên cứu để tiếp tục hoặc kiểm toán. Cache cuối gắn checkpoint hash; không chuyển bank từ model khác vào demo.

## 11. Đánh giá và báo cáo số liệu

### P12. Evaluation độc lập và tổng hợp

**Người 2 chủ trì.** Bổ sung evaluator nạp checkpoint, kiểm tra protocol/hash, dựng candidate bank từ đúng selector checkpoint, chạy `eval()`/`inference_mode`, tắt Gumbel. Với FINAL đánh giá epoch100, DEV dùng best-validation rule đã khóa.

Xuất dự đoán từng mẫu: query ID, true label, predicted label, class scores, context ID, method/seed/protocol/checkpoint hash. Có thể xuất `context_label` cho phân tích sau, không dùng làm input dự đoán.

Sửa analyzer hiện tại để không hardcode CIFAR-10 hoặc năm mode cũ. Từ chối gộp khác split, architecture, seed list, training budget, checkpoint rule hoặc approximation mode. Cho phép bảng hiển thị method thiếu seed với `n` rõ, không im lặng lấy trung bình phần còn lại.

Bảng chính có: paper accuracy, từng seed, mean, sample SD (`ddof=1`), n, chênh lệch điểm phần trăm với paper, protocol deviations. Bảng cải tiến có paired delta Two-Stage−TACS theo cùng seed. Ba seed ít: tránh diễn đạt “chắc chắn có ý nghĩa thống kê”; nếu bootstrap theo ảnh cần nói đó là bất định theo mẫu, không thay thế bất định do train seed.

Kết luận tách hai câu hỏi: mức accuracy tuyệt đối gần paper đến đâu, và TACS có lợi hơn No-context/Frozen DINO trong cùng protocol nhóm hay không. Nếu cả No-context và TACS đều thấp, xem fine-tuning/data; nếu No-context gần mà TACS thấp, xem selector/loss/cache. Đây là hướng chẩn đoán, không chứng minh nhân quả tự động.

**Nghiệm thu:** tái tạo bảng từ artifact không nhập tay; mean tính đúng, đơn vị % và điểm phần trăm nhất quán. **Commit:** `feat: evaluate and aggregate protocol-aware CIFAR-100 experiments`.

### P13. Đo tốc độ, bộ nhớ và tính hữu ích context

**Người 2 và 3.** Đo mọi method trên **cùng máy**, cùng image size, dtype, pool, cache state và batch size. Dùng seed42 đã định trước để profile chính, có thể kiểm tra thêm seed nếu chọn context làm thay đổi I/O đáng kể.

- **Cold start:** thời gian load model, build/load bank, bộ nhớ và storage cache.
- **Warm online batch1:** không cache sẵn query; tính preprocessing, query embedding, coarse search nếu có, TACS scoring, tải context và task forward; báo median/p95. UI/network có thể đo riêng.
- **Batched throughput:** batch cố định, báo query/s và amortized ms/query; không gọi ms/query từ batch32 là độ trễ tương tác batch1.
- **Training:** epoch time, time-to-completion, bank refresh, validation, peak VRAM. Tách inference peak với training peak; V1 9.116,61 MB là peak tổng, không lấy nguyên làm VRAM demo.
- Warm-up khoảng 30 lượt, đo ít nhất 1.000 query cố định và lặp 3 lần; synchronize CUDA hoặc dùng CUDA events đúng đoạn GPU; ghi RAM/allocated/reserved, không chạy workload khác đồng thời.
- Nếu preload ảnh pool vào RAM để tăng tốc, áp cùng chính sách cho cả hai method và tính RAM đó. Nếu một mode dùng query cache còn mode kia online thì bảng không công bằng.

Phân tích context: cùng lớp/khác lớp, CE có-context trừ no-context trên cùng task network, tỷ lệ reward dương, số ứng viên bị chọn quá thường xuyên, các trường hợp context giúp/hại. Context khác lớp không có nghĩa chọn sai. Không dùng nhãn “context có cùng lớp” làm thước đo duy nhất của chất lượng TACS.

Chọn ví dụ minh họa gồm trường hợp đúng, được context giúp, bị context hại, cả hai phương pháp sai và ảnh ngẫu nhiên. Ghi ID, cách chọn; hình minh họa không thay thế accuracy toàn bộ test.

**Nghiệm thu:** raw measurements + config + script tái lập; ít nhất accuracy-vs-latency plot, K trade-off, learning curves và ví dụ lỗi. **Commit:** `feat: benchmark retrieval and inference with auditable measurements`.

## 12. Demo: sản phẩm cần xây và hành vi cụ thể

### P14. Lớp inference và bộ artifact

**Người 3 chủ trì, người 1 review.** Tạo module inference độc lập trainer, ví dụ `src/tacs/inference.py`, và exporter, ví dụ `scripts/export_demo.py` (đường dẫn dự kiến).

Bộ demo gồm: task/selector weights phù hợp, pretrained coarse encoder nếu cần, candidate bank đúng checkpoint, pool image IDs/ảnh, 100 class names, eval transform, protocol manifest và test examples. `best_encoder.pt` một mình không đủ: nó không thay thế toàn bộ selector + task network + classifier.

API chung nhận ảnh/query ID và method, trả prediction top-k, context ID/ảnh, score model, timing breakdown và provenance. True label là metadata hiển thị, không bắt buộc và không được đưa vào lựa chọn context.

Trước UI, kiểm tra ít nhất 100 query với evaluator: class scores trong tolerance, predicted class/context ID nhất quán; kiểm tra ảnh RGB/grayscale/RGBA và lỗi input; không tải weights mạng khi đã đủ artifact offline. Lưu cache riêng theo method/checkpoint/pool/transform hash.

**Nghiệm thu:** chạy inference thực từ checkpoint V1 trước, rồi nạp được artifact mới bằng cùng API; không vô tình train khi mở ứng dụng. **Commit:** `feat: export checkpoints and expose deterministic demo inference`.

### P15. Giao diện trình diễn

Đề xuất **Gradio chạy local** cho giao diện nghiên cứu nhỏ; pin phiên bản đã test. Không cần triển khai website công khai để đủ demo. UI dự kiến:

| Vùng | Nội dung |
|---|---|
| Đầu trang | Tên đề tài, dataset, protocol/model version đang dùng |
| Input | Chọn test image bằng ID/lớp hoặc nút ngẫu nhiên; upload tùy chọn |
| Query | Ảnh gốc, nhãn thật nếu có; CIFAR-100 gốc32×32 và ảnh model resize256×256 |
| So sánh | No-context, Frozen DINO, TACS và Two-Stage theo cùng query |
| Mỗi kết quả | Predicted class, top-5 model scores, ảnh context, thời gian online |
| Cải tiến | Pool N, K cố định đã train, số ứng viên selector xét, coarse/re-rank/task time |
| Kết quả nghiên cứu | Bảng mean±SD đã tổng hợp; tách rõ số benchmark đã đo khỏi lần bấm hiện tại |
| Chi tiết | Seed, checkpoint epoch/hash, thiết bị và chế độ live/offline |

Nhãn softmax là **điểm dự đoán của mô hình**, không khẳng định là xác suất đúng đã hiệu chuẩn. Khi upload ảnh ngoài CIFAR-100, ghi rõ model chỉ chọn trong100 lớp đã học; đây không phải kết quả benchmark.

K của demo mặc định đúng K checkpoint. Nếu cho đổi K để khảo sát inference thì ghi “thay K khi suy luận, chưa train lại”, không gán mọi K cùng một accuracy báo cáo. K=N không làm checkpoint Two-Stage trở thành checkpoint TACS Full đã train.

Cho phép chọn method/load lười nếu VRAM máy demo không chứa tất cả model; ghi load time riêng. Mặc định seed42 được chọn trước khi xem test, không chọn seed test cao nhất cho demo rồi coi đó là mean nhóm.

**Nghiệm thu:** nút chọn ảnh, chạy, đổi ảnh và so sánh hoạt động; prediction có nguồn model thật; không mock số liệu; phần lỗi file/model thông báo rõ. **Commit:** `feat: add an interactive TACS comparison demo`.

### P16. Đóng gói và diễn tập kỹ thuật

**Người 3 chủ trì, người 2 chạy trên máy thứ hai.** Cung cấp dependency lock, hướng dẫn cài và một lệnh khởi động sau khi đã triển khai; không đưa lệnh demo giả vào README trước khi có module.

Kiểm tra máy bảo vệ thật: CUDA/CPU fallback, tổng RAM/VRAM, thời gian khởi động, màn hình máy chiếu, font tiếng Việt, port. Thử tắt Internet sau cài; model/data/cache phải ở local. CPU có thể chậm, phải đo trước để chọn số method hiển thị và câu chuyện demo phù hợp.

Kiểm tra20 lượt chọn ảnh/đổi phương pháp không crash, không tăng bộ nhớ liên tục, không dùng cache query sai, không đưa ảnh query vào pool. Có thể live bằng checkpoint V1 nếu các run mới chưa xong, nhưng phải hiện version và số liệu85,66% của V1 rõ ràng.

Chuẩn bị video 2–3 phút và hình/screenshots từ lần chạy thật. Nếu mất GPU dùng CPU; nếu không đủ thời gian live dùng video, nói rõ đó là bản ghi. Không hiển thị dữ liệu lưu sẵn như một lần inference trực tiếp.

**Nghiệm thu:** người không viết UI mở bộ hướng dẫn, khởi chạy và thực hiện kịch bản trong một lần thử; có backup hoạt động. **Commit:** `docs: package and verify the offline demonstration`.

## 13. Báo cáo, slide và kịch bản bảo vệ

### P17. Báo cáo cuối

**Người 3 biên tập, cả nhóm kiểm chứng.** Cấu trúc đề xuất:

1. Bài toán, mục tiêu môn học, phạm vi CIFAR-100 và lý do chọn.
2. Phương pháp tác giả: selector/task network, Eq.(1)–(10), bốn baseline và kết quả paper.
3. Bản tái hiện: nguồn code, protocol, giả định/sai khác, môi trường, quy tắc resume và verification.
4. Cải tiến nhóm: giả thuyết Two-Stage, K, chi phí tính toán, điều kiện có thể có lợi.
5. Thực nghiệm: main table, seed, ablation, accuracy/latency/memory, qualitative/error analysis.
6. Demo, giới hạn, kết luận theo bằng chứng và hướng phát triển.

Mỗi bảng ghi dataset/backbone/protocol/epoch/seed/checkpoint rule. Các bảng CIFAR-10/tiny cũ đưa phụ lục có nhãn lịch sử; không để chữ “hoàn thành100%” của giai đoạn cũ gây hiểu sai. Accuracy paper nằm cột nguồn ngoài, không gộp vào mean nhóm. Khẳng định cải tiến chỉ khi phép đo hỗ trợ; kết quả âm cũng trình bày nguyên nhân và trade-off.

Đầu ra: source báo cáo, bản PDF cuối đã kiểm tra, thư mục figures, bảng số xuất tự động, tài liệu tham khảo. **Commit:** `docs: consolidate reproduction and improvement results for the final report`.

### P18. Slide, phân vai và diễn tập

Kịch bản mặc định **12 phút trình bày + 3 phút demo**, điều chỉnh theo thời lượng cô yêu cầu khi biết. Khoảng12 slide chính:

| Slide | Nội dung | Người / thời lượng gợi ý |
|---:|---|---|
| 1 | Đề tài, tác giả paper, thành viên, phạm vi | N1 / 30 giây |
| 2 | Ví dụ query và vì sao similarity chưa chắc hữu ích | N1 / 60 giây |
| 3 | Kiến trúc Selector → Context → Task | N1 / 60 giây |
| 4 | Hai tín hiệu học: task gradient và policy reward | N1 / 90 giây |
| 5 | Nhóm tái hiện gì, giả định/sai khác nào còn lại | N1 / 60 giây |
| 6 | Split, seeds, protocol và chống leakage | N2 / 60 giây |
| 7 | Kết quả paper và nhóm, mean±SD | N2 / 75 giây |
| 8 | Two-Stage và giả thuyết cải tiến | N2 / 60 giây |
| 9 | Accuracy–latency, K sweep, bộ nhớ | N2 / 75 giây |
| 10 | Ablation, trường hợp giúp/hại, giới hạn | N3 / 75 giây |
| 11 | Thiết kế demo và hướng dẫn nhìn kết quả | N3 / 45 giây |
| 12 | Đóng góp, kết luận đúng bằng chứng, hướng tiếp | N3 / 30 giây |

Demo3 phút: (0:00–0:30) nêu checkpoint/dataset, chọn ảnh; (0:30–1:20) chạy No-context và TACS, chỉ context và dự đoán; (1:20–2:10) chạy Two-Stage, giải thích K và timing; (2:10–2:40) thử ảnh ngẫu nhiên hoặc trường hợp sai; (2:40–3:00) liên hệ với bảng đánh giá toàn test. Giữ app/bank nạp sẵn; cold-start được báo ở bảng riêng.

Câu hỏi cả ba người phải trả lời được:

- Nhóm dựng lại phần nào của tác giả và tự làm thêm gì?
- Vì sao query/context khác lớp vẫn có ích? Reward đo điều gì?
- Gumbel-Softmax và policy loss có vai trò khác nhau thế nào?
- Vì sao tách pretrained frozen encoder khỏi đường dẫn artifact No-context cũ, và kiểm chứng hai encoder có cùng weights như thế nào?
- Pool lấy ở đâu? Có dùng ảnh test hoặc nhãn test để chọn context không?
- Ba seed có nghĩa gì? Tại sao không chỉ báo seed tốt nhất?
- Tại sao chưa bằng89,2%? Điểm nào chắc chắn khác, điểm nào chưa biết?
- Two-Stage có thực sự nhanh hơn khi Full đã cache embedding không?
- Vì sao cùng effective batch vẫn có thể khác policy gradient?
- Demo có train lại không? Checkpoint nào được dùng? Nếu hết mạng/GPU thì sao?
- Có gì chứng minh đóng góp nếu cải tiến không tăng accuracy hoặc không tăng tốc?

Diễn tập lần 1 tập trung tính đúng và câu hỏi; lần 2 bấm giờ, đổi người mở app và dùng video dự phòng; lần 3 trên máy/màn hình dự kiến nếu có. Chốt artifact trước buổi bảo vệ ít nhất 24 giờ, tránh thay thuật toán vào phút cuối.

**Nghiệm thu:** mọi hình/bảng có nguồn; không có slide ghi số kỳ vọng như số đã đo; ba người nói đúng nhiệm vụ; video và app chạy được. **Commit:** `docs: finalize presentation, demo script and defense checklist`.

## 14. Phân công ba người và cơ chế bàn giao

| Vai trò | Trách nhiệm chính | Seed train | Người kiểm tra |
|---|---|---:|---|
| Người 1: thuật toán | P01/P03/P04/P05, khóa recipe, review inference | 42 | N2 kiểm thử; N3 đối chiếu paper |
| Người 2: thực nghiệm | P00/P02/P06/P07/P08/P09/P11/P12/P13, artifacts | 43 | N1 duyệt config; N3 kiểm bảng |
| Người 3: demo/trình bày | P14/P15/P16/P17/P18, figures, ledger hỗ trợ | 44 | N1 kiểm model; N2 chạy máy khác |

Nhãn N1/N2/N3 là vai trò, có thể gán tên thực tế sau. Mỗi người chịu artifact seed của mình cho mọi method, không cần sở hữu riêng một baseline. Người3 phát triển demo trong khi GPU chạy; người1/2 không phải chờ hết train để review UI.

Mỗi bàn giao kèm run ID, source commit, command/config, hash dữ liệu/model, epoch, metrics, log, đường dẫn artifact và vấn đề còn mở. Mỗi nhiệm vụ có commit riêng; **không push**, theo yêu cầu người dùng. Commit code và commit kết quả khác nhau; không commit weights/data/tokens.

Nếu giao tiếp nguồn qua file zip, tên bundle chứa commit/hash và manifest; chạy preflight từ chối code version không khớp. Không sửa trực tiếp notebook của từng người theo các phiên bản khác nhau mà không đưa sửa đổi chung về repo.

## 15. Lịch triển khai theo mốc tương đối

Chưa có deadline bảo vệ và số GPU-giờ khả dụng, nên dùng D0 là ngày bắt đầu thực hiện. Lịch dưới là khung công việc; chỉ chốt ngày hoàn thành train sau P08.

| Mốc dự kiến | Công việc | Phần làm song song |
|---|---|---|
| D0–D1 | P00/P01: inventory, audit, chốt nguồn/giả định | N3 dựng nội dung report và danh sách UI |
| D1–D3 | P02–P06: split, thuật toán, baseline, batch, resume | P14 đọc/export checkpoint V1 |
| D3–D4 | P07: kiểm thử kỹ thuật, smoke GPU | P15 UI với V1 |
| Từ D4 | P08: pilots, DEV hội tụ, dự toán lại | Hoàn thiện inference/UI/test offline |
| Sau gate P08 | P09 FINAL, chia seed; P10 shortlist đã khóa → Two-Stage FINAL | Viết method/report, kiểm output từng phiên |
| Trong/sau train chính | P11 ablation theo budget | Chuẩn bị hình/slides chưa điền kết luận |
| Sau đủ checkpoints | P12–P13: test, aggregate, benchmark cùng máy | P16 thay artifact demo cuối |
| 2–3 ngày cuối | P17/P18: chốt tài liệu, video, diễn tập | Backup, bàn giao bộ demo |

Đây không phải cam kết xong trong một tuần. Với mốc tham chiếu V1, riêng train chính có thể đã chiếm nhiều ngày đến nhiều tuần khi có quota. Nếu hạn nộp ngắn hơn dự toán, dùng phương án giảm phạm vi tại mục8.1 và ghi rõ mức bằng chứng đạt được.

## 16. Tổ chức artifact và sổ thí nghiệm dự kiến

```text
configs/                         # protocol/recipe khóa sau pilot
docs/PAPER_ALIGNMENT_AUDIT.md     # nguồn và khác biệt
docs/EXPERIMENT_LEDGER.md         # trạng thái, nguồn, checksum từng run
data/splits/                     # indices DEV/FINAL; không tự commit data lớn
outputs/<protocol>/<method>/seed_<seed>/
    config.json
    manifest.json
    metrics.json
    history.jsonl
    checkpoint_latest.pt
    checkpoint_best.pt           # DEV, nếu dùng best-validation
    checkpoint_epoch100.pt       # FINAL
    predictions.jsonl
reports/<protocol>/              # bảng/figures xuất tự động
artifacts/demo/<version>/        # weights, bank, ảnh, manifest và class names
docs/DEMO_GUIDE.md
docs/DEFENSE_SCRIPT.md
```

Đây là cấu trúc mục tiêu cần xây, không phải danh sách file đã tồn tại. Manifest tối thiểu có protocol ID, method definition version, source commit, dependency versions, pretrained hash, checkpoint hash, split/pool/transform hashes, all seeds, hardware/dtype, schedule, approximation mode, epoch và checkpoint-selection rule.

Sổ run dùng các trạng thái: `planned`, `preflight_passed`, `running`, `interrupted_resumable`, `trained`, `evaluated`, `verified`, `invalid`. Chỉ `verified` được đưa vào bảng cuối. Đủ file `.pt` nhưng thiếu metadata/evaluation chưa phải `verified`.

## 17. Rủi ro và quyết định tương ứng

| Tình huống | Việc làm | Không suy diễn |
|---|---|---|
| Không tìm được code/config tác giả | Khóa giả định, cung cấp audit và ablation khả thi | Không tuyên bố exact100% |
| Full reference không vừa GPU | Profile/chunk/recompute; nếu cần approximation, ghi và kiểm chứng | Không đổi silently sang top-k rồi gọi Full |
| Acc thấp hơn paper | So baseline cùng protocol, xem curve/gradient/data; báo khoảng cách | Không train thêm vô hạn chỉ để đuổi89,2 |
| Validation giảm cuối run | Kiểm selector/bank/preprocess/checkpoint; phân biệt DEV-best/FINAL-last | Không dùng test chọn epoch |
| Two-Stage chậm hơn | Báo runtime breakdown, kết quả âm; chỉ mở biến thể nếu DEV có lý do | Không lấy N/K làm speedup thực |
| Cloud cancel | Resume latest đã xác minh, giữ schedule100 | Không đổi epochs theo mỗi session |
| Hết quota gần deadline | Thu hẹp theo mục8.1, giữ same-budget comparisons | Không giả số seed chưa chạy |
| Disk gần đầy | Kiểm từng run/version, backup, loại bản trùng có kiểm soát | Không xóa rộng data/output |
| Demo OOM/chậm | Load lười, CPU đo trước, video thực dự phòng | Không hiện mock prediction như live |
| Nhầm cũ/mới trong báo cáo | Dùng protocol ID và audit dữ liệu, cập nhật tài liệu lịch sử | Không so tiny5epoch với DINOv3-100epoch như cùng setup |

## 18. Checklist hoàn thành từ hiện tại đến buổi demo

### Thuật toán và dữ liệu

- [ ] P00: Artifact V1 và các run đang có được kiểm kê, sao lưu.
- [ ] P01: Audit paper/code hoàn thành; weight tying/pool overlap/feature definition được ghi đúng mức chắc chắn.
- [ ] P02: DEV/FINAL split và chống leakage được test.
- [ ] P03: Gradient reference/optimized được so sánh; approximation có tên rõ.
- [ ] P04: Batch64/advantage/AMP được kiểm chứng.
- [ ] P05: Đủ bốn baseline paper + TACS; Frozen DINO không bị nhầm fine-tuned encoder.
- [ ] P06: Resume giữ schedule, RNG và provenance; notebook chạy theo đoạn.
- [ ] P07: Smoke ảnh thật và artifact verification đạt.

### Thực nghiệm

- [ ] P08: Pilot và DEV hội tụ xong; khóa recipe, budget, K-selection rule.
- [ ] P09: Main runs đủ phạm vi đã đăng ký; mọi thiếu seed có nhãn rõ.
- [ ] P10: Two-Stage đã train/eval; có kiểm tra speedup thật, không chỉ số candidate.
- [ ] P11: Ablation có control và budget nhất quán.
- [ ] P12: Test, raw predictions, mean±SD, paired delta có thể tái tạo.
- [ ] P13: Benchmark batch1/throughput/cold-start/memory trên cùng máy; phân tích lỗi.

### Demo và bảo vệ

- [ ] P14: Inference API khớp evaluator; bộ artifact đủ và đúng hash.
- [ ] P15: UI chạy dữ liệu/model thật; upload/score/K có giải thích chính xác.
- [ ] P16: Máy bảo vệ chạy offline;20 lượt thao tác ổn; video dự phòng.
- [ ] P17: Báo cáo/figures thống nhất số liệu và công khai sai khác paper.
- [ ] P18: Slide/kịch bản/diễn tập xong; mỗi thành viên trả lời được phương pháp và đóng góp.
- [ ] Mỗi nhiệm vụ có commit mô tả, checkpoint nằm ngoài Git; không push.

**Việc đầu tiên sau bản kế hoạch:** hoàn tất P00–P01, chốt đặc tả P02/P03/P04 rồi mới triển khai đợt training mới. Đồng thời khởi động P14 từ checkpoint V1 để có demo nền sớm. Không khởi chạy18 run dài ngay từ bản code hiện tại.

## 19. Nguồn và thời điểm kiểm tra

- PDF gốc trong thư mục dự án: `18. Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.pdf`; đã đối chiếu Sec.3–4, Fig.2 và Table2; kiểm tra trực quan Table2, không lấy số từ bảng dịch.
- [Trang CVPR2026 chính thức](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.html): metadata hội nghị và liên kết paper; tìm kiếm ngày30/10/2026 trả metadata, mở trực tiếp có lỗi403.
- [Preprint arXiv:2512.00489](https://arxiv.org/abs/2512.00489): nguồn đối chiếu phiên bản, truy cập30/10/2026. Bản CVPR2026 là tài liệu mục tiêu; preprint nộp năm2025 không đổi năm của bản hội nghị.
- Source đã xem: `src/tacs/{data,model,losses,train,retrieval,multiseed}.py`, `scripts/analyze_all_models.py`, README, pyproject; không dùng knowledge graph `.ua`.
- Artifact đã xem: manifest V1, config và100 records metrics của CIFAR-100/DINOv3/TACS seed42; thời gian trong mục10 tính từ tổng các record train/validation.

Các con số “mục tiêu”, “dự kiến”, “đề xuất”, thời lượng và criteria trong kế hoạch là quy ước của nhóm để triển khai; chỉ Table2 và các chi tiết có nguồn tại mục3 được gán cho tác giả.
