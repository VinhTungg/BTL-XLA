# Kiểm tra reward và gradient của selector

## Phạm vi

`python -m tacs.gradient_audit` là probe forward/backward chỉ đọc trên checkpoint
role_v2, không phải huấn luyện tiếp. Không tạo optimizer, không gọi optimizer
step, không đánh giá test, không ghi đè checkpoint/output. Dùng ảnh validation
CIFAR-100 thật và evaluation transforms để giữ input cố định; bật train mode
để selector lấy mẫu Gumbel như khi train, thay vì argmax của audit accuracy.

Mặc định 64 query lấy ngẫu nhiên không lặp bằng seed 44, 16 microbatch × 4 ảnh,
khớp đúng một cửa sổ gradient accumulation của pilot. Cả TACS và Two-Stage dùng
cùng query IDs. Candidate bank gồm toàn pool 9.000, detach và giữ cố định như
trainer trong một epoch; Two-Stage lọc cùng top-32 bằng encoder cố định của
No-context cùng pilot. Không dùng nhãn candidate để chọn context.

Checkpoint/config/metrics/split được kiểm tra bằng logic provenance của
selector audit. So SHA256 file checkpoint và toàn bộ state_dict trong RAM
trước/sau; chỉ đánh dấu hoàn tất nếu không thay đổi. Các batch kết quả ghi dần
vào `batches.jsonl`; chỉ coi thành công khi `report.json` có `status=complete`.

## Những phép đo

1. Reward từng query = CE không context − CE có context; reward dương nghĩa
   context làm giảm lỗi. Ghi advantage đã chuẩn hóa theo đúng loss hiện tại.
2. Đạo hàm task loss và policy loss theo **cùng các tham số selector**; norm,
   tỷ lệ độ lớn và cosine giữa hai vector gradient. Gradient đối hướng có
   cosine âm, nhưng đây không tự động là bug hoặc bằng chứng học sai.
3. Đạo hàm theo selector scores để kiểm tra policy gradient khớp công thức
   `advantage * (softmax(scores) - chosen_one_hot) / microbatch_size`.
4. Trung bình gradient của 16 microbatch, đúng hệ số tích lũy 1/16 nhưng
   không thực hiện update. Đo cả task, policy và tổng có `policy_weight`.
5. So advantage chuẩn hóa riêng mỗi microbatch với phép chuẩn hóa giả định
   trên cả 64 reward. Chỉ so số liệu reward/advantage; **không** chạy lại
   gradient của một loss global-batch khác để tuyên bố cách nào tốt hơn.

## Vì sao batch 4 × accumulation 16 không tương đương batch 64 ở policy loss?

`TACSLoss.forward` tính mean/std reward trong batch đầu vào. Trainer gọi loss
trên 4 ảnh rồi chia loss cho 16 trước backward. Vì vậy chuẩn hóa advantage vẫn
trên 4 ảnh; cộng gradient sau đó không tính lại mean/std trên 64 ảnh.
Đây là tính chất xác định từ source, **không phải lỗi đã được chứng minh so với
paper**. Chuẩn hóa tương đối còn có thể tạo advantage dương cho reward âm nếu
reward đó tốt hơn các ảnh khác trong microbatch; cần phân biệt reward với
advantage khi diễn giải.

## Kiểm soát bộ nhớ và precision

Task parameters được đặt `requires_grad=False` trong bản model chỉ dùng để
probe. Giá trị trọng số, forward function và train mode không đổi; gradient
vẫn đi qua task network tới zero-value proxy rồi tới selector. Không đo
gradient tham số task network trong công cụ này. Unit test đối chiếu selector
gradient trước/sau khi cố định task parameters để kiểm tra đường đạo hàm.

CUDA AMP theo config, fixed diagnostic loss scale mặc định 128. Gradient được
chia lại cho scale trước khi đo. Đây **không phải GradScaler state lịch sử**
của run train. Nếu gradient không hữu hạn, công cụ dừng rõ lỗi, không ghi kết
quả complete; dùng output mới và scale nhỏ hơn hoặc `--no-amp` để kiểm tra.
Gradient bằng 0 phải phân biệt với tensor không nối vào graph (`None`) và với
giá trị bị underflow/saturation. Một cửa sổ Gumbel không đại diện toàn quá trình
train; không kết luận gradient hữu ích chỉ vì khác 0 hoặc cùng hướng.

## Cách chạy

```powershell
$pilotRoot = 'artifacts/role_v2/kaggle_v2/TACS_selector_role_v2_pilot/cifar100/dinov3_vits16/role_v2/seed_44'
.\.venv\Scripts\python.exe -X utf8 -m tacs.gradient_audit --checkpoint "$pilotRoot/tacs/checkpoint_best.pt" --output artifacts/role_v2/gradient_tacs_recheck --device cuda
.\.venv\Scripts\python.exe -X utf8 -m tacs.gradient_audit --checkpoint "$pilotRoot/two_stage/checkpoint_best.pt" --standard-encoder "$pilotRoot/no_context/best_encoder.pt" --output artifacts/role_v2/gradient_two_stage_recheck --device cuda
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_gradient_audit.py -v
```

Thư mục output phải mới. Chạy hai probe tuần tự trên GPU 4 GB. Không giảm
microbatch xuống 1 rồi coi đó là phép đo tương đương: reward standard deviation
khi đó bằng 0 và advantage chuẩn hóa bằng 0, làm policy term mất tác dụng.

## Trạng thái thực nghiệm

Ngày 07/11/2026: hai probe chính đã hoàn tất trên cùng 64 query validation duy
nhất, cùng thứ tự, audit seed 44. 9 tests mới và 20 tests selector audit đều
đạt. Các probe không cập nhật model trong RAM hoặc checkpoint trên đĩa: SHA256
state trước/sau và file checkpoint đều không đổi. Không có optimizer step.

### Kết quả chính, diagnostic loss scale 128

| Phép đo | TACS | Two-Stage |
|---|---:|---:|
| Microbatch có gradient task khác 0 | 16/16 | 16/16 |
| Microbatch có gradient policy khác 0 | 16/16 | 16/16 |
| L2 gradient task sau tích lũy | 0,056616 | 0,010021 |
| L2 gradient policy sau tích lũy | 1,790876 | 0,441659 |
| Tỷ lệ policy/task (policy_weight=1) | **31,63×** | **44,07×** |
| Cosine task/policy sau tích lũy | −0,1090 | −0,5894 |
| Microbatch có hai gradient đối hướng | 10/16 | 5/16 |
| Mean reward | +0,013521 | +0,007288 |
| Reward standard deviation | 0,077519 | 0,117124 |
| Query có reward dương | 33/64 | 34/64 |
| Advantage đổi dấu giữa chuẩn hóa batch 4 và window 64 | 25/64 | 25/64 |

Cả hai nhánh nối vào 162 parameter tensors của selector và gradient đều hữu
hạn. Không có bằng chứng đường gradient bị đứt trong các probe này. Gradient
task ở đây đi qua **zero-value embedding proxy của bản triển khai hiện tại**;
việc nó khác 0 không chứng minh estimator này tương đương đạo hàm chọn ảnh
thật hoặc hoàn toàn khớp thuật toán paper.

Policy có độ lớn lớn hơn task sau tích lũy, nhưng không tự động là bug hoặc
chứng minh policy loss có hại trong toàn bộ training. Hai nhánh tối ưu những
thành phần khác nhau; cần thí nghiệm đối chứng trước khi thay trọng số loss.
Không gọi 25/64 advantage đổi dấu là "25 ảnh học sai": đây là ảnh hưởng của
baseline tương đối khi thay nhóm chuẩn hóa.

TACS có 8 query reward âm nhưng microbatch advantage dương, và 11 query reward
dương nhưng advantage âm; Two-Stage tương ứng 9 và 8 query. Đây là hành vi có
thể xảy ra của chuẩn hóa theo nhóm, không phải lỗi dấu trong implementation.
Unit test FP32 xác minh công thức policy score gradient; sai lệch tuyệt đối
lớn nhất trong probe AMP khoảng 1,16e-4, không tuyên bố khớp bit tuyệt đối.

Gradient task dao động rất mạnh giữa microbatch. TACS min/median/max L2 là
2,35e-6 / 0,003135 / 0,894962; Two-Stage là 4,40e-8 / 0,000862 / 0,209892.
Microbatch lớn nhất chiếm 87,74% (TACS) và 70,41% (Two-Stage) tổng các L2 norm
microbatch. Đây không phải phần trăm đóng góp vector vào gradient cuối vì
gradient còn có hướng và có thể triệt tiêu. Một window không đủ để khẳng định
các tỷ lệ trên ổn định qua nhiều batch/epoch.

Artifact chính:

- `artifacts/role_v2/gradient_tacs_window44/report.json`, SHA256
  `7c61c33d3991fbe230f19c7e8f5774bd5c028578e74a74a21df2a23e809f9d54`.
- `artifacts/role_v2/gradient_two_stage_window44/report.json`, SHA256
  `758a48bd5b1ae62a8219ebfb66bb89c8b14bf58953ce8b4cb8697893c90a4990`.
- Mỗi thư mục có `manifest.json` và `batches.jsonl` (16 dòng), chứa ID query,
  selected context, reward/advantage và gradient statistics từng microbatch.

### Đối chứng loss scale 1024

Đã lặp lại đúng cùng query IDs và Gumbel seeds, chỉ đổi diagnostic loss scale
128 → 1024. Context được chọn, CE, reward và advantage của từng query khớp
hoàn toàn giữa hai scale cho cả hai mô hình. Cả bốn probe đều finite, hai nhánh
gradient khác 0 ở mọi microbatch, state/checkpoint không đổi, optimizer step 0.

| Sau tích lũy 16 microbatch | TACS scale 128 | TACS scale 1024 | Two-Stage scale 128 | Two-Stage scale 1024 |
|---|---:|---:|---:|---:|
| Policy/task L2 | 31,63× | 32,07× | 44,07× | 43,96× |
| Cosine task/policy | −0,1090 | −0,1113 | −0,5894 | −0,5813 |
| Microbatch đối hướng | 10/16 | 10/16 | 5/16 | 6/16 |

Nhận xét policy lớn hơn task sau tích lũy giữ ở cả hai scale trong window này.
Tuy nhiên một số gradient task nhỏ có sai khác tương đối đáng kể do precision;
đặc biệt số batch đối hướng của Two-Stage thay đổi 5 → 6. Không tuyên bố các
gradient nhỏ hoặc dấu từng batch hoàn toàn không nhạy AMP. Hai scale vẫn dùng
cùng AMP features/forward; đây chưa phải đối chứng re-encode/backward FP32.
Các phép lặp dùng cùng 64 query, không phải thêm 64 mẫu độc lập.

Artifact đối chứng:

- `artifacts/role_v2/gradient_tacs_window44_scale1024/report.json`, SHA256
  `9dbd25d822e1bec4003e8144005f15dc8c034d340cb1ffefa5b537313ac95dac`.
- `artifacts/role_v2/gradient_two_stage_window44_scale1024/report.json`, SHA256
  `5e77b42dcdbc8ace33572c48b7803d20d8f74ddd843f0bc345d901b89d3cac74`.

### Lặp trên audit seed 45 và 46 — hoàn tất 07/11/2026

Đã chạy thêm bốn probe ở scale 128, giữ nguyên hai checkpoint **training seed
44** và toàn bộ protocol. Đây là ba cửa sổ validation trên mỗi checkpoint,
**không phải ba lần huấn luyện với seed khác nhau**. Mỗi cửa sổ gồm 64 query;
TACS và Two-Stage có cùng ID và thứ tự query trong từng cửa sổ.

Ba cửa sổ có tổng cộng **190 query ID duy nhất**: 44/45 trùng ID 7525,
45/46 trùng ID 33497, 44/46 không trùng. Gumbel seeds cũng có phần chồng lấn
do công thức `audit_seed + 1000 + batch_index`: các dải 1044–1059,
1045–1060, 1046–1061. Không coi đây là các lần lặp độc lập để kiểm định
thống kê hoặc thay thế kết quả nhiều training seed.

| Checkpoint | Audit seed | Task L2 | Policy L2 | Policy/task | Cosine task/policy | Batch đối hướng | Advantage đổi dấu |
|---|---:|---:|---:|---:|---:|---:|---:|
| TACS | 44 | 0,056616 | 1,790876 | 31,63× | −0,1090 | 10/16 | 25/64 |
| TACS | 45 | 0,004801 | 1,810359 | 377,11× | −0,4836 | 9/16 | 21/64 |
| TACS | 46 | 0,071898 | 2,061469 | 28,67× | −0,1586 | 8/16 | 22/64 |
| Two-Stage | 44 | 0,010021 | 0,441659 | 44,07× | −0,5894 | 5/16 | 25/64 |
| Two-Stage | 45 | 0,000352 | 0,601661 | 1709,35× | +0,1870 | 6/16 | 20/64 |
| Two-Stage | 46 | 0,003050 | 0,601243 | 197,14× | +0,1139 | 7/16 | 20/64 |

L2 và cosine trong bảng đo **sau tích lũy 16 microbatch**, policy_weight=1.
Advantage đổi dấu so chuẩn hóa theo microbatch 4 với chuẩn hóa giả định theo
window 64; không phải gradient của một lần train bằng global batch 64.

Quan sát policy norm lớn hơn task norm giữ ở cả sáu probe, nhưng tỷ lệ biến
động mạnh. Đặc biệt tỷ lệ 1709,35× đi kèm task norm rất nhỏ (0,000352),
không đủ để gọi là nổ gradient. TACS có cosine âm ở cả ba cửa sổ; Two-Stage
có cosine dương ở hai cửa sổ mới. Vì vậy **không kết luận hai nhánh luôn
đối hướng**, cũng không suy ra giảm policy weight chắc chắn tăng accuracy.

Mean reward TACS theo cửa sổ 44/45/46 lần lượt là +0,013521 / −0,004748 /
+0,003619; Two-Stage là +0,007288 / +0,008484 / +0,012842. Đây là reward
trên mẫu Gumbel trong probe, không phải accuracy mới hay đánh giá toàn validation.

Đã kiểm tra sáu report `complete`, 96 microbatch, gradient encoder và score
hữu hạn, cả hai nhánh encoder khác 0 ở mọi microbatch. Checkpoint SHA256
hiện tại khớp report, state trước/sau không đổi, optimizer step bằng 0,
không đánh giá test. Chín unit tests gradient audit được chạy lại và đạt.
Đối chứng scale 1024 chỉ áp dụng cửa sổ 44; chưa lặp precision control cho
45/46 và chưa kiểm tra toàn pipeline FP32.

Artifact bổ sung (mỗi thư mục có manifest, 16 dòng batch và report):

| Thư mục dưới `artifacts/role_v2/` | SHA256 của `report.json` |
|---|---|
| `gradient_tacs_window45` | `a17e9f955cb32727a9fa771eea1a5173e17bfa0183db66033b4aa51486087fe0` |
| `gradient_two_stage_window45` | `00e19f12684484f2c6ed006be8799d94033d6210b1b2a680320f01ca2ab355b5` |
| `gradient_tacs_window46` | `b706bbb1b44b30dc5eef9a2987059c085553b71e2484c67e4107e4890928027d` |
| `gradient_two_stage_window46` | `d936339ad00735a78944b145eeaba43d72fa71e8dd9e120a99020b7bb72e43b6` |

Tái chạy bằng lệnh ở phần Cách chạy, thêm `--seed 45` hoặc `--seed 46`
và `--loss-scale 128`, luôn chọn thư mục output mới. Artifact lớn nằm ngoài
Git; bảng hash này giữ dấu vết kết quả, không thay thế việc sao lưu artifact.

### Kết luận và nhiệm vụ kế tiếp

Chưa tìm thấy bằng chứng nhánh selector bị đứt gradient. Đã xác định từ source
rằng advantage được chuẩn hóa theo microbatch 4, không phải toàn window 64.
Đã đo được policy gradient lớn hơn task gradient nhiều lần ở cả ba cửa sổ
trên mỗi checkpoint, nhưng hướng tương đối không ổn định ở Two-Stage.
**Chưa chứng minh các yếu tố này là nguyên nhân duy nhất
của accuracy hoặc rằng thay loss chắc chắn giúp tăng accuracy.**

Thứ tự tiếp theo:

1. **Đã hoàn tất** lặp probe trên các window audit seed 45, 46, checkpoint
   training seed vẫn 44; kết quả và giới hạn ở bảng trên.
2. Tiếp theo, thiết kế ablation độc lập cho `policy_weight` 1,0 / 0,1 /
   0,03; các giá trị nhỏ chỉ là ứng viên cần train/validation, chưa là cấu hình
   thắng. Không coi tỷ lệ gradient trên một window là công thức tìm lambda tối ưu.
   Giữ cùng split, khởi tạo, số epoch, batch/accumulation, augmentation và
   encoder truy xuất; output riêng từng cấu hình. Không so một run train mới
   với checkpoint cũ như thể chỉ khác policy weight nếu ngân sách/khởi tạo
   khác nhau. Chọn theo validation, lưu cả accuracy lẫn paired context utility.
3. Tách riêng ablation chuẩn hóa advantage theo window 64 so với microbatch 4;
   cần giữ replay input/Gumbel nhất quán nếu tính lại theo hai lượt để tiết kiệm
   VRAM. Không thay đồng thời loss weight, cosine score và normalization rồi
   gán mọi cải thiện cho một thay đổi.
4. Giữ nguyên bản tái hiện gốc, giữ frozen-cosine top-1 làm đối chứng; chỉ khóa
   cải tiến sau pilot và nhiều seed. Không đánh giá test khi đang chọn phương án.

Chưa thực hiện các bước train/ablation này. Mã được thêm chỉ phục vụ audit;
`losses.py`, `model.py`, `train.py` không bị sửa trong nhiệm vụ này.
