# Caltech-101 Colab: lịch train 100 epoch, người dùng tự chạy

Ngày 14/11/2026. Không thay đổi checkpoint/source thí nghiệm CIFAR-100.
Đây là nhánh demo ảnh nguồn lớn hơn CIFAR; không phải tái hiện đầy đủ paper,
không đảm bảo đạt một mức accuracy và không dùng tập test để chọn model.

**Cập nhật16/11/2026:** cả hai run100epoch đã tải về và được audit; không cần
train lại để mở demo. Xem [kết quả](CALTECH101_FULL100_RESULTS.md) và
[hướng dẫn demo](CALTECH101_DEMO_GUIDE.md). Các bước dưới giữ làm hướng dẫn
tái lập, không phải yêu cầu khởi chạy lại một lượt cloud.

## Notebook đã lưu

- [Mở notebook trên Colab](https://colab.research.google.com/drive/1Uxt2g6IgY-IKdIMIZtlC5i3ZTiCYS1pB).
- Tên `TACS_Caltech101_Full100_seed44.ipynb`.
- Bản local: `artifacts/caltech_demo/TACS_Caltech101_Full100_seed44.ipynb`.
- Đã sửa cấu hình ban đầu 20 epoch do agent tự đặt; người dùng yêu cầu train đầy
  đủ. Lịch hiện hành là **100 epoch mỗi mô hình**, tổng 200 epoch cho hai mô hình.
  Bản notebook 20 epoch local được giữ để truy vết, không phải bản nên chạy.
- Builder tái tạo: `scripts/build_caltech_colab.py`. Script từ chối ghi đè notebook
  đã có. Source được nhúng base64 với SHA256; không upload ZIP/dataset thủ công.
- Đã kiểm tra UI có T4, đủ 7 code cells, mỗi cell paste khớp bản local và trạng thái
  All changes saved. **Chưa chạy cell TRAIN**, chưa chạy full setup Colab thành công.
  Preflight GPU trước đó nhận T4, nhưng mount Drive bị lỗi; phải cấp quyền lại khi chạy.

## Cách tự chạy trên máy hiện tại hoặc máy khác

1. Đăng nhập **cùng tài khoản Google đang sở hữu notebook và Drive output** trên
   máy muốn treo; mở link ở trên. Không tạo bản sao hoặc chạy hai runtime ghi chung output.
2. Kiểm tra Runtime → Change runtime type → Python3/GPU T4. Nếu runtime hiện có
   vẫn hoạt động, chỉ mở notebook, không bấm Disconnect and delete runtime.
3. Chạy 7 cell **từ trên xuống**, có thể bấm Run all và xử lý các bước cấp quyền
   khi Colab hỏi. Cell 1 kiểm tra GPU/cài thư viện; cell 2 yêu cầu kết nối Drive;
   chọn đúng tài khoản và tự cấp quyền. Nếu CAPTCHA xuất hiện, tự hoàn tất.
4. Chờ cell 3 cài source và cell 4 tải dữ liệu thật, kiểm tra 101 lớp/8.677 ảnh cùng
   số lượng 1919/606/505/5647. Không chạy tiếp khi một cell trước đó báo lỗi.
5. Cell 5 khóa `EPOCHS=100`, `BASELINE='both'`, `STOP_AFTER=None`. Cell 6 có nhãn
   **TRAIN** mới bắt đầu huấn luyện. Thứ tự No-context → Two-Stage tự động;
   cell 7 chỉ xác minh, không train thêm và không đánh giá test.

GPU nằm trên server Colab, không chạy trên GPU của máy đang mở trình duyệt.
Mở cùng notebook trên máy khác không có nghĩa là sao chép tiến trình sang GPU
khác. Không có cách setup nào bảo đảm Colab không ngắt phiên/hết quota; không
dùng script giả tương tác/keep-alive để lách giới hạn. Không cần UltraViewer
để huấn luyện sau khi đã mở notebook bằng đúng tài khoản.

## Những gì sẽ chạy

| Hạng mục | Cấu hình |
|---|---|
| Dataset | Caltech-101 thật, bỏ BACKGROUND_Google, chuyển grayscale thành RGB |
| Protocol | `caltech101_demo_19query_6pool_5val_rest_test_v1` |
| Split seed / train seed | 42 / 44 |
| Query train / candidate pool / validation / test | 1919 / 606 / 505 / 5647 |
| Hai lượt riêng | No-context 100 epoch, rồi Two-Stage 100 epoch |
| Backbone / task network | DINOv3 ViT-S/16 pretrained, role_v2, 101 logits |
| Input / batch / accumulation | 256×256 / 4 / 16, AMP |
| Optimizer/schedule | AdamW, LR 1e-4, WD 1e-4, warmup 5, cosine 100 epoch |
| Context | frozen pretrained DINOv3 top-32 → learned selector; temperature 0.1, policy weight 1.0 |
| Test | Không dùng để train/chọn checkpoint, chưa đánh giá |

Mỗi lớp có 30 ảnh phát triển: 19 query, 6 pool, 5 validation. Các ảnh còn lại thuộc
test; không giới hạn 50 ảnh test/lớp như một số protocol Caltech khác. Không so
trực tiếp điểm validation này với mAcc paper. Vì validation cân bằng 5 ảnh/lớp,
accuracy chung và mean-per-class accuracy của **validation này** bằng nhau.
Test không cân bằng nên không được suy rộng tính chất đó sang test.

`best_encoder.pt` của No-context lưu **frozen pretrained retriever**: nó không
phải task encoder đã fine-tune, dù tên file theo quy ước project cũ. Two-Stage
khởi tạo task/selector pretrained riêng; không transfer task weights CIFAR.

## Log, output và điều kiện hoàn tất

Output bền vững:

```text
MyDrive/TACS_Caltech101_Demo/seed44_full100_v1/
  inventory.json
  split.json
  train.log
  no_context/
    experiment_manifest.json
    checkpoint_latest.pt
    checkpoint_best.pt
    best_encoder.pt
    metrics.json
    result.json
  two_stage/
    experiment_manifest.json
    checkpoint_latest.pt
    checkpoint_best.pt
    metrics.json
    result.json
```

Log hiển thị trực tiếp và nối thêm vào `train.log`: bắt đầu epoch/số batch,
bắt đầu validation, train/val accuracy, thời gian và saved latest. Hiện chưa có
log từng batch; giai đoạn chuẩn bị top32 hoặc một epoch vẫn có thể im lặng một
khoảng thời gian. Không nhầm điều này với đã hoàn tất hoặc tự bấm TRAIN lần nữa.

Một run chỉ được đánh dấu complete khi đủ 100 epoch và có `result.json`. Cell 7
kiểm tra số epoch, hash best checkpoint/metrics và hash reference encoder.
Chỉ thấy checkpoint không chứng minh cả hai run đã hoàn tất.

## Nếu bị ngắt giữa chừng

Mở lại **chính notebook**, cấp T4 và mount đúng Drive, chạy lại từ cell 1.
Giữ nguyên source, EPOCHS=100, output và config. Với `BASELINE='both'`, run đã
complete được kiểm tra rồi bỏ qua, run chưa xong tự khôi phục `checkpoint_latest.pt`
gồm model/optimizer/scheduler/AMP scaler/RNG và bắt đầu epoch kế tiếp đã lưu.
Ví dụ epoch 8 lưu xong thì tiếp từ 9; đang dở epoch 9 sẽ chạy lại epoch 9, không
khôi phục chính giữa batch. Drive phải ghi xong checkpoint để phục hồi được.

Không tăng EPOCHS để giả resume: scheduler đã thiết kế tổng 100 epoch và manifest
khóa config. Muốn dừng theo chặng, đặt `STOP_AFTER=25` trước khi chạy; sau đó
chuyển `STOP_AFTER=None` nhưng vẫn EPOCHS=100. Số này là epoch đích, không phải
số epoch cộng thêm. Sau khi No-context hoàn tất, Two-Stage mới được phép bắt đầu.
Output 100 epoch có tên riêng; không lấy checkpoint từ `seed44_demo_v1` rồi sửa
config/scheduler thành 100 và gọi là train 100 epoch từ đầu. Lịch dài vẫn không tự
biến protocol chia tập demo này thành protocol tái hiện đầy đủ bài báo.

Manifest sai, output không có nguồn gốc hoặc runtime đổi torch/torchvision/timm/
numpy/GPU sẽ bị từ chối để tránh trộn thí nghiệm. Giữ nguyên output và xem lỗi;
không xóa manifest hay tự ghi đè checkpoint để ép chạy. Nếu môi trường Colab
đã đổi, cần tái lập môi trường cũ hoặc thiết kế lượt mới có khai báo riêng.

## Sau khi chạy xong

Chờ cell 7 in COMPLETE cho **cả hai** run, rồi tải nguyên thư mục
`seed44_full100_v1` về `artifacts/caltech_demo/` (không đổi tên checkpoint thành
CIFAR). Bước tiếp theo mới là audit artifact và tích hợp Caltech-101 vào GUI,
preview ảnh native, top-5/nhãn/context/provenance đúng dataset. GUI CIFAR hiện
có không tự hiểu 101 lớp Caltech khi chỉ đặt file checkpoint vào cạnh nó.

## Kiểm thử đã thực hiện

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 scripts/verify_caltech_setup.py
```

- 155 test chạy, không lỗi, 2 test tích hợp opt-in skipped; 10 test Caltech kiểm tra split,
  RGB/background, leakage, manifest, resume fullstate và notebook nhúng source.
- Dữ liệu Caltech thật: 8.677 ảnh, 101 lớp; loader và split kiểm kê thành công.
- Real-data GPU smoke: No-context/Two-Stage logits [1,101], gradient/loss hữu hạn;
  **0 optimizer step**, không checkpoint train, không accuracy claim. Smoke không
  thay thế đánh giá lượt train 100 epoch trên T4 Colab.

Hai trường hợp biên: không có GPU hoặc Drive chưa mount phải dừng trước train;
config/split/reference khác lượt cũ phải từ chối resume, không âm thầm train lại.
