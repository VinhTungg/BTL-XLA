# Caltech-101 bổ sung cho demo ảnh rõ

Ngày14/11/2026. Phạm vi theo lời làm rõ của người dùng: **giữ CIFAR-100 + thêm
Caltech-101**, ưu tiên demo, không phải bỏ dataset mới. Oxford Pets không phải
bộ đã chốt. ADR-006 thay cho cách hiểu sai trong ADR-005.

## Phần đã có và phần chưa có

- CIFAR-100: giữ nguyên checkpoint, audit, source thí nghiệm và GUI showcase.
- Caltech-101: đã tải và kiểm kê **8.677 ảnh/101 lớp**, có loader RGB, split
  không trùng vai trò, trainer riêng và notebook Colab đã thiết lập/saved.
  Chưa train dài, chưa có checkpoint đã huấn luyện hoặc GUI Caltech hợp lệ.
  Hướng dẫn vận hành: [CALTECH101_COLAB_GUIDE.md](CALTECH101_COLAB_GUIDE.md).
- Không dùng ảnh Caltech chạy qua đầu phân loại CIFAR-100 rồi tuyên bố đã
  hỗ trợ dataset mới. Hiển thị ảnh nét không tự làm model nhận diện101lớp mới.

## Trình tự triển khai

1. **Dữ liệu:** thêm loader RGB cho Caltech-101, class map101lớp không có
   `BACKGROUND_Google`, giữ ảnh nguồn để preview; ảnh model vẫn được tiền xử lý
   nhất quán. Tạo split theo lớp với seed cố định, query/pool/validation/test
   tách biệt; ghi manifest, class counts, tên file và hash. Chốt protocol trước
   train; nếu dùng split phục vụ demo riêng phải ghi rõ, không tự gọi là đúng
   protocol paper hoặc so trực tiếp mAcc paper.
2. **Huấn luyện riêng:** đầu ra101lớp, DINOv3 pretrained, output/cache/resume
   tách khỏi CIFAR-100. Bắt đầu bằng No-context để kiểm tra pipeline; nếu dùng
   Two-Stage thì reference encoder phải thuộc thí nghiệm Caltech tương ứng,
   sau đó train model có context cần cho demo. Đây là số run phục vụ sản phẩm,
   không phải bộ baseline đầy đủ của bài báo. Không chuyển trọng số task/selector
   CIFAR hoặc resume epoch cũ mà không khai báo thành transfer-learning riêng.
3. **Smoke trước chạy dài:** nhãn/shape/gradient hữu hạn, không lẫn split,
   lưu/nạp/resume đúng experiment; chọn ngân sách và scheduler trước lượt train.
   Chuẩn bị notebook cloud theo nền tảng người dùng chọn; checkpoint lưu bền vững.
   Không tự thuê GPU hoặc dùng tài khoản/cloud khác.
4. **GUI:** thêm lựa chọn Caltech-101, ảnh validation hoặc file riêng, ảnh context
   thật và top-5 đúng class map101lớp. Preview dùng ảnh nguồn, không phóng từ32×32.
   Các cách chọn context vẫn được ghi là can thiệp trên cùng task weights.
5. **Nghiệm thu:** kiểm tra checkpoint/class map/split, inference ảnh thật,
   ví dụ đúng/sai, đổi dataset không giữ cache hoặc nhãn cũ, xuất JSON có dataset/
   protocol/input/checkpoint hash. Giữ trường hợp thất bại và không hứa accuracy.

Không train lại CIFAR-100, không quét lại các weight hoặc normalization chỉ để
có giao diện Caltech. Đổi dataset là mở rộng demo/kiểm chứng, không tự tính là
cải tiến thuật toán thành công. Không chọn model bằng tập test.

## Nguồn đã kiểm tra

- [CaltechDATA chính thức](https://data.caltech.edu/records/mzrjq-6wc02):101loại
  đối tượng, ảnh khoảng300×200; kích thước thực tế thay đổi, không đảm bảo mọi
  ảnh đều sắc nét. Gói chính thức caltech-101.zip137,4MB,
  MD5`3138e1922a9193bfa496528edbbc45d0` theo trang dữ liệu.
- [Torchvision Caltech101](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Caltech101.html):
  loader category/annotation, không có tham số official train/test split trong API.
  Cần tự xác định và lưu protocol chia tập, không mặc định dữ liệu tự có split.
- Source torchvision cài trên máy đã kiểm tra: loại `BACKGROUND_Google` khỏi
  category labels. Trainer CIFAR cũ vẫn khóa CIFAR-10/100; Caltech dùng module
  riêng `tacs.caltech.run`, không chỉ đổi `--dataset` trong trainer cũ.

## Cấu hình đã chốt cho lượt demo

- Người dùng chọn **Colab**, setup trên máy hiện tại rồi tự mở/chạy trên máy khác.
  Đã tạo notebook nhúng source; không cần upload dataset/source ZIP.
- Split demo seed42: mỗi lớp19 query,6 pool,5 validation, phần còn lại test.
  Tổng1919/606/505/5647; kiểm tra hash ảnh không cắt ngang vai trò. Seed train44.
  Đây là protocol demo tự xác định, **không phải protocol đánh giá paper**.
- No-context và Two-Stage tách output; mỗi lượt20 epoch, DINOv3 ViT-S/16
  pretrained, task head101 lớp, role_v2, input256×256, batch4/accumulation16,
  AMP, AdamW1e-4, warmup5, cosine20, weight decay1e-4, temperature0.1,
  policy weight1.0, top-k32. Không tự quét hyperparameter.
- Reference retrieval là **frozen pretrained DINOv3**, xuất kèm No-context,
  không phải task encoder đã fine-tune. File được hash trước Two-Stage.
- Đã qua154 test (2 integration opt-in skipped) và forward/backward trên ảnh
  Caltech thật cho cả hai mô hình, logits[1,101], gradient hữu hạn; **không có
  optimizer step**, không dùng smoke để báo accuracy.
- Source thí nghiệm CIFAR top-level không sửa; bundle đối chứng cũ vẫn được
  kiểm thử hash. Checkpoint CIFAR không dùng để khởi tạo/resume Caltech.

Bước tiếp: người dùng chạy notebook, thu nguyên output Drive, kiểm tra checkpoint
và class map rồi mới tích hợp lựa chọn Caltech vào GUI. Không khẳng định demo
Caltech hoàn tất khi mới chỉ chuẩn bị notebook.

Hai trường hợp biên bắt buộc: ảnh grayscale/nhãn nền phải xử lý đúng; bất kỳ
ảnh validation/test nào lọt vào candidate pool phải khiến preflight từ chối.
