# Caltech-101 bổ sung cho demo ảnh rõ

Ngày14/11/2026. Phạm vi theo lời làm rõ của người dùng: **giữ CIFAR-100 + thêm
Caltech-101**, ưu tiên demo, không phải bỏ dataset mới. Oxford Pets không phải
bộ đã chốt. ADR-006 thay cho cách hiểu sai trong ADR-005.

## Phần đã có và phần chưa có

- CIFAR-100: giữ nguyên checkpoint, audit, source thí nghiệm và GUI showcase.
- Caltech-101: **chưa** tải/kiểm kê dữ liệu trong nhiệm vụ điều chỉnh phạm vi;
  chưa có loader tích hợp, split, training checkpoint hoặc demo hợp lệ.
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
  category labels. Mã train project hiện vẫn khóa CIFAR-10/100; không chỉ đổi
  `--dataset caltech101` là chạy được.

## Bước cần chốt

Nền tảng train Caltech-101: notebook Kaggle, notebook Colab hay máy local.
Đây là lựa chọn môi trường/ngân sách; không thay dataset đã chốt. Chưa khởi
chạy GPU dài hoặc mua tài nguyên trong nhiệm vụ cập nhật phạm vi này.

Hai trường hợp biên bắt buộc: ảnh grayscale/nhãn nền phải xử lý đúng; bất kỳ
ảnh validation/test nào lọt vào candidate pool phải khiến preflight từ chối.
