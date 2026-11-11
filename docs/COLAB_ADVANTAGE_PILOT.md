# Chạy effective-batch seed44 trên Colab

## Mục tiêu và định danh

Chỉ chạy nhánh effective_batch, khởi tạo task/selector mới, 20 epoch,
CIFAR-100 validation-only. Đối chứng microbatch đã hoàn thành: 87,92% validation.
Không dùng checkpoint task microbatch làm khởi tạo nhánh mới.

Notebook Colab nhúng đúng source bundle đã dùng ở đối chứng:
`1d85d0bbb17201c08e38b7428f720fccfe2859cc94cd0b256fac1103c7c9dce5`.
Không sửa thuật toán/recipe trong source bundle để chuyển nền tảng.

## File bàn giao

- `artifacts/effective_batch_pilot/Advantage_effective_batch_seed44_Colab.ipynb`.
- `artifacts/effective_batch_pilot/TACS_Advantage_Colab_Inputs.zip` (~80 MB):
  chỉ reference No-context encoder gốc và manifest đối chứng. Không chứa dataset,
  optimizer hoặc task/selector checkpoint của nhánh microbatch.

Builder từ archive đã xác minh:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/build_advantage_colab.py --archive 'C:\Users\ADMIN\Downloads\results (3).zip'
```

Builder từ chối ghi đè hai artifact đã tồn tại hoặc đóng gói source/encoder sai hash.
Không cần chạy builder lại khi đã có file.

## Cách chạy

1. Import notebook Colab. Chọn Python3, GPU T4; không cần mua gói trong workflow.
2. Chạy cell1 và tự chấp thuận mount Drive nếu Colab yêu cầu.
3. Nếu thiếu input, cell1 hiện nút upload: chọn đúng `TACS_Advantage_Colab_Inputs.zip`.
   File được giữ tại `MyDrive/TACS_Advantage_Pilot/` cho lần sau.
4. Cell2 ghim torch2.11.0+cu130, torchvision0.26.0+cu130 từ index PyTorch chính thức,
   timm1.0.30. Cài gói có thể cần tải nhiều GB; không coi cài xong là train xong.
   Nếu torch đã import với phiên bản khác, restart session rồi chạy lại từ cell1.
5. Cell3 giải nén source đã xác minh và cài package nội bộ; data/source nằm trên SSD
   runtime, không tải CIFAR-100 từng batch từ Drive.
6. Cell4 kiểm tra encoder/control environment, nối output vào Drive. Cell5 tải
   CIFAR-100 thật, xác nhận split36.000/9.000/5.000 và có CUDA.
7. Cell6 chạy train với log trực tiếp; cell7 kiểm tra completion và best hash.

Output bền vững:
`MyDrive/TACS_Advantage_Pilot/advantage_effective_batch_seed44/`.
Best/latest/metrics/log được ghi trực tiếp vào đó, không chỉ vào `/content`.
Checkpoint latest ghi sau mỗi epoch; phần epoch chưa hoàn thành có thể phải chạy lại
khi runtime mất. Không đảm bảo Colab giữ runtime đủ20epoch.

## Resume và đối chiếu

Chạy lại notebook với cùng Drive output: runner chỉ resume latest của chính
effective_batch khi manifest/source/environment/recipe khớp. Không tự reset/xóa
thư mục khi lỗi. Không chuyển seed hoặc scope trong thư mục đã có kết quả.

Giữ đường dẫn project Linux giống đối chứng (`/kaggle/working/advantage_source_...`)
ngay cả trên Colab. Output canonical là symlink tới Drive; runner resolve thành
đường dẫn Drive. Comparator sẵn có chỉ bỏ qua output_dir và scope, vẫn kiểm tra
mọi đường dẫn input/recipe/source/environment còn lại.

Notebook chủ động dừng nếu GPU không phải Tesla T4 hoặc environment không khớp
manifest đối chứng. Không sửa comparator để che môi trường khác. Nếu Colab không
chạy được CUDA13, cần tìm runtime phù hợp hoặc chạy lại cả hai nhánh trong môi
trường mới; không gọi một so sánh lệch environment là ablation kiểm soát.

Sau khi tải cả output, dùng `scripts/compare_advantage_pilot.py`, sau đó audit
selected/random/no-context trên validation; không chọn cải tiến bằng test.

## Kiểm tra đã làm

- 8 unit tests `test_advantage*.py` đạt, gồm AST tất cả cell, identity bundle,
  chặn source thay đổi, GPU/environment guard và các comparator tests đã có.
- Notebook được parse lại và so bằng object với kết quả builder, không chỉ nhìn tên.
- Bản input lấy encoder từ archive microbatch đã kiểm tra local, hash được builder
  đối chiếu trước đóng gói và notebook đối chiếu lại trước train.
- Live UI11/11/2026: tài khoản người dùng đã mở notebook trống; chọn T4,
  Colab yêu cầu CAPTCHA và người dùng đã hoàn tất. UI còn `Allocating runtime`
  tại lần quan sát cuối trước import. Chưa xác nhận GPU thật hay bắt đầu train.
- Upload tự động bị chặn do quyền file URL của tiện ích Brave; đã yêu cầu người
  dùng chọn notebook bằng Browse. Không thay đổi quyền extension.

Hai tình huống biên bắt buộc: input encoder sai phải bị từ chối; output cũ khác
manifest phải bị từ chối thay vì ghi đè/resume nhầm. Kiểm tra live setup còn phụ
thuộc import, mount Drive và cấp phát runtime của người dùng.

## Live setup sau import — 11/11/2026

Người dùng đã import thành công notebook:
https://colab.research.google.com/drive/1Joi1TEJ6cpStizwBVFyo61buS8BkQWk6

Đã xác nhận title, scope effective_batch/seed44 và 7 code cell đúng bộ bàn giao.
Chạy cell1 yêu cầu cấp runtime; người dùng hoàn tất CAPTCHA. Colab sau đó báo
`Too many sessions`. Manage sessions hiển thị một GPU session `Unknown notebook`.
Chưa terminate phiên đó vì chưa biết nội dung và dữ liệu tạm của nó; đã hỏi
người dùng xác nhận trước hành động có thể làm mất trạng thái runtime.

Trạng thái: notebook đã import, **chưa mount Drive, chưa cài môi trường, chưa
upload input, chưa xác minh CUDA thực tế và chưa có epoch train**. Cần giải phóng
phiên cũ an toàn rồi tiếp tục cell1. Không bấm Run all lặp khi runtime chưa có.
Bằng chứng: `artifacts/effective_batch_pilot/colab_active_session_handoff.jpg`.

### Kết nối runtime và Drive

Người dùng đồng ý terminate `Unknown notebook`, nhưng lần kiểm tra sau danh sách
đã báo `No active sessions`; agent không thực hiện terminate/xóa runtime nào.
Kết nối lại và người dùng hoàn tất CAPTCHA mới. Runtime đã kết nối, status T4
(Python3); người dùng hoàn tất cấp quyền Drive và output cell1 xác nhận
`Mounted at /content/drive`.

Resources hiển thị `You are not subscribed`, zero compute units, runtime có thể
kéo dài tới5giờ tại mức sử dụng lúc kiểm tra. Đây không phải đảm bảo đủ20epoch.
Chưa mua gói hay thay đổi quyền extension. Cell1 đang chờ người dùng chọn
`TACS_Advantage_Colab_Inputs.zip` qua nút Choose Files; đã yêu cầu chọn đúng file
~80MB. Cell2 đã được yêu cầu chạy sau cell1; chưa quan sát log cài đặt hoặc
`MATCHED ENVIRONMENT`. Chưa train. CUDA thực tế và tương thích cu130 còn phải
kiểm tra sau input. Bằng chứng: `artifacts/effective_batch_pilot/colab_waiting_input.jpg`.

## Setup lại trên tài khoản người dùng đang mở

Theo yêu cầu mới, chuyển sang tài khoản đang mở (UI `Colab Pro home`), không
tiếp tục notebook/Drive tài khoản miễn phí ở trên. Người dùng import cùng file
notebook, tạo bản mới:
https://colab.research.google.com/drive/1SrEvcDTYVQGUKhpRyl_s2befBZAdfqKo

Đã xác nhận scope effective_batch, seed44, source SHA cũ, 7 code cell; runtime
type chọn T4, Python3, Latest, không bật High-RAM. Không mua/nâng cấp gói.
Brave ngắt kết nối hai lần trong quá trình cấp runtime; đã nối lại notebook
đúng ID, không import lặp, không terminate phiên và không khởi chạy train lặp.

Lần quan sát cuối runtime đã tới bước xin quyền `Permit this notebook to access
your Google Drive files?`. Đã yêu cầu người dùng tự cấp quyền bằng đúng tài khoản
đang chạy. **Chưa xác nhận mount Drive/input/môi trường/CUDA thực tế hoặc epoch
train trên tài khoản này.** Không suy luận Pro đảm bảo đủ quota hay hoàn thành.
Bằng chứng: `artifacts/effective_batch_pilot/colab_tung_drive_permission.jpg`.

## Tiếp tục setup — 12/11/2026

Drive tài khoản hiện tại đã mount. Người dùng upload đúng gói input80.242.752byte;
output cell1 xác nhận ghi vào `MyDrive/TACS_Advantage_Pilot/`. Không cần tải lại
gói này sau khi thay runtime nếu mount lại đúng Drive.

Lần kiểm tra môi trường đầu thất bại dù torch/torchvision/timm khớp. Chẩn đoán
trực tiếp cho thấy `nvidia-smi: None`, CUDA availableFalse, device count0,
`torch.cuda.init()` báo không có NVIDIA driver; Resources cũng hiển thị backend
không có GPU. Tùy chọn notebook vẫn chọn T4, vì vậy chỉ nhìn tùy chọn T4 không đủ.
Người dùng đã đồng ý ngắt/xóa runtime CPU để cấp T4 mới; runtime CPU đã ngắt,
file/RAM tạm không được giữ, nhưng input trên Drive được giữ nguyên.

Runtime mới hiển thị backendGPU và `T4 (Python3)`. Preflight `nvidia-smi` xác nhận
Tesla T4 trước cài đặt. Đã quan sát pip cài thành công torch2.11.0+cu130,
torchvision0.26.0+cu130; timm1.0.30 đang ở bước cài trong lần quan sát cuối trước
kết nối Brave bị ngắt. Sau nối lại UI đang `Resuming execution`; chưa xác nhận
`MATCHED ENVIRONMENT`, mount lại Drive, encoder/source/data hoặc epoch train.

Builder/notebook Colab đã bổ sung fail-fast trước pip khi không có nvidia-smi
hoặc GPU khác TeslaT4, và stream stdout/stderr của pip theo dòng để hiện tiến độ.
11 unit tests `test_advantage*.py` đạt, bao gồm CPU/wrongGPU/T4preflight.
Source bundle thuật toán, recipe và comparator không đổi.

Cell môi trường live đã cập nhật và xác minh bằng copy toàn bộ nội dung editor
so sánh đúng với builder. Monaco fill ban đầu giữ thừa đoạn cũ; đã thay toàn bộ
bằng Select All + Paste và kiểm tra chỉ có một assignment `required`. Có thêm
một cell chẩn đoán GPU chỉ đọc trên notebook live; không thay thuật toán.
