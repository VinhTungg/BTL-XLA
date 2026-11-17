# Cleanup project, 17/11/2026

## Phạm vi và trạng thái thực tế

Yêu cầu: làm gọn project, bỏ thành phần không cần thiết nhưng giữ demo,
khả năng kiểm chứng kết quả và lịch sử Git. Không train thêm, không đổi
model/loss/split/recipe, không xóa checkpoint hoặc sửa source bundle đã train.

Đã thực hiện:

- Bỏ 18 file tracked của `.git-initialized-backup/`. Thư mục này là khung
  Git khởi tạo cũ, chỉ có config/HEAD/sample hooks; không có objects hoặc
  refs chứa commit. Có thể khôi phục các file từ Git của project nếu cần.
- Rút gọn README, đưa demo hai dataset và kịch bản slide lên trước; sửa
  lệnh cài extra `dev` không tồn tại thành cài project thông thường.
- Thêm mục lục `docs/README.md`, tách tài liệu hiện hành khỏi báo cáo
  CIFAR-10/tiny và kế hoạch cũ, giữ nguyên link và nội dung bằng chứng.
- Ignore khung backup Git cũ để không bị add lại.
- Thêm script cleanup opt-in có dry-run mặc định và kiểm thử read-only.
- Cô lập hai test sinh notebook Kaggle: kiểm tra nội dung được sinh trong
  bộ nhớ, không ghi lại notebook trong `artifacts/` thật mỗi khi chạy test.

**Chưa xóa cache và 3 file tạm Git:** công cụ thực thi chặn lệnh xóa
filesystem, kể cả thư mục cache cụ thể. Không chuyển sang công cụ khác
để lách chặn và không báo đã thu hồi dung lượng khi chưa xóa được.
Chủ project đã duyệt xóa đúng 3 file tạm Git phía dưới; có thể thực hiện
bằng script sau khi tự xem dry-run trên máy.

## Nhóm có thể dọn, đã xác định đường dẫn

Dung lượng là snapshot trước cleanup, không phải số byte đã giải phóng:

| Mục | Dung lượng xấp xỉ | Lý do có thể dọn |
|---|---:|---|
| `.venv-clean/` | 0,684 GiB | Môi trường kiểm thử cũ; source và thư viện đang dùng từ `.venv/` |
| `.pytest_cache/`, bytecode trong source/tests/scripts | Nhỏ | Cache kiểm thử/Python có thể tái tạo |
| `tmp/pdfs/`, `tmp/uv-cache/` | Khoảng 2 MiB tổng | Bản trích/render PDF và cache cài gói tạm, không phải PDF gốc |
| `.git/objects/8c/tmp_obj_jLGv0Z` | 964.796.416 byte | Git báo garbage; được chủ project duyệt xóa |
| `.git/objects/8c/tmp_obj_WHH6O0` | 0 byte | Git báo garbage; được chủ project duyệt xóa |
| `.git/objects/96/tmp_obj_RDE2Fw` | 1.110.908.928 byte | Git báo garbage; được chủ project duyệt xóa |

Ba file tạm có tổng 1,93 GiB. Tổng tiềm năng cùng môi trường cũ và cache
khoảng **2,62 GiB**, chưa phải dung lượng đã giải phóng. Xóa 3 file tạm
không có nghĩa xóa các loose objects hoặc prune/rewrite lịch sử Git.
Đây là tổng kích thước logic các file; dung lượng đĩa thực thu hồi có thể
khác do hardlink, nén filesystem hoặc các tiến trình khác ghi đĩa.

## Tự chạy cleanup đã giới hạn phạm vi

Từ thư mục gốc, PowerShell. Xem danh sách, **không xóa gì**:

```powershell
.\scripts\clean_project.ps1
```

Sau khi kiểm tra danh sách, thực hiện:

```powershell
.\scripts\clean_project.ps1 -Apply
```

Script chỉ nhắm các cache đã liệt kê, bytecode trong `src/tests/scripts`,
khung backup rỗng và đúng ba tên file Git đã duyệt. Các file mới không rõ
trong `tmp/` hoặc file Git tạm tên khác không tự được thêm vào danh sách.
Mặc định không xóa; `-Apply -WhatIf` cũng chỉ liệt kê. Script từ chối đường
dẫn ngoài project, symlink/junction và môi trường phụ đang được sử dụng.
Khi áp dụng, trạng thái từng mục được lưu vào JSON mới trong `artifacts/cleanup/`;
lỗi một phần được ghi rõ. Script không có chức năng restore sau xóa.

Cache và môi trường phụ có thể dựng lại; ba file Git tạm bị xóa thì không
được script sao lưu. Giữ nguyên `.venv` chính, dữ liệu và artifact nghiên cứu.

## Nhóm được giữ nguyên

- `.venv/`, thiết lập IDE và hướng dẫn agent của người dùng.
- `data/`, `artifacts/`, `outputs/`, `cifar100/`: dataset, split, checkpoint,
  ZIP nguồn, manifest, metrics và kết quả đối chứng.
- `artifacts/demo_cache/`: demo cần dùng lại cache, tránh encode lại trước buổi bảo vệ.
- Bài báo PDF gốc, bản dịch, kịch bản slide và tài liệu kết quả đã xác minh.
- Toàn bộ source huấn luyện/demo, scripts cloud, notebook và tests.
- Toàn bộ Git objects, refs và lịch sử commit ngoài ba file tạm đã duyệt.

Không xóa archive chỉ vì đã giải nén; không xóa checkpoint legacy vì chúng
vẫn là bằng chứng cho số liệu lịch sử. Khi muốn giảm tiếp nhiều GB phải
kiểm tra backup/provenance và duyệt riêng, không gọi chúng là cache.

## Kiểm tra lại

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101 --check-ui
git count-objects -vH
```

`--check-ui` chỉ kiểm tra tạo widget, không chứng minh inference hoặc bố cục
trực quan. Hai integration dữ liệu thật mặc định không bật. Không chạy lại
train hay mở test để nghiệm thu cleanup.

Nghiệm thu trong lượt cleanup: **167 tests, 0 errors, 0 failures, 2 skipped**.
Hash và thời gian sửa của hai notebook role-pilot không đổi sau chạy test.
Đã tạo/hủy widget thành công ở cả Caltech và CIFAR. Đã suy luận thật Caltech
validation position178: selected/random/frozen/No-context độc lập dự đoán
electric_guitar, bỏ context trên cùng Two-Stage weights dự đoán mandolin;
`weights_updated=false`, `test_evaluated=false`. Không ghi thêm report inference.
Chưa thực thi nhánh xóa `-Apply`, không nhận đã nghiệm thu thu hồi dung lượng.

Hai trường hợp biên: mặc định script chỉ liệt kê và phải giữ nguyên filesystem;
nếu môi trường phụ đang chạy hoặc target có junction/symlink thì phải dừng,
không xóa một đường dẫn rộng hơn để xử lý lỗi.
