# Kịch bản thuyết trình và demo TACS

Ngày chốt nội dung: **16/11/2026**. Mốc source trước khi viết: `e681d38`.
Đây là **kịch bản để dựng slide và luyện nói**, chưa phải file PowerPoint.

Tên đề tài đề nghị:

> **Tái hiện TACS cho phân loại ảnh và phát triển truy xuất context hai giai đoạn**
>
> Thực nghiệm CIFAR-100 và demo Caltech-101

Tên bài báo gốc: *Learning What Helps: Task-Aligned Context Selection for Vision Tasks*,
Jingyu Guo và cộng sự, CVPR 2026. Dùng đúng bản PDF CVPR người dùng cung cấp,
không thay bằng thông số của bản arXiv trước đó.

## 1. Cách dùng kịch bản

Mỗi slide có phần **Chữ trên slide**, **Hình/bố cục**, **Lời nói**, **Chuyển ý**
và **Nguồn/điểm cần giữ đúng**. Chỉ đưa phần chữ và hình lên màn hình. Lời nói
đặt trong speaker notes, không chép nguyên đoạn thành slide.

Giả định chưa có quy định thời lượng khác từ giảng viên: **15 slide chính,
12 phút 30 giây trình bày và 3 phút 30 giây demo**, tổng **16 phút** trước hỏi đáp.
Thời gian là ngân sách luyện tập, không phải thời lượng đã đo của người nói.
Có 6 slide dự phòng A1-A6, chỉ mở khi được hỏi.

Nhóm có 3 người. Thay TV1, TV2, TV3 bằng tên thật; đây là phân công thuyết trình,
không phải khẳng định ai đã làm phần source nào.

| Slide | Nội dung | Người | Giây |
|---|---|---|---:|
| 1 | Giới thiệu đề tài | TV1 | 30 |
| 2 | Vấn đề chọn ảnh hỗ trợ | TV1 | 45 |
| 3 | Bài báo và phạm vi nhóm | TV1 | 50 |
| 4 | Kiến trúc TACS | TV1 | 60 |
| 5 | Cơ chế học chọn context | TV1 | 75 |
| 6 | Sửa vai trò query/context | TV2 | 55 |
| 7 | Phát triển Two-Stage | TV2 | 55 |
| 8 | Dữ liệu và cách chia tập | TV2 | 60 |
| 9 | Thiết lập và đối chứng | TV2 | 50 |
| 10 | Pilot CIFAR-100 | TV2 | 60 |
| 11 | Kết quả Caltech-101 | TV2 | 60 |
| 12 | Tác dụng thực tế của context | TV3 | 60 |
| 13 | Demo trực tiếp | TV3 | 210 |
| 14 | Giới hạn và hướng tiếp theo | TV3 | 50 |
| 15 | Kết luận | TV3 | 40 |
| **Tổng** | **Thuyết trình 750 giây và demo 210 giây** | | **960** |

TV1 nói 4 phút 20 giây. TV2 nói 5 phút 40 giây. TV3 nói 6 phút, gồm demo.
Trong demo, TV2 có thể thao tác máy để TV3 tập trung giải thích.

### Các quy tắc bắt buộc về số liệu

- Ghi **validation** ngay trên các bảng pilot và Caltech. Không gọi là test.
- Không gộp `legacy_v1`, `role_v2`, 20 epoch và 100 epoch thành một bảng xếp hạng.
- Không gọi 4 chế độ suy luận trên cùng weights là 4 baseline train độc lập.
- Two-Stage đã triển khai nhưng **chưa chứng minh tăng accuracy** so với đối chứng.
- Multi-Context và cơ chế tự bỏ context mới chỉ là **đề xuất**, chưa có kết quả.
- Không nhận đã tái hiện toàn bộ bài báo, đủ baseline tác giả hoặc đủ 3 seed mới.
- Kết quả tốt/xấu ở ảnh demo không thay cho kết quả toàn validation.

## 2. Kịch bản 15 slide chính

### Slide 1. Tái hiện TACS và truy xuất context hai giai đoạn

**Người nói:** TV1. **Thời gian:** 30 giây.

**Mục tiêu:** Nêu đúng đề tài và báo trước hai phần tái hiện, phát triển.

**Chữ trên slide**

> Tái hiện TACS cho phân loại ảnh
>
> Phát triển truy xuất context hai giai đoạn
>
> CIFAR-100 và Caltech-101
>
> Học phần Xử lý ảnh | Nhóm [điền tên và mã sinh viên]
>
> Dựa trên Guo và cộng sự, CVPR 2026

**Hình/bố cục**

Một sơ đồ nhỏ gồm ba khối: ảnh cần phân loại, ảnh hỗ trợ, kết quả dự đoán.
Không đưa ảnh trang bìa bài báo chiếm hết slide. Chỉ cần tên bài báo ở chân trang.

**Lời nói**

“Em chào cô và các bạn. Nhóm em thực hiện đề tài tái hiện phương pháp TACS,
tức học cách chọn ảnh hỗ trợ cho một mô hình thị giác. Dựa trên bài báo CVPR
2026 của Guo và cộng sự, nhóm tập trung vào nhánh phân loại ảnh, sau đó phát
triển truy xuất hai giai đoạn để giới hạn số ứng viên mà selector cần xét.
Phần trình bày gồm nguyên lý, triển khai và thực nghiệm, sau đó demo bằng
checkpoint thật trên CIFAR-100 và Caltech-101.”

**Chuyển ý**

“Trước hết, vì sao một mô hình đã có ảnh đầu vào lại cần thêm một ảnh khác?”

**Nguồn/điểm cần giữ đúng:** [S01]. Không đặt tên “Tái hiện hoàn chỉnh bài báo TACS”.

### Slide 2. Chọn ảnh hỗ trợ cho phân loại

**Người nói:** TV1. **Thời gian:** 45 giây.

**Mục tiêu:** Người nghe hiểu “hữu ích” khác “nhìn giống”.

**Chữ trên slide**

> Query: ảnh cần dự đoán nhãn
>
> Context: ảnh được chọn để hỗ trợ dự đoán
>
> Ảnh giống query chưa chắc làm dự đoán tốt hơn
>
> Mục tiêu: học chọn context theo lợi ích cho tác vụ

**Hình/bố cục**

Sơ đồ một query đi vào hai cách xử lý: phân loại độc lập hoặc phân loại cùng
context lấy từ kho ứng viên. Đặt câu “Context có thể giúp hoặc gây nhiễu” dưới
nhánh thứ hai. Đây là minh họa cơ chế, không dùng phần trăm giả.

**Lời nói**

“Query là ảnh cần nhận dạng, còn context là ảnh phụ mà mô hình được xem thêm.
Ví dụ, khi phân biệt hai loại nhạc cụ, một ảnh đối chiếu có thể giúp làm rõ
đặc điểm khác nhau. Nhưng nếu chọn ảnh phụ không phù hợp thì mô hình cũng có
thể bị nhiễu. Vì vậy, lấy ảnh nhìn giống nhất chưa chắc là lựa chọn tốt nhất.
TACS đặt vấn đề học trực tiếp cách chọn ảnh nào làm mô hình phân loại tốt hơn.
Điểm cần nhớ là ảnh hỗ trợ không bắt buộc cùng lớp với query, và thêm context
không mặc định làm accuracy tăng.”

**Chuyển ý**

“Từ ý tưởng này, nhóm xác định phần nào của nghiên cứu sẽ được tái hiện.”

**Nguồn/điểm cần giữ đúng:** [S01], Sec. 1 và Sec. 3. Ví dụ nhạc cụ là cách giải
thích khái niệm, không phải một bằng chứng cơ chế chú ý đã đo.

### Slide 3. Bài báo gốc và phạm vi thực hiện

**Người nói:** TV1. **Thời gian:** 50 giây.

**Mục tiêu:** Phân biệt quy mô nghiên cứu gốc với phạm vi có bằng chứng của nhóm.

**Chữ trên slide**

| Bài báo TACS | Phần nhóm thực hiện |
|---|---|
| 18 bộ dữ liệu tự nhiên và y tế | CIFAR-100 và Caltech-101 |
| Phân loại và phân đoạn | Nhánh phân loại ảnh |
| Selector và task network học đồng thời | Triển khai cơ chế chọn context và hybrid loss |
| 4 baseline đối sánh và TACS | Các run lịch sử, pilot có kiểm soát và demo |

> Phần phát triển: Two-Stage retrieval và phân tích tác dụng của context

**Hình/bố cục**

Bảng hai cột là nội dung chính. Không vẽ biểu đồ accuracy trên slide này.
Thêm tên bài báo đầy đủ ở speaker notes. Bảng baseline tác giả nằm ở A1.

**Lời nói**

“Bài báo đánh giá trên 18 bộ dữ liệu, gồm phân loại ảnh tự nhiên, phân loại
ảnh y tế và phân đoạn y tế. Nhóm không nhận đã tái hiện toàn bộ phạm vi đó.
Trong đồ án, nhóm triển khai nhánh phân loại với DINOv3, thực nghiệm trên
CIFAR-100 và mở rộng sang Caltech-101 để demo ảnh nguồn rõ hơn.
Bài báo có bốn baseline đối sánh cộng với TACS. Two-Stage không phải tên
baseline của tác giả mà là phần phát triển trong đồ án. Các kết quả mới của
nhóm vẫn còn giới hạn về số seed và protocol, nên sẽ được trình bày tách
khỏi điểm benchmark của bài báo.”

**Chuyển ý**

“Sau đây là luồng xử lý cốt lõi mà nhóm tái hiện từ TACS.”

**Nguồn/điểm cần giữ đúng:** [S01], Tables 1-2, Sec. 4.1 và 4.3; [S02]-[S07].
Không đánh dấu đã hoàn thành Feature-Averaged Context trong protocol mới.

### Slide 4. Kiến trúc TACS

**Người nói:** TV1. **Thời gian:** 60 giây.

**Mục tiêu:** Phân biệt selector chọn ảnh với task network dự đoán nhãn.

**Chữ trên slide**

> Selector: chấm điểm ứng viên và chọn một context
>
> Task network: nhận query và context để dự đoán nhãn query
>
> Backbone: DINOv3 ViT-S/16
>
> Phân loại: ghép patch token của query và context

**Hình/bố cục**

Vẽ lại sơ đồ Fig. 2 theo 5 khối: Query, Candidate pool, Selector, Selected
context, Task network. Hai đầu vào của task network phải thấy rõ là query
và context. Nhánh nhãn thật đi vào loss khi train, không đi vào selector
khi inference. Chú thích “Vẽ lại theo Guo và cộng sự, Fig. 2”.

**Lời nói**

“TACS có hai bộ phận chính. Selector nhận biểu diễn của query và các ảnh
trong kho ứng viên, sau đó chấm điểm để chọn một context. Biểu diễn, hay
embedding, có thể hiểu là một dãy số tóm tắt đặc điểm của ảnh.
Task network mới là bộ phận đưa ra nhãn cuối cùng. Với bài toán phân loại,
ảnh được chia thành các ô nhỏ gọi là patch, rồi ghép token của query với
context trước các khối Transformer. Hai bộ phận có hai nhiệm vụ khác nhau:
một bên học chọn ảnh nào có ích, bên còn lại học phân loại query từ cặp ảnh.
Nhãn context không được đưa vào để suy ra nhãn query trong quá trình dự đoán.”

**Chuyển ý**

“Vấn đề tiếp theo là làm sao dạy selector biết lựa chọn nào thực sự có ích.”

**Nguồn/điểm cần giữ đúng:** [S01], Sec. 3.1 và 4.2; [S09]. ViT-S/16 nghĩa là
bản ViT small, patch 16 pixel, không phải ảnh đầu vào chỉ có 16 pixel.
Đừng khẳng định selector và task bắt buộc dùng chung một instance trọng số
chỉ vì paper dùng từ “share a backbone”.

### Slide 5. Học chọn context bằng hai tín hiệu

**Người nói:** TV1. **Thời gian:** 75 giây.

**Mục tiêu:** Giải thích được reward, Gumbel-Softmax và hybrid loss.

**Chữ trên slide**

> Reward = Loss không context − Loss có context
>
> Loss tổng = Loss phân loại + λ × Loss policy
>
> Gumbel-Softmax: chọn rời rạc nhưng vẫn truyền gradient
>
> Train có lấy mẫu, inference chọn ứng viên điểm cao nhất
>
> Bản CVPR 2026: τ = 0,1 và λ = 1,0

**Hình/bố cục**

Hai nhánh phản hồi từ task về selector: gradient theo task loss và policy
gradient theo reward. Chỉ giữ 2 công thức trên màn hình; công thức advantage
đầy đủ để ở A6. Có thể lần lượt hiện hai nhánh, không cần animation phức tạp.

**Lời nói**

“TACS kết hợp hai tín hiệu để huấn luyện selector. Tín hiệu thứ nhất đến từ
loss phân loại, thông qua Gumbel-Softmax. Kỹ thuật này cho phép lượt tính
đi tới chọn một ảnh cụ thể, trong khi lượt truyền ngược vẫn có tín hiệu
gradient để học. Gradient có thể hiểu là chỉ dẫn cách điều chỉnh trọng số.
Tín hiệu thứ hai là reward: lấy loss khi không có context trừ loss khi có
context. Chẳng hạn loss giảm từ 0,8 xuống 0,5 thì reward bằng 0,3, tức cặp
ảnh có lợi theo tiêu chí này. Reward được chuẩn hóa thành advantage để cập
nhật policy, tức quy tắc chọn ảnh. Loss tổng kết hợp hai nhánh. Khi suy
luận, không cần nhãn thật hay tính reward, mà chọn ứng viên có điểm cao nhất.
Lưu ý reward dương nghĩa là loss giảm, không nhất thiết dự đoán vừa đổi từ
sai thành đúng.”

**Chuyển ý và bàn giao TV2**

“Đó là nguyên lý của bài báo. Tiếp theo, bạn [TV2] sẽ trình bày các quyết
định triển khai và phần phát triển của nhóm.”

**Nguồn/điểm cần giữ đúng:** [S01], Eqs. 4-10, Sec. 4.2; [S09]. Reward phải
detach khi dùng cho policy loss. Với embedding bank, code nhóm dùng proxy
gradient và refresh bank theo epoch. Đây là lựa chọn triển khai, không phải
bằng chứng đã khớp mọi chi tiết gradient của tác giả. Xem A3.

### Slide 6. Phân biệt vai trò query và context

**Người nói:** TV2. **Thời gian:** 55 giây.

**Mục tiêu:** Trình bày một vấn đề triển khai có nguyên nhân và kiểm chứng rõ.

**Chữ trên slide**

> Vấn đề bản cũ: đổi chỗ hai ảnh gần như không đổi dự đoán
>
> Nhưng nhãn cần dự đoán luôn là nhãn của query
>
> role_v2: thêm vector vai trò vào patch của context
>
> Đã kiểm tra tính bất đối xứng và đường gradient

**Hình/bố cục**

Minh họa hai cặp đầu vào (A là query, B là context) và cặp đổi vai trò.
Đặt nhãn “cần phân biệt vai trò” ở giữa. Không dùng biểu đồ accuracy trước/sau.

**Lời nói**

“Khi kiểm tra source và checkpoint cũ, nhóm phát hiện một vấn đề ở task
network. Hai nhóm patch dùng chung cấu trúc nhưng thiếu dấu phân biệt vai
trò, nên đổi chỗ query và context gần như không làm logits thay đổi, trong
khi nhãn phải đi theo query. Logits là điểm phân loại trước softmax.
Bản role_v2 thêm một vector học được vào patch context để phân biệt hai
vai trò. Phép thử xác nhận việc đổi ảnh đã làm đầu ra khác nhau, nhánh
query-only không đổi và gradient vẫn đi qua được. Đây là bản sửa triển
khai của nhóm, chưa phải bằng chứng rằng accuracy chắc chắn tăng, cũng
không được đổi tên checkpoint cũ thành checkpoint đã train role_v2.”

**Chuyển ý**

“Ngoài việc sửa triển khai, nhóm phát triển một cách giới hạn không gian chọn context.”

**Nguồn/điểm cần giữ đúng:** [S03]. Chẩn đoán 4 cặp thật cho legacy có max
logit delta khoảng `3,34e-6`; role_v2 mới khởi tạo khoảng `0,0678-0,1134`.
Không diễn giải các số delta này thành mức cải thiện accuracy.

### Slide 7. Two-Stage retrieval

**Người nói:** TV2. **Thời gian:** 55 giây.

**Mục tiêu:** Chỉ đúng phần nhóm phát triển, lợi ích kỳ vọng và đánh đổi.

**Chữ trên slide**

> Giai đoạn 1: Frozen DINOv3 và cosine để lấy top-32
>
> Giai đoạn 2: Selector đã học chọn một context trong top-32
>
> Task network phân loại query cùng context được chọn
>
> Đánh đổi: giảm số ứng viên ở selector nhưng có thể bỏ sót context hữu ích

**Hình/bố cục**

Sơ đồ một hàng: kho N ảnh, retrieval, 32 ứng viên, learned selector, 1 context,
task network. Dùng nhãn “frozen” và “trainable” để phân biệt hai encoder.
Ghi nhỏ: CIFAR pool 9.000; Caltech pool 606. Không ghi hệ số tăng tốc.

**Lời nói**

“Phần phát triển của nhóm là truy xuất hai giai đoạn. Trước hết, encoder
DINOv3 cố định lấy 32 ảnh gần query theo cosine. Cosine là cách so mức độ
cùng hướng của hai vector đặc trưng. Sau đó, selector đã học chỉ chấm điểm
32 ảnh này và chọn một ảnh cho task network.
Mục đích là thu hẹp phần tìm kiếm có học. Tuy nhiên, retrieval vẫn phải
tìm trên kho đặc trưng và còn chi phí cache, nên không thể lấy tỷ lệ
9.000 chia 32 để khẳng định hệ thống nhanh từng đó lần. Đánh đổi quan
trọng là một context có ích nhưng không lọt top-32 sẽ không được selector
xem xét. Vì vậy, hiệu quả accuracy và tốc độ đều cần đo, không mặc định tốt hơn.”

**Chuyển ý**

“Để kiểm tra hướng này, nhóm sử dụng hai bộ dữ liệu với vai trò khác nhau.”

**Nguồn/điểm cần giữ đúng:** [S04], [S07], [S09]. Frozen reference là encoder
pretrained được đóng băng, không gọi nó là task encoder đã fine-tune.
Cosine ở bước retrieval khác với việc thử đổi learned selector sang cosine.

### Slide 8. Dữ liệu và cách chia tập

**Người nói:** TV2. **Thời gian:** 60 giây.

**Mục tiêu:** Hiểu tại sao có hai dataset và vì sao không trộn pool/validation/test.

**Chữ trên slide**

| Thuộc tính | CIFAR-100 | Caltech-101 |
|---|---:|---:|
| Số lớp | 100 | 101 |
| Query train | 36.000 | 1.919 |
| Candidate pool | 9.000 | 606 |
| Validation | 5.000 | 505 |
| Test tách riêng | 10.000 | 5.647 |
| Ảnh nguồn | 32 × 32 | Kích thước thay đổi, lớn hơn CIFAR |
| Ảnh vào mô hình | 256 × 256 | 256 × 256 |

> Split seed 42. Caltech dùng protocol riêng 19 query, 6 pool, 5 validation mỗi lớp.

**Hình/bố cục**

Bảng là phần chính. Nếu thêm ảnh, chỉ một cặp minh họa kích thước nguồn,
giữ đúng tỷ lệ và không làm nét CIFAR bằng AI. Không cố nhét nhiều ảnh vào bảng.

**Lời nói**

“CIFAR-100 là bộ dữ liệu thực nghiệm chính ban đầu, gồm 100 lớp nhưng ảnh
gốc chỉ 32 nhân 32 pixel. Phóng lên 256 không tạo lại chi tiết bị thiếu.
Vì muốn demo trên ảnh nguồn rõ hơn, nhóm thêm Caltech-101, không thay thế
hay bỏ kết quả CIFAR.
Trên Caltech, mỗi lớp lấy 19 ảnh query train, 6 ảnh vào pool và 5 ảnh
validation; số còn lại tách thành test. Pool chỉ cung cấp ảnh hỗ trợ,
không lấy ảnh validation hay test để bổ sung. Validation dùng chọn
checkpoint, còn test của các thí nghiệm mới chưa được dùng để chọn
phương án. Đây là protocol riêng của nhóm, không được coi là bản sao
chính xác standard split của bài báo.”

**Chuyển ý**

“Ngoài cách chia tập, cần tách rõ thiết lập train và loại phép so sánh.”

**Nguồn/điểm cần giữ đúng:** [S04], [S07], [S10]. CIFAR test lịch sử đã được
đánh giá, nhưng **pilot mới** và Caltech chưa test. Caltech loại
BACKGROUND_Google, còn 8.677 ảnh. Không nhận đã loại mọi near-duplicate:
kiểm tra hiện có là index và hash ảnh giữa các vai trò.

### Slide 9. Thiết lập và hai loại đối chứng

**Người nói:** TV2. **Thời gian:** 50 giây.

**Mục tiêu:** Chặn hiểu nhầm giữa baseline train riêng và thao tác bỏ context.

**Chữ trên slide**

> DINOv3 ViT-S/16, AdamW, LR 0,0001
>
> Batch vật lý 4, tích lũy 16 bước, AMP
>
> Pilot CIFAR: 20 epoch | Caltech: 100 epoch
>
> Baseline độc lập: No-context train riêng
>
> Can thiệp inference: đổi cách lấy context, giữ nguyên task weights

**Hình/bố cục**

Chia hai vùng bằng một đường dọc. Vùng “Train độc lập” có hai checkpoint
No-context và Two-Stage. Vùng “Giữ nguyên weights” có selected, random,
retrieval top-1, bỏ context. Không vẽ thành 6 model ngang hàng.

**Lời nói**

“Các run mới dùng DINOv3 ViT-S/16, AdamW với learning rate 0,0001. Vì giới
hạn GPU, batch vật lý là 4 và tích lũy 16 bước, tương đương 64 mẫu cho một
bước cập nhật optimizer, nhưng không tự động tương đương batch vật lý 64
trong mọi phép tính.
Có hai câu hỏi đối chứng khác nhau. Train No-context riêng rồi so với
Two-Stage cho biết hệ thống nào tốt hơn trên protocol này. Còn giữ nguyên
weights của Two-Stage và thay cách chọn ảnh chỉ giúp phân tích ảnh hưởng
của context. Bỏ context trên model đã train với context không phải là
baseline No-context đã train riêng.”

**Chuyển ý**

“Trước hết là kết quả pilot kiểm tra cách chuẩn hóa reward trên CIFAR-100.”

**Nguồn/điểm cần giữ đúng:** [S04], [S07]-[S08]. AMP là tính toán hỗn hợp độ
chính xác để giảm bộ nhớ/tăng hiệu suất. Không dùng 20 epoch pilot như
kết quả đã hoàn tất recipe 100 epoch của paper.

### Slide 10. Pilot CIFAR-100: chuẩn hóa advantage

**Người nói:** TV2. **Thời gian:** 60 giây.

**Mục tiêu:** Báo cáo một ablation có đối chứng, kể cả kết quả không cải thiện.

**Chữ trên slide**

> Two-Stage role_v2 | Seed train 44 | 20 epoch | 5.000 validation

| Phạm vi chuẩn hóa advantage | Best validation | Epoch best |
|---|---:|---:|
| Microbatch 4 | 87,92% | 20 |
| Effective batch 64 | 87,80% | 17 |

> Chênh lệch best: −0,12 điểm phần trăm
>
> Chưa có bằng chứng effective-batch cải thiện accuracy

**Hình/bố cục**

Bảng ba cột, không cần biểu đồ cột cắt trục phóng đại chênh lệch. Ghi
“validation-only, một seed” ở ngay dưới bảng, không giấu trong notes.

**Lời nói**

“Advantage là reward đã được chuẩn hóa để cập nhật selector ổn định hơn.
Với gradient accumulation, optimizer cập nhật sau 64 mẫu nhưng nếu code
chuẩn hóa ngay ở từng microbatch thì reward vẫn chỉ được so trong 4 mẫu.
Nhóm làm một pilot giữ cùng source, split và recipe, thay phạm vi chuẩn
hóa từ 4 sang 64 mẫu. Kết quả best validation lần lượt là 87,92 và 87,80
phần trăm. Effective-batch thấp hơn 0,12 điểm phần trăm, nên chưa có cơ
sở chọn nó là hướng tăng accuracy.
Đây là kết quả sàng lọc 20 epoch với một seed. Nhóm không lấy điểm này
so trực tiếp với test 100 epoch của bài báo, cũng không coi chênh lệch
nhỏ này là kết luận có ý nghĩa thống kê.”

**Chuyển ý**

“Tiếp theo là hai mô hình đã train đủ 100 epoch trên Caltech-101.”

**Nguồn/điểm cần giữ đúng:** [S04]. Các số trên là best ghi lúc train.
Audit inference đo lại cho 87,86% và 87,82%, ở môi trường khác; không thay
lẫn hai bộ số. Pilot có khác biệt nền tảng và resume nên không quảng cáo
đây là phép thử mọi điều kiện đều giống hệt bit-for-bit.

### Slide 11. Caltech-101: kết quả hai mô hình train riêng

**Người nói:** TV2. **Thời gian:** 60 giây.

**Mục tiêu:** Trình bày đúng kết quả chính của dataset demo.

**Chữ trên slide**

> Cùng split và recipe 100 epoch | Seed train 44 | 505 validation

| Mô hình train độc lập | Epoch best | Best validation | Đúng/505 khi replay |
|---|---:|---:|---:|
| No-context | 21 | 95,64% | 483 |
| Two-Stage role_v2 | 54 | 94,26% | 476 |

> Two-Stage thấp hơn 7 ảnh, khoảng 1,39 điểm phần trăm
>
> Chưa đánh giá test. Đây là protocol Caltech riêng của nhóm.

**Hình/bố cục**

Bảng hai dòng với cột No-context được giữ trung tính, không tô “kém” vì là
baseline. Dòng test/protocol phải đủ lớn để người ngồi cuối lớp đọc được.

**Lời nói**

“Trên Caltech-101, cả No-context và Two-Stage đều đã train 100 epoch với
cùng split, cùng source và recipe. Checkpoint tốt nhất được chọn theo
validation: No-context ở epoch 21 đạt 95,64 phần trăm, còn Two-Stage ở
epoch 54 đạt 94,26 phần trăm. Replay toàn bộ 505 ảnh cho đúng 483 và
476 ảnh, tương ứng chênh 7 ảnh.
Như vậy, ở lượt chạy này Two-Stage chưa vượt baseline độc lập. Nhóm dùng
best checkpoint để demo chứ không mặc định dùng epoch 100. Các số này
là validation đã tham gia chọn checkpoint và chỉ có một seed, nên chưa
đủ kết luận tổng quát. Cũng không thể nói No-context của nhóm vượt tác
giả, vì protocol và cách tổng hợp kết quả không giống bài báo.”

**Chuyển ý và bàn giao TV3**

“Dù chưa thắng baseline, context có thực sự làm thay đổi dự đoán hay không?
Bạn [TV3] sẽ phân tích và demo phần này.”

**Nguồn/điểm cần giữ đúng:** [S07]-[S08], [S11]. Hiệu hai số cloud đã làm tròn
là 1,38; hiệu chính xác 7/505 × 100 là 1,3861, làm tròn thành 1,39.
Chọn cách báo theo số ảnh để tránh sai khác làm tròn. Không gọi epoch 21/54
là run chỉ train 21/54 epoch: lịch sử cả hai đã đủ 100.

### Slide 12. Context giúp trong trường hợp nào?

**Người nói:** TV3. **Thời gian:** 60 giây.

**Mục tiêu:** Cho thấy lợi ích cục bộ không đồng nghĩa thắng baseline độc lập.

**Chữ trên slide**

> Giữ nguyên task weights của Two-Stage, thay cách lấy context

| Cách suy luận | Đúng/505 | Validation replay |
|---|---:|---:|
| Selector đã học | 476 | 94,26% |
| Frozen retrieval top-1 | 477 | 94,46% |
| Random trong top-32 | 474 | 93,86% |
| Bỏ context, cùng weights | 470 | 93,07% |

> Context đã chọn sửa đúng 8 ảnh nhưng làm sai 2 ảnh so với bỏ context
>
> Random là một trial seed 44. Không phải bốn baseline train độc lập.

**Hình/bố cục**

Bảng bốn dòng. Có thể thêm một dòng tóm tắt “Ròng +6 ảnh trên cùng weights”.
Không lặp bảng No-context độc lập ở đây; nhắc bằng lời để tránh trộn phép so sánh.

**Lời nói**

“Ở phép phân tích này, nhóm không train thêm mà giữ nguyên weights của
Two-Stage, chỉ thay cách cung cấp context trên cùng 505 ảnh validation.
Selector đã học đúng 476 ảnh, trong khi bỏ context trên chính model đó
đúng 470 ảnh. Cụ thể, context sửa đúng 8 ảnh nhưng làm sai 2 ảnh, nên
lợi ích ròng là 6 ảnh.
Tuy nhiên, retrieval top-1 vẫn đúng nhiều hơn selector một ảnh, còn
No-context train riêng ở slide trước đạt 483 ảnh. Kết luận đúng là
context có tác dụng trên một số query, nhưng selector hiện tại chưa
tạo ra ưu thế so với baseline độc lập. Random ở bảng là một lượt lấy
mẫu, không phải trung bình nhiều lần hay mô hình random đã train riêng.”

**Chuyển ý**

“Nhóm sẽ minh họa cả một trường hợp có lợi và một trường hợp gây sai trên ứng dụng.”

**Nguồn/điểm cần giữ đúng:** [S08], [S11]. 8 giúp/2 hại là so selected với
no_context cùng Two-Stage weights, không phải so với trained_no_context.
Chỉ đưa chữ “causal” nếu giải thích rõ đây là can thiệp đầu vào inference,
không suy ra mô hình có cơ chế suy luận giống con người.

### Slide 13. Demo query, context và dự đoán

**Người nói:** TV3. **Thao tác máy:** TV2 nếu thuận tiện. **Thời gian:** 210 giây.

**Mục tiêu:** Chứng minh ứng dụng chạy checkpoint thật và giải thích được giới hạn.

**Chữ trên slide**

> Demo Caltech-101 và CIFAR-100
>
> Xem query, context và top-5 dự đoán
>
> Đối chiếu selector với No-context train riêng
>
> Quan sát cả context giúp và context gây sai
>
> Lưu bằng chứng JSON theo checkpoint

**Hình/bố cục**

Một slide chờ đơn giản, sau đó chuyển sang ứng dụng đã mở sẵn. Không chạy
pip install, tải weights hoặc encode pool lần đầu ngay trước lớp.
Ảnh dự phòng là screenshot thật trong [S12], không dựng lại output giả.

**Lời mở đầu demo, khoảng 20 giây**

“Đây là ứng dụng suy luận offline từ checkpoint đã train. Query là ảnh cần
phân loại, còn context được lấy từ pool train cố định. Các hàng kết quả
không đều là model train riêng: bốn hàng đầu dùng cùng task Two-Stage,
riêng hàng No-context train riêng dùng checkpoint khác.”

**Kịch bản thao tác theo giây**

| Khoảng thời gian | Thao tác | Điều chỉ ra / lời nói |
|---|---|---|
| 0-20 | Mở cửa sổ đã ở trạng thái Sẵn sàng, dataset Caltech | Nêu dữ liệu thật, checkpoint thật và hai loại No-context |
| 20-45 | Chọn `Cùng đúng ·0`, bấm so sánh | “Ảnh nguồn ví dụ này 492 × 325 pixel; mô hình vẫn nhận ảnh resize 256 × 256.” |
| 45-95 | Chọn `Context giúp ·178`, so sánh, bấm hàng selector rồi hàng bỏ context | Giải thích trường hợp electric_guitar ở đoạn dưới |
| 95-145 | Chọn `Context gây sai ·169`, so sánh, bấm hàng selector và retrieval top-1 | Giải thích trường hợp dolphin ở đoạn dưới |
| 145-175 | Bấm No-context train riêng, sau đó Lưu bằng chứng JSON với tên mới | “Không còn context; checkpoint và hash của baseline độc lập khác Two-Stage.” |
| 175-200 | Nếu đã rehearsal cache, chuyển Dataset sang CIFAR-100 và chạy ví dụ có sẵn | “Đây là dataset và checkpoint khác, không phải lấy model Caltech chấm ảnh CIFAR.” |
| 200-210 | Quay về slide | Chốt thông điệp demo |

**Lời nói khi mở vị trí 178**

“Ảnh query này có nhãn electric_guitar. Selector có context dự đoán đúng,
nhưng khi bỏ context trên cùng weights thì dự đoán thành mandolin. Đây
là một ví dụ ảnh hỗ trợ làm thay đổi kết quả theo hướng đúng. Tuy nhiên,
No-context được train riêng cũng dự đoán đúng, nên không thể dùng ảnh này
để nói Two-Stage vượt baseline. Context được chọn thuộc lớp accordion;
việc context khác lớp không tự động có nghĩa là lựa chọn sai.”

**Lời nói khi mở vị trí 169**

“Đây là trường hợp thất bại. Query là dolphin nhưng selector chọn context
hawksbill và dự đoán thành hawksbill. Khi bỏ context trên cùng weights,
hoặc dùng No-context train riêng, kết quả lại là dolphin. Vì vậy nhóm
không chỉ chọn ảnh đẹp để trình bày. Ví dụ này cho thấy ảnh hỗ trợ cũng
có thể gây nhiễu, và cần đo trên toàn validation chứ không kết luận từ
một vài ảnh.”

**Lời chốt demo**

“Demo cho quan sát trực tiếp context đã chọn và kết quả, nhưng không thay
thế đánh giá toàn tập. Bằng chứng JSON giữ thông tin model và dữ liệu để
kiểm tra lại.”

**Nếu máy chậm hoặc lỗi**

Nếu nạp quá 20 giây hoặc có lỗi artifact, không cài lại trên sân khấu.
Bỏ bước chuyển CIFAR nếu chưa đủ cache. Mở hai screenshot giúp/hại và JSON
đã lưu, nói rõ “Đây là bằng chứng của lượt chạy trước, không phải inference
trực tiếp đang thành công”. Không giả vờ screenshot là video live.

**Chuyển ý**

“Từ kết quả định lượng và hai ví dụ vừa rồi, nhóm rút ra các giới hạn sau.”

**Nguồn/điểm cần giữ đúng:** [S08], [S11]-[S12]. Vị trí là validation position
từ 0, không phải ID ảnh nguồn. Các ví dụ được chọn có chủ đích từ validation.
Softmax top-5 chưa calibration, không gọi 99% là chắc chắn đúng 99% ngoài đời.

### Slide 14. Giới hạn và hướng phát triển

**Người nói:** TV3. **Thời gian:** 50 giây.

**Mục tiêu:** Nêu việc chưa làm mà không biến ý tưởng thành kết quả.

**Chữ trên slide**

> Giới hạn hiện tại
>
> Chưa tái hiện đủ baseline và 3 seed theo paper
>
> Two-Stage chưa vượt No-context trên Caltech
>
> Protocol riêng, validation nhỏ, chưa có kết quả test mới
>
> Hướng đề xuất, chưa triển khai
>
> Chọn nhiều context hoặc học quyết định khi nào không dùng context

**Hình/bố cục**

Hai cột “Đã quan sát” và “Cần kiểm chứng”. Không đưa số accuracy mục tiêu
như 97% hoặc 99%. Trạng thái “chưa triển khai” phải ở trên slide.

**Lời nói**

“Giới hạn chính là chưa đủ toàn bộ baseline và ba seed trên cùng protocol
mới như bài báo. Caltech có validation chỉ 505 ảnh, đã dùng chọn checkpoint,
và Two-Stage chưa vượt No-context. Vì vậy nhóm chưa nhận kết quả khái quát
hay ý nghĩa thống kê.
Hướng tiếp theo có thể là cho phép nhiều context bổ sung nhau, hoặc học
khi nào nên bỏ context để tránh các ca gây nhiễu. Đây mới là giả thuyết,
chưa phải mô hình đã train. Nếu tiếp tục nghiên cứu, nhóm cần khóa phương
án trên validation, bổ sung đối chứng và seed rồi mới đánh giá test cuối,
thay vì thay cấu hình nhiều lần theo điểm test.”

**Chuyển ý**

“Nhóm xin kết luận bằng ba kết quả có thể kiểm chứng của đồ án.”

**Nguồn/điểm cần giữ đúng:** [S01], Limitations; [S04], [S07]. Multi-Context
chưa được code/train trong trạng thái này. Không gọi ý tưởng attention là
phát minh mới trên toàn bài báo: nhánh segmentation gốc đã có gated cross-attention.

### Slide 15. Kết luận

**Người nói:** TV3. **Thời gian:** 40 giây.

**Mục tiêu:** Chốt đúng giá trị và mức độ hoàn thành.

**Chữ trên slide**

> Tái hiện cơ chế TACS cho phân loại ảnh
>
> Phát triển Two-Stage và kiểm chứng bằng đối chứng
>
> Tích hợp demo hai dataset từ checkpoint thật
>
> Context có thể giúp, nhưng chưa bảo đảm vượt mô hình không context
>
> Cảm ơn cô và các bạn

**Hình/bố cục**

Ba dòng kết quả, một câu kết luận. Không thêm bảng số mới, lời hứa hiệu quả
hay dòng “Hoàn thành tái hiện 100%”.

**Lời nói**

“Qua đồ án, nhóm đã triển khai cơ chế chọn context của TACS cho bài toán
phân loại, phát triển truy xuất hai giai đoạn và tích hợp demo hai dataset
từ checkpoint thật. Các phép đối chứng cho thấy context có thể cải thiện
một số dự đoán, nhưng selector hiện tại chưa tạo ra ưu thế so với mô hình
No-context train riêng trên Caltech. Điểm nhóm rút ra là phải đánh giá
chất lượng ảnh hỗ trợ bằng tác dụng thực tế, không chỉ bằng độ giống
hay vài ví dụ thành công. Nhóm em cảm ơn cô và các bạn, và sẵn sàng
trả lời câu hỏi.”

**Nguồn/điểm cần giữ đúng:** Tổng hợp [S01]-[S12]. “Phát triển” là đã thay
đổi/triển khai phương pháp, không đồng nghĩa đã chứng minh tốt hơn paper.

## 3. Slide dự phòng cho hỏi đáp

Không đọc toàn bộ phần này trong 16 phút chính. Mỗi slide mở khi phù hợp câu hỏi.

### A1. Baseline và điểm tham chiếu trong paper

**Chữ trên slide**

| Phương pháp của tác giả | CIFAR-100 Acc (%) | Caltech-101 mAcc (%) |
|---|---:|---:|
| No-Context | 87,8 | 94,3 |
| Random Context Retrieval | 87,6 | 94,2 |
| Frozen DINO Similarity Retrieval | 88,4 | 94,7 |
| Feature-Averaged Context | 88,6 | 94,3 |
| TACS | 89,2 | 95,9 |

> Nguồn: Guo và cộng sự, CVPR 2026, Table 2. Kết quả trung bình 3 seed.
>
> Không đặt điểm validation protocol riêng của nhóm vào cùng bảng này.

**Lời giải thích khi được hỏi**

“Tác giả dùng bốn baseline và TACS. Feature-Averaged là gộp mềm đặc trưng
của các ứng viên top-k, không phải Two-Stage của nhóm. 89,2 là CIFAR-100
accuracy tham chiếu của bài báo. Caltech dùng mean-per-class accuracy,
tức tính accuracy từng lớp rồi lấy trung bình. Nhóm chưa có một bảng
mới đủ bốn baseline, cùng protocol và ba seed để nhận so sánh ngang hàng.”

**Nguồn:** [S01], Sec. 4.2-4.3 và Table 2.

### A2. Kết quả lịch sử không trộn với pilot mới

**Chữ trên slide**

> CIFAR-100, legacy_v1, seed train 44, lịch sử 100 epoch

| Phương pháp | Best validation (%) | Test checkpoint best (%) |
|---|---:|---:|
| No-context | 86,86 | 87,07 |
| Random | 39,28 | 39,26 |
| Similarity | 74,98 | 75,36 |
| TACS | 85,54 | 85,54 |
| Two-Stage | 75,94 | 75,68 |

> Run TACS lịch sử seed 42 khác: test 85,66%, 100 epoch
>
> Không dùng bảng này để tính mức tăng accuracy do role_v2

**Lời giải thích khi được hỏi**

“Đây là các run lịch sử có checkpoint, nhưng dùng bản task network cũ.
Khi thấy một số baseline kém bất thường, nhóm kiểm tra kiến trúc và tìm
thấy vấn đề phân biệt vai trò query/context. Nhóm giữ nguyên artifact cũ
để truy vết thay vì xóa hay sửa điểm. Tuy nhiên, giữa lịch sử và pilot mới
còn thay nhiều điều kiện, nên không được lấy chênh lệch accuracy để gán
toàn bộ công lao cho bản sửa role_v2.”

**Nguồn:** [S02]-[S03]. Chỉ mở nếu cô hỏi lịch sử thực nghiệm hoặc vì sao
không đưa mọi run vào bảng chính.

### A3. Chi tiết recipe và sai khác với paper

**Chữ trên slide**

| Thành phần | Paper CVPR 2026 | Triển khai/run nhóm |
|---|---|---|
| Backbone | DINOv3 ViT-S/16 | DINOv3 ViT-S/16 |
| Ảnh vào | 256 × 256 | 256 × 256 |
| Objective | Task + policy, τ 0,1 và λ 1,0 | Giữ dạng objective và hai giá trị này ở run mới đang trình bày |
| Epoch | 100 | CIFAR pilot 20, Caltech 100 |
| Batch classification | 64 | Microbatch 4, accumulation 16 |
| Phạm vi advantage | Chuẩn hóa trong batch | Đối chứng microbatch/effective-batch |
| Pool | 20% train theo mô tả paper | CIFAR 9.000 sau khi tách val, Caltech 6 ảnh/lớp |
| Seed | Trung bình 3 seed | Run mới đang báo cáo seed train 44 |
| Hardware | A100 | Cloud T4, replay local RTX 3050 |

**Speaker notes bổ sung**

Recipe nhóm: AdamW LR `1e-4`, weight decay `1e-4`, warmup 5, cosine theo
ngân sách đã cấu hình từ đầu. Không nói LR/WD/warmup cụ thể là paper quy định
nếu PDF không nêu. Seed chia tập 42 khác seed train 44.

Code nhóm dùng embedding bank cho candidate, không giữ toàn bộ ảnh trên GPU;
refresh bank theo epoch, candidate bank không có gradient, forward nạp
ảnh context hard top-1. Một proxy có giá trị forward bằng 0 nối đường
gradient từ task tới selected embedding. Đây là estimator/lựa chọn triển
khai riêng, không phải bằng chứng khớp hoàn toàn tối ưu của code tác giả.

Không coi gradient accumulation là tương đương mọi mặt với batch vật lý 64.
Không suy ra từ chữ “share” rằng task và selector chắc chắn cùng weights.

**Nguồn:** [S01], Sec. 3 và 4.2; [S04], [S07], [S09]. Bản PDF CVPR ghi
λ=1,0; không lấy λ=0,5 từ preprint trước đó để sửa lịch sử run.

### A4. Audit selector và thử cosine inference

**Chữ trên slide**

> CIFAR-100, 5.000 validation, task weights cố định theo từng checkpoint

| Chế độ inference | Microbatch checkpoint (%) | Effective-batch checkpoint (%) |
|---|---:|---:|
| Selector gốc, dot-product AMP | 87,86 | 87,82 |
| Selector cosine | 87,56 | 87,64 |
| Frozen retrieval top-1 | 87,94 | 87,76 |
| Random, trung bình 3 trial | 87,69 | 87,72 |
| Bỏ context, cùng task weights | 87,48 | 87,64 |

> Chỉ đổi inference sang cosine chưa cải thiện. Chưa thử train cosine từ đầu.

**Lời giải thích khi được hỏi**

“Dot-product chịu ảnh hưởng cả hướng và độ dài vector, còn cosine chuẩn
hóa độ dài. Nhóm kiểm tra thay cách chấm điểm learned selector bằng cosine
trên các features hiện có, nhưng accuracy giảm ở cả hai checkpoint.
Đây là bằng chứng rằng phép thay ở inference này chưa hiệu quả, không
chứng minh mọi mô hình được train bằng cosine đều kém. Hai cột là hai
checkpoint task khác nhau, không phải cùng một task model.”

**Nguồn:** [S05]-[S06]. Môi trường audit khác môi trường train, nên số đo
lại không khớp tuyệt đối số best do trainer ghi. Không sửa artifact gốc
để làm khớp. Không dùng timing audit làm benchmark tốc độ model.

### A5. Bằng chứng và trạng thái nghiệm thu

**Chữ trên slide**

> Config, split, source bundle, checkpoint và metrics được lưu để truy vết
>
> Caltech có đủ lịch sử 100 epoch cho hai mô hình
>
> Replay 505 validation, không cập nhật weights, không đánh giá test
>
> Demo lưu query, context, predictions và hash checkpoint trong JSON

**Lời giải thích khi được hỏi**

“Nhóm không chỉ giữ ảnh chụp phần trăm. Mỗi kết quả có cấu hình, checkpoint,
lịch sử và thông tin chia tập. Hash là mã kiểm tra nội dung file, giúp phát
hiện file đã đổi hoặc dùng nhầm model. Trong lượt replay, weights không
được cập nhật và test không được mở. Hash không tự chứng minh thuật toán
đúng, nhưng giúp truy vết kết quả về đúng artifact.”

**Nguồn:** [S04]-[S08], [S11]-[S12]. Hai best Caltech có hash khác nhau:
No-context `f2fa1b9f26c1...`, Two-Stage `27091fd7ca24...`. Dùng hash đầy đủ
trong báo cáo nguồn, không nhét 64 ký tự vào slide.

### A6. Reward và loss đầy đủ

**Chữ trên slide**

```text
r_i = CE(f(q_i), y_i) - CE(f(q_i, c_i), y_i)
A_i = (r_i - mean(r)) / (std(r) + epsilon)
L_policy = -mean(log pi(c_i | q_i) * A_i)
L_total = mean(CE(f(q_i, c_i), y_i)) + lambda * L_policy
```

> Reward detach khi dùng cho policy. Nhãn thật chỉ dùng trong train/đánh giá.

**Lời giải thích khi được hỏi**

“CE là cross-entropy, một loss phạt mô hình khi xác suất cho nhãn đúng còn
thấp. Reward so hai lượt của cùng task network, một có context và một
không context. Không dùng No-context train riêng để tính reward này.
Advantage chuẩn hóa reward trong phạm vi batch đang xét. Policy loss
tăng xu hướng chọn các action có advantage dương. Trong code nhóm,
temperature Gumbel điều khiển relaxation của đường gradient, không tự
thêm temperature vào công thức log policy. Khi inference không có các
nhãn này để tính reward.”

**Nguồn:** [S01], Eqs. 8-10; [S09]. Khi dựng PPT, nhập công thức bằng
equation editor, không dùng ảnh chụp đoạn code làm nội dung chính.

## 4. Bộ câu hỏi dự kiến và cách trả lời

### 1. Nhóm đã tái hiện được toàn bộ bài báo chưa?

“Chưa. Nhóm triển khai cơ chế TACS cho phân loại và có thực nghiệm, nhưng
chưa đủ 18 dataset, nhánh phân đoạn, toàn bộ baseline và ba seed trên
protocol mới. Nhóm ghi rõ phạm vi thay vì nhận tái hiện hoàn chỉnh.”

### 2. Phần cải tiến của nhóm nằm ở đâu?

“Nhóm phát triển truy xuất hai giai đoạn: frozen retrieval lấy top-32,
learned selector chọn context trong đó. Ngoài ra có sửa vai trò ảnh ở
task network và thử chuẩn hóa advantage theo effective batch. Đây là
thay đổi đã triển khai; chưa chứng minh chúng tạo ưu thế accuracy so với
đối chứng. Demo và lưu bằng chứng là phần kỹ thuật hỗ trợ kiểm chứng.”

### 3. Two-Stage chưa tốt hơn thì có được gọi là cải tiến không?

“Có thể gọi là phương án phát triển hoặc đề xuất cải tiến đã triển khai,
nhưng nhóm không gọi là cải tiến hiệu quả đã được chứng minh. Đánh đổi
accuracy, bộ nhớ và thời gian phải được đo riêng. Trong kết quả hiện có,
nhóm báo cáo trung thực rằng chưa tăng accuracy.”

### 4. Tại sao No-context lại cao hơn Two-Stage?

“Đây là quan sát của lượt chạy, chưa có thí nghiệm đủ để quy cho một
nguyên nhân duy nhất. Các khả năng cần kiểm chứng gồm context gây nhiễu,
top-32 bỏ sót ứng viên hữu ích, tối ưu selector chưa tốt và dữ liệu
validation nhỏ. Hai ảnh demo không đủ xác định nguyên nhân chính.”

### 5. Tại sao không train thêm epoch để accuracy tăng?

“Accuracy không tăng đơn điệu theo epoch. Trên Caltech, No-context tốt nhất
ở epoch 21 và Two-Stage ở epoch 54, dù cả hai đều chạy đủ 100 epoch. Validation
epoch 100 lần lượt 94,26% và 92,87%, thấp hơn best. Train thêm cần một giả
thuyết và recipe phù hợp, không bảo đảm tốt hơn.”

### 6. 95,64% của nhóm có vượt Caltech 95,9% trong paper không?

“Không. Ngay cả khi số gần nhau, đây vẫn là protocol và cách chọn checkpoint
khác. Paper báo mAcc trung bình ba seed, còn nhóm hiện báo validation
protocol riêng với một seed. Validation nhóm cân bằng 5 ảnh/lớp nên Acc
và mAcc trùng nhau trên tập này, nhưng không làm hai phép đánh giá tương đương.”

### 7. Context khác lớp query có phải lỗi lấy dữ liệu không?

“Không tự động là lỗi. Selector không bị buộc chọn cùng lớp; mục tiêu là
làm task tốt hơn. Tuy vậy, khác lớp cũng không tự chứng minh bổ sung hữu ích.
Phải kiểm tra loss hoặc dự đoán với và không có context trên cùng query.”

### 8. Nhãn context có được đưa vào mô hình để đoán query không?

“Không. Context đưa vào task là ảnh, không phải nhãn của ảnh phụ. Nhãn
được giữ để chấm điểm và phân tích sau dự đoán. Kho ứng viên được tách
khỏi validation và test.”

### 9. Bỏ context trong demo có phải baseline No-context của paper không?

“Không. Đó là một can thiệp trên model đã train với context. Hàng No-context
train riêng mới là đối chứng độc lập. Hai loại này có mục đích khác nhau
và được gắn nhãn riêng trên ứng dụng.”

### 10. Demo có nhận được mọi ảnh ngoài đời không?

“Không. Đây là phân loại đóng trong 100 hoặc 101 lớp đã định nghĩa. Ảnh ngoài
lớp vẫn bị ép chọn một lớp, ứng dụng chưa có cơ chế nhận diện unknown.
Ảnh tải riêng không có ground truth thì không tự đánh dấu đúng/sai.”

### 11. Tại sao cần Caltech khi đã có CIFAR?

“Ảnh CIFAR nguồn chỉ 32 × 32, resize lớn không tạo thêm chi tiết. Caltech có
ảnh nguồn lớn hơn để quan sát query và context rõ hơn. Nhóm train model
Caltech riêng, không chỉ đổi class map hay phóng ảnh CIFAR.”

### 12. Two-Stage nhanh hơn bao nhiêu lần?

“Hiện chưa có benchmark end-to-end được kiểm soát để đưa một hệ số tăng
tốc. Chỉ có thể nói learned selector xét 32 ứng viên thay vì toàn pool.
Retrieval, mã hóa query, cập nhật bank và task vẫn tốn thời gian; tốc độ
phải đo trên cùng phần cứng, batch và trạng thái cache.”

### 13. Accumulation 4 × 16 có giống hệt batch 64 của tác giả không?

“Không hoàn toàn. Nó cho 64 mẫu trước một bước optimizer, nhưng các phép
phụ thuộc batch như chuẩn hóa advantage có thể vẫn chỉ dùng 4 mẫu. Đó là
lý do nhóm làm pilot microbatch so với effective-batch; kết quả chưa cải thiện.”

### 14. Multi-Context đã chạy chưa?

“Chưa. Đó là hướng đề xuất cho bước sau, không nằm trong các checkpoint
hay bảng accuracy đang báo cáo. Cần triển khai và thực nghiệm mới mới
có thể kết luận.”

### 15. Cần làm gì để kết luận chắc hơn?

“Khóa recipe, bổ sung baseline còn thiếu, chạy nhiều seed trên cùng split
và ngân sách, đánh giá độ biến thiên cùng benchmark chi phí. Sau khi chốt
phương pháp mới mở test để báo cáo cuối, không điều chỉnh tiếp theo test.”

## 5. Chuẩn bị và chạy thử trước buổi bảo vệ

### 5.1. Mở demo

Chạy tại thư mục gốc repository bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101
```

Không cần train thêm. Máy phải có dependencies, checkpoint, frozen reference
và dữ liệu thật. Chỉ clone source không đủ vì artifact lớn không được commit Git.

Kiểm tra UI mà không chạy inference mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101 --check-ui
```

Nếu cần xuất lại một ví dụ, dùng **tên mới chưa tồn tại**:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tacs.showcase --dataset caltech101 --headless --validation-position 178 --output artifacts/caltech_demo/presentation_rehearsal_178.json
```

Lệnh headless này có inference, không train và không mở test. Lần tạo cache
đầu có thể chậm. Không cần replay lại toàn bộ 505 ảnh cho mỗi lần luyện nói.

### 5.2. Checklist trước buổi diễn tập

- [ ] Điền tên nhóm, tên giảng viên, tên học phần theo thông tin thật.
- [ ] Giữ đúng số slide và thứ tự 1-15, A1-A6 để dự phòng sau slide 15.
- [ ] Mỗi bảng điểm ghi rõ dataset, validation/test, seed và epoch budget.
- [ ] Slide 10 dùng 87,92/87,80, không lấy số audit thay vào không chú thích.
- [ ] Slide 11 ghi train 100 epoch nhưng best 21/54, không ghi model train 54 epoch.
- [ ] Slide 12 ghi cùng weights; hàng Random là một trial.
- [ ] Ứng dụng có đủ checkpoint/dataset ở máy sẽ mang lên lớp.
- [ ] Mở sẵn ứng dụng, chờ Sẵn sàng, chạy vị trí 0, 178, 169 một lần.
- [ ] Nếu định chuyển CIFAR, nạp và tạo cache trước, không làm lần đầu trên sân khấu.
- [ ] Mở được hai screenshot và JSON dự phòng khi tắt mạng.
- [ ] TV1, TV2, TV3 tập bàn giao đúng câu chuyển ý, bấm giờ theo bảng mục 1.
- [ ] Không cài lại, xóa cache hoặc thay checkpoint ngay sát buổi demo.

### 5.3. Hai trường hợp biên cần kiểm tra

**Ảnh ngoài các lớp đã học:** mở ảnh riêng không thuộc 101 lớp. Ứng dụng vẫn
buộc dự đoán một lớp, nên người thuyết trình phải nói rõ giới hạn, không
gọi đó là nhận diện đa dụng và không tự chấm đúng/sai nếu thiếu nhãn thật.

**Đổi dataset hoặc thiếu artifact:** chuyển Caltech sang CIFAR phải xóa
query/context/report cũ và nạp class map/checkpoint đúng. Nếu thiếu hoặc
không khớp checkpoint, ứng dụng cần báo lỗi, không được thay bằng model
chưa train. Không xóa/đổi tên artifact thật để thử lỗi ngay trước bảo vệ;
dùng kiểm thử hoặc môi trường bản sao khi cần.

### 5.4. Khi chỉ được trình bày 10 phút

Giữ slide 1-5, 7-8, 11-15; chuyển slide 6, 9, 10 sang dự phòng. Phải rút
lời nói theo ngân sách sau, không chỉ bỏ slide rồi giữ nguyên tốc độ:

| Slide giữ lại | Giây tối đa |
|---|---:|
| 1 | 20 |
| 2 | 30 |
| 3 | 30 |
| 4 | 45 |
| 5 | 50 |
| 7 | 40 |
| 8 | 40 |
| 11 | 50 |
| 12 | 45 |
| 13 | 150 |
| 14 | 50 |
| 15 | 30 |
| **Tổng** | **580** |

Tổng 9 phút 40 giây, còn 20 giây dự phòng. Demo chỉ mở hai vị trí 178/169,
bỏ bước chuyển CIFAR và xuất JSON live. Mỗi slide chỉ nói một ý chính.
Vẫn giữ thông tin “validation”, “cùng weights” và “No-context train riêng”;
không cắt các nhãn này để tiết kiệm thời gian.

## 6. Nguồn, số liệu và tài sản hình ảnh

Các đường dẫn bên dưới tính từ file này. Số liệu được chốt theo artifact đã
đọc và tài liệu nghiệm thu hiện tại; không lấy roadmap cũ làm trạng thái hoàn thành.

### S01. Bài báo gốc

[PDF CVPR 2026 do người dùng cung cấp](<../18. Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.pdf>).
Đã đối chiếu Sec. 3.1-3.2, Fig. 2, Sec. 4.1-4.3, Table 2 và phần Limitations.
Trang PDF 5 ghi τ = 0,1 và λ = 1,0; PDF 6 là bảng kết quả chính; PDF 8 là giới hạn.

### S02. Kết quả CIFAR lịch sử seed42

[CIFAR100_TACS_SEED42_RESULTS.md](CIFAR100_TACS_SEED42_RESULTS.md).
Chỉ dùng cho slide lịch sử A2, không dùng làm đối chứng trực tiếp của pilot role_v2.

### S03. Audit query/context và role_v2

[CONTEXT_ROLE_FIX.md](CONTEXT_ROLE_FIX.md).
Có bảng legacy seed44, nguyên nhân, thay đổi và phép thử cấu trúc trên checkpoint thật.

### S04. Pilot advantage normalization

[EFFECTIVE_BATCH_SEED44_RESULTS.md](EFFECTIVE_BATCH_SEED44_RESULTS.md).
Đọc đối chiếu trực tiếp hai metrics:

- `artifacts/effective_batch_pilot/kaggle_microbatch_seed44/advantage_microbatch_seed44/run/metrics.json`
- `artifacts/effective_batch_pilot/colab_effective_batch_seed44/advantage_effective_batch_seed44/run/metrics.json`

`best_val_acc`: 0,8792 và 0,8780. Cả hai `epochs=20`, `test_result=null`.
Source bundle chung: `1d85d0bbb17201c08e38b7428f720fccfe2859cc94cd0b256fac1103c7c9dce5`.
Không dùng source UI được thêm sau train để nhận là source cũ của thực nghiệm.

### S05. Audit selector trên CIFAR

[EFFECTIVE_BATCH_SELECTOR_AUDIT.md](EFFECTIVE_BATCH_SELECTOR_AUDIT.md).
Phân biệt best trainer 87,92/87,80 với replay 87,86/87,82.

### S06. Audit cosine inference

[EFFECTIVE_BATCH_COSINE_AUDIT.md](EFFECTIVE_BATCH_COSINE_AUDIT.md).
Chỉ là thay đổi inference, không có run train selector cosine từ đầu.

### S07. Caltech 100 epoch và provenance

[CALTECH101_FULL100_RESULTS.md](CALTECH101_FULL100_RESULTS.md).
Đã thu hồi/kiểm tra hai run 100 epoch. Source train:
`ad14eece024694224d7ae1eaa2544e532cd91038261ee2931eff4300406890a9`.

### S08. Hướng dẫn và số liệu demo hiện hành

[CALTECH101_DEMO_GUIDE.md](CALTECH101_DEMO_GUIDE.md),
[SHOWCASE_GUIDE.md](SHOWCASE_GUIDE.md).
Không dùng hướng dẫn demo policy-weight cũ làm mô tả checkpoint mặc định hiện tại.

### S09. Source giải thích thuật toán

[model.py](../src/tacs/model.py), [losses.py](../src/tacs/losses.py),
[train.py](../src/tacs/train.py), [retrieval.py](../src/tacs/retrieval.py).
Source hiện tại giúp giải thích cơ chế, còn provenance run phải theo bundle đã train.

### S10. Split CIFAR đã kiểm tra

[cifar100_seed42.json](../data/splits/cifar100_seed42.json) có 36.000 query,
9.000 pool và 5.000 validation. Test chính thức 10.000 ảnh là partition tách riêng.
Seed train44 không có nghĩa split seed cũng44.

### S11. Replay Caltech trên toàn validation

[summary.json](../artifacts/caltech_demo/rehearsal_validation/summary.json).
Đã kiểm tra status complete,505rows, `test_evaluated=false`, `weights_updated=false`.
Đếm lại đúng 476/474/477/470/483 cho selected/random/frozen/no-context-same/independent.
Đếm lại selected so với no-context-same: 8 giúp, 2 hại. Hai vị trí 169 và 178 khớp lời demo.

### S12. Hình và JSON dự phòng đã có

- [Context giúp, ảnh chụp GUI thật](../artifacts/caltech_demo/gui_context_helps.png).
- [Context gây sai, ảnh chụp GUI thật](../artifacts/caltech_demo/gui_context_hurts.png).
- [JSON GUI vị trí178](../artifacts/caltech_demo/gui_validation_178.json).
- [JSON trường hợp context giúp](../artifacts/caltech_demo/rehearsal_validation/context_helps_same_weights.json).
- [JSON trường hợp context gây sai](../artifacts/caltech_demo/rehearsal_validation/context_hurts_same_weights.json).

Giữ đúng tỷ lệ screenshot, không crop mất nhãn dataset/model hoặc tạo số liệu
minh họa giả. File `artifacts` thường không nằm trong Git, cần mang theo khi
chuyển máy hoặc dựng slide ở nơi khác.

## 7. Tiêu chí nghiệm thu bản slide được dựng từ kịch bản này

Bản PowerPoint về sau chỉ được coi là sẵn sàng khi đã mở/render và kiểm tra
không tràn chữ, bảng/hình rõ khi chiếu, công thức đọc được và số liệu khớp
các nguồn trên. Kịch bản này chưa phải bằng chứng PPT đã render hay demo
đã chạy thành công trong lượt soạn nội dung.

Nội dung chốt để cả nhóm thống nhất:

> Nhóm tái hiện cơ chế TACS cho phân loại, phát triển truy xuất hai giai đoạn
> và xây dựng demo từ dữ liệu/checkpoint thật. Thực nghiệm cho thấy context
> có thể giúp ở một số query nhưng chưa chứng minh Two-Stage vượt baseline
> độc lập. Những giới hạn còn lại được báo cáo rõ, không thay bằng lời hứa accuracy.
