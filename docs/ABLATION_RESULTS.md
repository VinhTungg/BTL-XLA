# Kết quả ablation TACS và Two-Stage

> Đây là kết quả lịch sử trên CIFAR-10, không phải pilot CIFAR-100/DINOv3
> role_v2 hiện tại. Xem [CONTEXT_ROLE_FIX.md](CONTEXT_ROLE_FIX.md) cho pilot
> seed 44 và [SELECTOR_UTILITY_AUDIT.md](SELECTOR_UTILITY_AUDIT.md) cho kiểm tra
> ích lợi context trên checkpoint tải từ Kaggle. Không gộp các protocol này.

Ngày chạy: 2026-10-24. Tất cả thí nghiệm dùng CIFAR-10 thật, seed 42,
batch size 64, CPU và cùng split 36.000/9.000/5.000/10.000.

## Phương pháp

- Sàng lọc trong 3 epoch, chọn cấu hình chỉ bằng validation accuracy.
- Không chạy test trong giai đoạn sàng lọc để tránh test leakage.
- Cấu hình thắng được chạy lại 5 epoch và chỉ khi đó mới đánh giá test.
- Standard encoder cho Two-Stage lấy từ No-context 5 epoch chính thức.

## Ablation loss TACS (3 epoch)

| Temperature | Policy weight | Best validation |
|---:|---:|---:|
| 0,1 | 1,0 (mặc định) | 37,02% |
| **0,1** | **0,0** | **37,26%** |
| 0,1 | 0,1 | 37,20% |
| 0,5 | 0,0 | 37,24% |
| 1,0 | 0,0 | 37,16% |

Temperature 0,1 tiếp tục phù hợp. Việc giảm policy weight có lợi hơn
thay đổi temperature, nhưng chênh lệch screening chỉ 0,24 điểm phần trăm.

## Ablation top-K Two-Stage (3 epoch)

Tất cả cấu hình dưới đây dùng temperature 0,1 và policy weight 0.

| Top-K | Best validation |
|---:|---:|
| 8 | 37,20% |
| 16 | 35,04% |
| **32** | **37,94%** |
| 64 | 37,18% |

K=32 là điểm tốt nhất trong các giá trị đã thử. Tăng K lên 64 không
cải thiện accuracy và làm selector phải xử lý nhiều candidate hơn.

## Xác nhận 5 epoch

| Mô hình | Cấu hình | Validation | Test |
|---|---|---:|---:|
| Similarity-context | baseline chính thức | 42,44% | 42,67% |
| TACS Full mặc định | temperature=0,1; policy=1,0 | 40,76% | 42,05% |
| **TACS hybrid nhẹ** | **temperature=0,1; policy=0,1** | **42,30%** | **43,32%** |
| **TACS task-loss-only** | **temperature=0,1; policy=0,0** | **42,14%** | **43,48%** |
| Two-Stage mặc định | K=32; temperature=0,1; policy=1,0 | 41,22% | 41,91% |
| Two-Stage hybrid nhẹ | K=32; temperature=0,1; policy=0,1 | 41,42% | 41,61% |
| Two-Stage task-loss-only | K=32; temperature=0,1; policy=0,0 | 41,58% | 42,30% |

TACS task-loss-only tăng 1,43 điểm phần trăm test so với TACS mặc định và
cao hơn Similarity-context 0,81 điểm. Two-Stage task-loss-only tăng 0,39
điểm so với Two-Stage mặc định, nhưng chưa vượt Similarity-context.
TACS hybrid nhẹ chỉ thấp hơn task-loss-only 0,16 điểm test và vẫn giữ
policy-gradient term. Ngược lại, policy weight 0,1 không cải thiện Two-Stage.

## Kiểm tra nhiều seed cho TACS hybrid nhẹ

Cấu hình temperature 0,1 và policy weight 0,1 được chạy 5 epoch với ba
seed huấn luyện; split dữ liệu vẫn cố định:

| Seed | Validation | Test |
|---:|---:|---:|
| 7 | 41,54% | 41,67% |
| 42 | 42,30% | 43,32% |
| 123 | 41,88% | 43,51% |
| **Trung bình ± SD** | **41,91% ± 0,38** | **42,83% ± 1,01** |

Test accuracy có độ lệch chuẩn 1,01 điểm, cho thấy TACS vẫn nhạy
với khởi tạo và lấy mẫu Gumbel.

### Đối chứng Similarity-context cùng seed

| Seed | TACS hybrid test | Similarity test | TACS - Similarity |
|---:|---:|---:|---:|
| 7 | 41,67% | 41,97% | -0,30 |
| 42 | 43,32% | 42,67% | +0,65 |
| 123 | 43,51% | 44,52% | -1,01 |
| **Trung bình ± SD** | **42,83% ± 1,01** | **43,05% ± 1,32** | **-0,22 ± 0,83** |

TACS hybrid nhẹ chỉ thắng ở một trong ba seed. Trung bình paired thấp hơn
Similarity-context 0,22 điểm, vì vậy không có bằng chứng TACS vượt baseline
một cách ổn định trong thí nghiệm hiện tại. Với chỉ ba seed, kết quả này
cũng chưa đủ cho kiểm định ý nghĩa thống kê mạnh.

## Diễn giải và giới hạn

`policy_weight=0` loại policy-gradient term khỏi hybrid objective; selector vẫn học
qua task loss và straight-through Gumbel-Softmax. Vì vậy kết quả 43,48% phải
được ghi là **task-loss-only ablation**, không thay thế im lặng kết quả TACS
hybrid mặc định.

Kết quả nhiều seed xác nhận policy weight 0,1 có thể đạt accuracy cao, nhưng
không ổn định hơn Similarity-context. Kết luận an toàn là cấu hình tuned cải
thiện TACS mặc định ở seed 42, chứ chưa chứng minh TACS thắng baseline trên
nhiều seed.

## Artifact

- Screening TACS: `outputs/ablation_3epoch/tacs/`
- Screening Two-Stage: `outputs/ablation_3epoch/two_stage/`
- Checkpoint xác nhận: `outputs/ablation_final/`
- Checkpoint nhiều seed: `outputs/multiseed/`
- Benchmark 5 mô hình gốc: `outputs/final_5epoch/`
