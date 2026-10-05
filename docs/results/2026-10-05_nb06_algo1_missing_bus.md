# Kết quả NB06 — Algo 1, kịch bản missing, novel = bus, ngân sách 128 và 256

Ghi nhận: 2026-10-05. Nguồn: `art_algo1_results.zip` của `notebooks/06_algo1_eval.ipynb` (Kaggle, 2× T4), code commit `c443db5`.
Môi trường: `SMOKE = False`, Ultralytics 8.3.253, open_clip 3.3.0, torch 2.11.0+cu128.
Splits và base model giống hệt NB00 (2026-10-04): test 5.001 ảnh (799 instance bus), dev 3.000, pool 20.006 (12,8% positive), base 15.001 (không có bus).
Đặc tả thuật toán: [`docs/algo_1_semantic_retrieval_history_diversity.md`](../algo_1_semantic_retrieval_history_diversity.md).

**Hoàn tất 25/30 run.** Thiếu `s2_k256_*` (5 run): session dừng trước khi vượt ngân sách giờ. Vì vậy ở K = 256 chỉ có seed 0 và 1.

## Kiểm tra hợp lệ

- `max_diff_vs_base = 0.0` ở cả 3 seed: mở rộng head giữ nguyên output của 8 class cũ.
- Prompt được chọn trên dev là `a city bus in traffic` (dev AP 0,379; ensemble 0,369).
- τ chọn theo luật `dev_f1`: τ = 0,259. Trên dev, ngưỡng này cho F1 0,369, **precision 0,355**, recall 0,383.
- Thời gian mỗi run finetune đo được khoảng 1.100–1.150 s (khoảng 19 phút), thấp hơn ước tính 27 phút.
- Hội tụ: `cls_loss` của finetune vẫn đang giảm ở epoch 20 (0,857 → 0,767), và base model vẫn tăng ở epoch 30 (mAP dev 0,2023 → 0,2076). Các nhánh dùng cùng lịch train nên so sánh vẫn công bằng, nhưng giá trị AP tuyệt đối có thể đang thấp hơn mức hội tụ.

## Tập dữ liệu do mỗi cách tạo ra (pool, trung bình theo seed)

| K | Nhánh | precision@K | instance bus | nn_cos_mean | min_pair_d / vòng | min_hist_d / vòng |
|---|---|---|---|---|---|---|
| 128 | RANDOM | 0,135 | 25 | 0,702 | | |
| 128 | ALGO1_TOPB | **0,914** | **181** | 0,726 | | |
| 128 | ALGO1_BATCH | 0,695 | 130 | 0,680 | | |
| 128 | ALGO1 | 0,703 | 131 | **0,661** | | |
| 256 | RANDOM | 0,143 | 42 | 0,721 | | |
| 256 | ALGO1_TOPB | **0,836** | **326** | 0,741 | 0,047 | 0,032 |
| 256 | ALGO1_BATCH | 0,637 | 251 | 0,702 | **0,112** | 0,049 |
| 256 | ALGO1 | 0,668 | 243 | **0,690** | 0,090 | **0,083** |

Proxy diversity hoạt động đúng thiết kế. ALGO1 tối đa khoảng cách tới lịch sử, còn ALGO1_BATCH tối đa khoảng cách trong batch. Cả hai đều giảm độ trùng lặp so với TOPB. Cái giá phải trả là **positive yield giảm 20–25 điểm phần trăm**.

## Detection trên final test (AP50-95 bus)

| K | Nhánh | seed 0 | seed 1 | seed 2 | mean | base mAP Δ so với base model |
|---|---|---|---|---|---|---|
| 128 | RANDOM | 0,114 | 0,017 | 0,087 | 0,072 ± 0,050 | −0,0017 |
| 128 | ALGO1_TOPB | 0,195 | 0,191 | 0,189 | **0,192 ± 0,003** | +0,0012 |
| 128 | ALGO1_BATCH | 0,165 | 0,164 | 0,150 | 0,160 ± 0,008 | +0,0007 |
| 128 | ALGO1 | 0,156 | 0,153 | 0,145 | 0,151 ± 0,006 | −0,0000 |
| 256 | RANDOM | 0,104 | 0,084 | — | 0,094 | −0,0020 |
| 256 | ALGO1_TOPB | 0,211 | 0,212 | — | **0,212** | +0,0026 |
| 256 | ALGO1_BATCH | 0,197 | 0,200 | — | 0,199 | +0,0009 |
| 256 | ALGO1 | 0,187 | 0,195 | — | 0,191 | +0,0019 |
| cả hai K | REPLAY_ONLY | 0 | 0 | 0 | 0 | −0,0004 / −0,0010 |

Thứ tự **TOPB > BATCH > ALGO1** đúng ở mọi seed và cả hai mức ngân sách. Cách chia theo ngày/đêm và recall theo kích thước vật thể cho cùng thứ tự đó.

## So sánh theo cặp (§9 của đặc tả)

| Câu hỏi | Hiệu AP50-95 bus, K = 128 | K = 256 |
|---|---|---|
| Lịch sử có ích không? (ALGO1 − ALGO1_BATCH) | −0,008 | −0,008 |
| Diversity có ích không? (ALGO1_BATCH − ALGO1_TOPB) | −0,032 | −0,013 |
| Retrieval + τ có ích không? (ALGO1_TOPB − RANDOM) | +0,119 | +0,118 |
| Algo 1 so với random (ALGO1 − RANDOM) | +0,079 | +0,097 |

## Kết luận

1. **Retrieval có ích:** có. Lặp lại đúng kết quả của NB00: TOPB@256 đạt 0,212 so với RETRIEVAL@250 0,212.
2. **Diversity (trong batch hoặc theo lịch sử) cải thiện học novel class ở cùng ngân sách:** **không**. Kết quả bác bỏ giả thuyết của baseline theo đúng tiêu chí đặc tả đã đặt trước ở §9: "redundancy giảm nhưng AP không cải thiện, positive yield giảm quá nhiều".
3. **Nguyên nhân có khả năng nhất:** τ quá lỏng. Theo luật `dev_f1`, τ chỉ đạt precision 0,355 trên dev, nên tập ứng viên C_t (500 ảnh) chứa nhiều ảnh không có bus. Farthest-first lại ưu tiên các ảnh khác biệt, đúng rủi ro "ưu tiên outlier" mà đặc tả §10 đã nêu. Kết quả là chọn nhiều ảnh negative hơn TOPB.
4. **Theo từng instance, diversity cũng chưa thấy lợi ích rõ.** ALGO1_BATCH@256 có 251 instance và đạt 0,199; TOPB@128 có 181 instance mà đã đạt 0,192. Phép so sánh này không kiểm soát được biến nào khác, nên chưa phải bằng chứng; cần nhánh có số instance khớp nhau.

## Lưu ý khi trích dẫn

- K = 256 chỉ có 2 seed. RANDOM ở K = 128 dao động rất mạnh (seed 1 chỉ đạt 0,017).
- Algo 1 và TOPB tất định, nên std của các nhánh này chỉ phản ánh seed train.
- Annotation là mô phỏng từ ground truth. COCO pretrained đã có class bus.

## Việc tiếp theo

1. Chạy nốt `s2_k256_*`: gắn output version này làm input rồi chạy lại notebook (5 run, khoảng 1,6 h).
2. Thử τ chặt hơn, với luật khai báo trước. Ví dụ τ nhỏ nhất sao cho precision trên dev ≥ 0,8, hoặc chọn C_t là top-B×m theo relevance. Mục tiêu là xem diversity còn bị thiệt về yield không khi C_t chủ yếu là positive.
3. Biến thể cân bằng relevance và diversity, ví dụ điểm `r(x) + λ·δ(x)` (đặc tả §10 đã liệt kê là hướng nghiên cứu sau).
4. Nhánh khớp số instance (`RETRIEVAL_MATCHED` hoặc tương tự) để tách hiệu ứng "nhiều bus hơn" khỏi "ảnh tốt hơn".
