# Thí nghiệm curation cho P-026 (BDD100K, YOLO nano, Kaggle)

Repo bổ trợ cho P-026 (RAV-26 data curation). Mục đích: chạy các thuật toán proposal của P-026 trên BDD100K, nhận tập frame chúng chọn,
finetune YOLO từ base model trên tập đó, rồi đo uplift so với model base và so với chọn ngẫu nhiên.

**Chỉ chạy trên Kaggle.** Repo không giả định môi trường local.

## 4 block, 2 pipeline

```
 pipeline proposal (src/rav — cùng contract với P-026)          harness đánh giá (src/topicx)
┌────────────────────────────┐   ┌──────────────────────────────┐   ┌─────────────────────────────┐   ┌──────────────────────┐
│ 1. Data management         │──►│ 2. Proposal nhiều vòng       │──►│ 3. Finetune + đánh giá      │──►│ 4. Hiển thị + lọc    │
│ splits, index ảnh (không   │   │ router → propose → judge     │   │ export.csv → manifest chốt  │   │ frame theo tiêu chí, │
│ nhãn), cache embedding     │   │ (không người duyệt) → export │   │ → nhãn BDD → train → test   │   │ lỗi model, tập chọn  │
│ 10_data_management         │   │ 20_proposal_rounds           │   │ 30_finetune_eval            │   │ 40_explore           │
└────────────────────────────┘   └──────────────────────────────┘   └─────────────────────────────┘   └──────────────────────┘
```

| Block | Notebook | Vào | Ra |
|---|---|---|---|
| 1. Data management | `10_data_management` | BDD100K | `art/splits/` (oracle giữ nhãn pool), `art/pool/<split>_index.csv`, `art/fields/` (embedding) |
| 2. Proposal nhiều vòng | `20_proposal_rounds` | output 10 | `art/proposals/<cách>/s<seed>_k<K>/{export.csv, history.jsonl, meta.json}` |
| 3. Finetune + đánh giá | `30_finetune_eval` | output 10 + 20, hoặc `export.csv` từ P-026 | `art/runs/`, `art/eval/report.md` |
| 4. Hiển thị + lọc | `40_explore` | output 10 (+ 20, 30) | bảng lọc, lưới ảnh, lỗi base model, lỗi đã sửa |

### Pipeline proposal = code của P-026

- `src/rav` là bản copy của `P-026/src/rav` (core, components, pipeline, field store). Đường import giống P-026 (`src.rav...`), nên file chép qua lại được nguyên văn. Chi tiết và commit đồng bộ: [`src/rav/SYNC.md`](src/rav/SYNC.md). Test lõi của P-026 chạy lại ở `tests/p026/`.
- Recipe khai báo ở [`configs/proposals.yaml`](configs/proposals.yaml) theo định dạng `RecipeConfig` của P-026: `reference`, `objectives`, `combiner`, `selector`. Hiện có:

  | Recipe | Gồm |
  |---|---|
  | `random`, `diversity` | recipe có sẵn của P-026 |
  | `text_topk` | `text_match` + top |
  | `algo1` | `text_match` top-K + `farthest_history` |
  | `algo1_batch` | như `algo1` nhưng không xét lịch sử |

- Vòng lặp nhiều vòng là `run_workflow` của CONTRACTS §12.4 (chế độ "Mô phỏng"), gồm router `fixed_budget`, judge `accept_all` (keep mọi frame được đề xuất, actor `algo:accept_all@1`) và feedback `keep_drop`.
- Component mới viết theo contract P-026, có thể chép sang P-026:
  - nạp dữ liệu: source `bdd_images`, decoder `image_file`;
  - feature `emb.clip_b32x3`, `emb.clip_b32` (CONTRACTS §8);
  - objective `text_match` (CONTRACTS §8);
  - selector `farthest_history` (Algo 1);
  - judge `accept_all`, router `fixed_budget`, exporter `csv_ids`.
- **Thêm thuật toán mới:** viết component vào `src/rav/components/<kind>/`, thêm recipe vào `proposals.yaml`, rồi chạy lại notebook 20 → 30. Hoặc chạy thuật toán trong P-026 và đưa `export.csv` vào notebook 30 (xem bên dưới).

### Harness = chỉ đánh giá

Harness không chứa logic chọn mẫu. Nó nhận `proposals/<tên>/s<seed>_k<K>/export.csv`, với cột `unit_id` dạng P-026 `"<tên ảnh>:0.000-0.000"` hoặc cột `image`. Các bước:
1. Kiểm export: id phải thuộc pool, không trùng, và không quá K id.
2. Nếu ít hơn K id: bù ảnh control từ base train để giữ cùng số update, và ghi lại phần thiếu (`shortfall`).
3. Chốt manifest bằng sha256, rồi mới mở nhãn BDD.
4. Finetune từ base model, đánh giá trên final test.

Đối chứng do harness tự tạo:
- **model base**;
- **`random` cùng seed và K** (nhánh tham chiếu cho uplift);
- **REPLAY_ONLY** (train thêm nhưng không có ảnh mới);
- tuỳ chọn **ORACLE_POSITIVE** (dùng nhãn ẩn).

Báo cáo gồm: uplift AP novel theo từng seed, mean/std, AP50-95 từng class (thang 0–100, có dòng BASE_MODEL), base mAP giữ được, và thống kê tập được chọn.

**Đưa kết quả từ P-026 vào:** tạo Kaggle Dataset có cấu trúc `art/proposals/<tên>/s<seed>_k<K>/export.csv` (với smoke là `art_smoke/...`) rồi gắn vào notebook 30. Cell setup tự copy vào working.

## Chống rò nhãn

- `splits/oracle/` là nơi duy nhất chứa nhãn và thuộc tính ảnh của pool. Pipeline proposal chỉ đọc `pool/<split>_index.csv` (tên ảnh + đường dẫn).
- Block 1 (tính field) và block 2 (propose) chạy trong `no_oracle_access()`: audit hook chặn mọi `open` hoặc `listdir` tới đường dẫn chứa `oracle`.
- Có test kiểm toàn bộ `src/rav` không import `topicx`, oracle hay pandas.
- Manifest được chốt bằng sha256, không sửa được. `OracleStore.reveal` chỉ mở nhãn của ảnh có trong manifest đã chốt, và mọi truy cập được ghi log.
- Chia tập theo ride (`<ride>` trong `<ride>-<clip>.jpg`):
  - test chỉ lấy từ BDD val;
  - dev, pool và base lấy từ BDD train;
  - kiểm md5 exact duplicate giữa các tập.
- Câu topic và τ được chọn trên dev bằng luật khai báo trước (`proposals/dev_choice.json`). Final test chỉ dùng ở notebook 30, và ở notebook 40 sau khi đã đánh giá xong.

## Kịch bản base model

| `SCENARIO` | Base model | Finetune | Artifact |
|---|---|---|---|
| `missing` (mặc định) | train trên base train BDD, chưa có novel class | mở rộng head thêm novel, giữ weights class cũ | `art/` |
| `weak` | train đủ class, novel chỉ có `weak_novel_images` ảnh (mặc định 100) | finetune thẳng | `art_weak/` |

Cả hai đều bắt đầu từ `yolo11n.pt` (COCO) rồi train trên dữ liệu BDD của ta.

## Chạy trên Kaggle

1. Import notebook từ GitHub (File → Import Notebook): `notebooks/10_…`, `20_…`, `30_…`, `40_…`.
2. Ở mỗi notebook: **Add Input** `solesensei/solesensei_bdd100k` (đường dẫn đã cấu hình sẵn ở `configs/exp.yaml`), bật **Internet** và GPU.
3. Chạy theo thứ tự 10 → 20 → 30. Notebook sau gắn **output** của notebook trước làm input. Notebook 40 dùng bất cứ lúc nào sau 10.
4. Chạy `SMOKE = True` trước (tập nhỏ, 1 epoch, artifact riêng ở `art_smoke/`). Khi pass, chuyển `SMOKE = False` và pin `COMMIT`.
5. Notebook 30 dừng trước khi vượt `kaggle.time_budget_h`. Khi đó: Save Version, gắn output vừa xong làm input rồi chạy lại. Mọi bước đã xong đều được bỏ qua.

Mỗi notebook:
- **mở đầu:** chạy `pytest` (CPU, vài giây), ghi `env.json` (phiên bản thư viện, commit, GPU) và chạy health check dataset (cây thư mục, số ảnh, định dạng nhãn, ảnh mẫu có box);
- **kết thúc:** tạo `art_<stage>_results.zip` (json/csv/md) ở tab Output để gửi phân tích.

## Kết quả đã có

| Ngày | Notebook | Thiết lập | Kết quả chính | Ghi chép |
|---|---|---|---|---|
| 2026-10-04 | legacy NB00 | missing, novel = bus, K = 250, 3 seed | RETRIEVAL: AP50-95 bus 0,212 so với RANDOM 0,096 (+0,115 ở mọi seed); base mAP không giảm. Chưa tách được tác động của lượng supervision | [docs/results/2026-10-04_nb00_missing_bus_k250.md](docs/results/2026-10-04_nb00_missing_bus_k250.md) |
| 2026-10-05 | legacy NB06 | Algo 1, missing, bus, K ∈ {128, 256}, 25/30 run | TOPB (0,212) > ALGO1_BATCH (0,199) > ALGO1 (0,191) > RANDOM (0,094) ở K = 256. Diversity giảm positive yield 20–25 điểm % vì tập ứng viên top-500 rộng (τ không lọc ảnh nào), nên giả thuyết bị bác ở cấu hình này | [docs/results/2026-10-05_nb06_algo1_missing_bus.md](docs/results/2026-10-05_nb06_algo1_missing_bus.md) |

Các kết quả này chạy bằng `notebooks/legacy/` (commit `6fac1ab`, trước khi tách 4 block). Notebook legacy đã cố định commit đó nên vẫn chạy lại được.
Trong pipeline mới, các nhánh tương ứng là: `RANDOM` → `random`, `RETRIEVAL`/`ALGO1_TOPB` → `text_topk`, `ALGO1` → `algo1`, `ALGO1_BATCH` → `algo1_batch`, `DIVERSITY` → `diversity`. Recipe `diversity` của P-026 dùng novelty DINOv2 + `greedy_nms`, khác bản port cũ.

## Cấu trúc

```
configs/exp.yaml            tham số harness (splits, YOLO, eval, đường dẫn BDD) + override smoke
configs/proposals.yaml      recipe proposal (định dạng P-026), câu topic, luật τ, K, B
src/rav/                    pipeline proposal = copy P-026 + component mới (xem SYNC.md)
src/topicx/common.py        config, artifact trên Kaggle, manifest chốt, env log, sync input
src/topicx/data.py          đọc BDD100K, EDA, chia tập + kiểm tra, xuất YOLO
src/topicx/oracle.py        OracleStore + no_oracle_access
src/topicx/proposals.py     cầu nối block 1–2: index, Context rav, cache field, run_recipe, chọn câu/τ trên dev
src/topicx/pipeline.py      harness: splits, base model, import_exports, train, evaluate, bundle
src/topicx/explore.py       block 4: frame_table, model_errors, selection_frames, show, compare_errors
src/topicx/yolo.py          mở rộng head, train, AP theo tên class, recall
src/topicx/metrics.py       chỉ số tập chọn, tổng hợp theo cặp seed
tests/                      harness + component mới + tests/p026 (lõi P-026)
notebooks/10..40            4 block
notebooks/legacy/           notebook cũ (commit 6fac1ab)
docs/                       đặc tả Algo 1, kết quả
```

## Giới hạn đã biết

- Annotation là mô phỏng từ ground truth có sẵn. Judge `accept_all` keep mọi frame nên chưa mô phỏng người duyệt.
- COCO pretrained đã có bus, motorcycle, person và bicycle. "Novel" chỉ có nghĩa là mới với base model của tác vụ này.
- BDD100K images chỉ có 1 keyframe mỗi video (`video` của P-026 ở đây là 1 ảnh), nên không kiểm được lợi ích chống trùng lặp từ video.
- Feature `emb.dinov2_s` của P-026 chạy trên CPU, batch 8: block 1 chậm hơn CLIP. Giữ nguyên để đồng bộ với P-026.
- `metadata` cảnh (timeofday, weather) của pool đang bị ẩn khỏi proposer (nằm trong oracle). Nếu muốn lọc theo `tag.*` (CONTRACTS §8.0), cần quyết định trước metadata nào được coi là có sẵn lúc chọn.
- **Cấu trúc 4 block mới chưa chạy trên Kaggle.** Code, test và notebook mới qua kiểm tra cú pháp. Smoke run là bước xác minh đầu tiên.
