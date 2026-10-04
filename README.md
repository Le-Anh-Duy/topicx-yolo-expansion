# Topic retrieval để mở rộng class cho YOLO nano (BDD100K, Kaggle)

Thí nghiệm cho P-026 (RAV-26 data curation): người dùng đặt topic là một class còn thiếu → hệ thống chọn K ảnh từ
candidate pool chưa có nhãn → annotate (mô phỏng từ ground truth) → mở rộng head YOLO11n → train tiếp → đo AP class mới
và mức giữ class cũ.

**Giả thuyết:** truy hồi theo topic thu thập dữ liệu giúp YOLO nano học class mới tốt hơn lấy mẫu ngẫu nhiên.
Kết luận tách 3 câu: (1) thu được nhiều positive hơn? (2) detection tốt hơn khi cùng K ảnh? (3) còn tốt hơn khi
lượng supervision novel tương đương?

**Chỉ chạy trên Kaggle.** Repo không giả định môi trường local.

## Đánh giá uplift cho một thuật toán proposal: `notebooks/00_uplift_pipeline.ipynb`

Notebook này chạy cả luồng trong một chỗ: splits → base model → proposal → nhãn BDD → finetune → uplift so với RANDOM.

- **Pool** là ảnh BDD **train** chưa từng dùng để train YOLO. **Final test** là BDD **val**. Dev và base cũng lấy từ BDD train, chia theo group nên không giao nhau.
- **Cắm thuật toán mới:**
  - Cách A: viết `fn(pool, k, seed) -> list[id]` rồi thêm vào `MY = {"TÊN": fn}`. Proposer chạy trong guard chặn đọc nhãn pool và chỉ thấy `pool.ids`, `pool.paths`, `pool.topic`, `pool.clip()`, `pool.dino()`, `pool.topic_scores()`.
  - Cách B: chạy thuật toán ở nơi khác trên `splits/pool_ids.txt`, ghi danh sách xếp hạng ra `proposals/<tên>.txt` hoặc `.csv` (cột `image`), upload thành Kaggle Dataset rồi gắn vào. Nhánh tự xuất hiện với tên `EXT_<tên>`.
- Đầu ra của proposer được kiểm: đúng K id, không trùng, thuộc pool. Sau đó manifest được chốt.
- **Chạy tiếp khi hết giờ:** Save Version, gắn output đó làm input rồi chạy lại. Mọi bước đã xong (splits, base, manifest, run, eval) đều được bỏ qua. Muốn thử thuật toán mới trên cùng splits và base model thì gắn output cũ và thêm proposer: chỉ nhánh mới phải train.
- Mặc định: RANDOM, RETRIEVAL, REPLAY_ONLY, K = 250, 3 seed. Kết quả ở `art/eval/report.md`:
  - bảng 1: tập dữ liệu mỗi cách tạo ra;
  - bảng 2: uplift AP novel so với RANDOM theo từng seed;
  - bảng 3: mean/std và mức giữ base mAP;
  - bảng 4: từng run.

Các notebook 01–05 bên dưới là cùng pipeline đó tách thành từng bước có gate (gọi chung `src/topicx/pipeline.py`).

## Nhánh (cùng K ảnh, cùng expanded init và replay set trong mỗi seed)

| Nhánh | Chọn K ảnh | Dùng nhãn ẩn? |
|---|---|---|
| RANDOM | ngẫu nhiên từ pool | không |
| DIVERSITY | recipe P-026: greedy min_dist trên DINOv2-S, thứ tự ngẫu nhiên, không dùng topic | không |
| RETRIEVAL | top-K theo điểm CLIP với topic | không |
| RETRIEVAL_DIVERSE | greedy min_dist (CLIP) trong top 3K theo topic | không |
| ORACLE_POSITIVE | ngẫu nhiên trong ảnh có novel (tham chiếu, không phải upper bound) | **có** |
| REPLAY_ONLY | K ảnh base train khác (không có novel), đo riêng tác động của mở rộng head + train tiếp | không |
| RETRIEVAL_MATCHED (tuỳ chọn) | prefix ranking đủ số novel instance của RANDOM, bù ảnh control cho đủ K | **có** |

Mọi nhánh train trên `replay (5000) + K` ảnh, cùng epoch/batch/augmentation/schedule, nên có cùng số optimizer update.
Ảnh được chọn giữ nhãn đầy đủ base + novel, kể cả ảnh negative.

## Chống rò nhãn

- `splits/oracle/` là nơi duy nhất chứa nhãn và thuộc tính ảnh của pool. Selector chỉ đọc `splits/pool_ids.txt`.
- Embed và selection chạy trong `no_oracle_access()`: audit hook chặn mọi `open` hoặc `listdir` tới đường dẫn chứa `oracle`.
- `select.py` và `embed.py` không import oracle, data hay pandas (có test kiểm).
- Manifest được chốt bằng sha256 và không sửa được. `OracleStore.reveal` chỉ mở nhãn của ảnh có trong manifest đã chốt. Mọi truy cập oracle được ghi vào `oracle_log.jsonl`.
- Chia tập theo group (`<ride>` trong tên ảnh `<ride>-<clip>`). Test chỉ lấy từ BDD val; dev, pool và base lấy từ BDD train. Test, dev và pool lấy nguyên group, base bỏ ảnh có novel. Kiểm tra cả md5 exact duplicate giữa các tập.
- Prompt được chọn trên dev bằng luật khai báo trước (AP ranking cao nhất). Final test chỉ mở ở NB5.

## Chạy trên Kaggle

1. Repo: https://github.com/Le-Anh-Duy/topicx-yolo-expansion (public). Run thật: pin `COMMIT` ở cell đầu mỗi notebook. Bật **Internet** cho notebook.
2. Tạo 5 Kaggle Notebook từ `notebooks/*.ipynb`. Mỗi notebook gắn Kaggle Dataset BDD100K (cần `images/100k/{train,val}` và `bdd100k_labels_images_{train,val}.json` hoặc `det_{train,val}.json`; tự dò đường dẫn), cộng với output của các notebook trước:

| Notebook | Gắn thêm output | GPU | Gate |
|---|---|---|---|
| 01_eda_splits | — | không | xem EDA, đặt `NOVEL` + `REASON` |
| 02_base_model | 01 | có | AP từng base class trên dev hợp lý; xem ước tính giờ GPU cho NB4 |
| 03_retrieval_selection | 01 | có | dev AP của prompt, precision@K so với RANDOM |
| 04_expand_train_branches | 01, 02, 03 (+ version trước của 04) | có | `max_diff_vs_base` < 1e-4 trong `expand_log.json` |
| 05_evaluate | 01, 02, 03, 04 (version cuối) | có | điền kết luận 3 câu |

3. Chạy toàn bộ với `SMOKE = True` trước (tập nhỏ, 1 epoch, artifact riêng trong `art_smoke/`). Smoke pass rồi mới đặt `SMOKE = False`, pin `COMMIT`, chạy lại từ NB1.
4. NB4 (và NB00) dừng trước khi vượt `kaggle.time_budget_h`. Khi đó: Save Version, gắn output vừa xong làm input của chính notebook đó rồi chạy lại. Cell setup copy mọi artifact từ input sang output (`sync_inputs`), nên version mới chứa đủ kết quả cũ và mới.

Mỗi notebook chạy `pytest` (CPU, vài giây) ở cell đầu và ghi phiên bản thư viện, commit, GPU vào `art/<stage>/env.json`.

## Ngân sách (ước tính thô, chưa đo)

Mặc định có 3 seed × 2 K × 6 nhánh = 36 run fine-tune trên khoảng 5–6k ảnh × 20 epoch, cộng 1 base (15k ảnh × 30 epoch).
Thời gian đo thật ở NB2 (`sec_per_img_epoch`), NB2 in ước tính cho NB4. Nếu vượt quota GPU tuần (khoảng 30h), giảm
`seeds`, `ks` hoặc `finetune.epochs` trong `configs/exp.yaml`.

## Cấu trúc

```
configs/exp.yaml          mọi tham số + override smoke
src/topicx/common.py      config, tìm artifact trong /kaggle/working và /kaggle/input, manifest chốt, env log
src/topicx/data.py        đọc BDD100K, EDA, chia tập + kiểm tra, xuất YOLO (symlink ảnh)
src/topicx/oracle.py      OracleStore + no_oracle_access
src/topicx/select.py      random / top-K / greedy diverse (port P-026 greedy_nms), không nhãn
src/topicx/embed.py       OpenCLIP 3 crop, DINOv2-S
src/topicx/yolo.py        mở rộng head, train, AP theo tên class, recall ở conf cố định
src/topicx/metrics.py     chỉ số retrieval, tổng hợp theo cặp seed
src/topicx/pipeline.py    các bước pipeline + Pool / PROPOSERS / from_ranked_file (điểm cắm proposal)
tests/test_protocol.py    kiểm tra protocol bằng dữ liệu tổng hợp
notebooks/00              cả pipeline uplift trong một notebook
notebooks/01..05          cùng pipeline, tách bước có gate
```

## Giới hạn đã biết

- Annotation là mô phỏng từ ground truth có sẵn, chưa đo chất lượng hay thời gian của người annotate.
- `yolo11n.pt` pretrained trên COCO, mà COCO đã có bus, motorcycle, person và bicycle. "Novel" chỉ có nghĩa là mới với base model của tác vụ này.
- Protocol được dùng dữ liệu cũ (replay), nên đây không phải incremental learning không có dữ liệu cũ.
- BDD100K 100k images chỉ có 1 keyframe mỗi video, nên vốn ít trùng lặp. Lợi ích của diversity có thể nhỏ hơn so với frame lấy từ video như trong P-026.
- Base train bỏ ảnh có novel nên phân phối base class bị lệch. Mức lệch xem ở bảng `removal_cost` trong NB1.
- RETRIEVAL và RETRIEVAL_DIVERSE tất định theo seed, nên phương sai của hai nhánh này chỉ đến từ seed train.
- AP theo kích thước vật thể chưa làm. Theo kích thước chỉ có recall ở conf cố định. AP theo day/night chỉ làm khi đủ `min_instances_subset`.
- **Chưa chạy lần nào.** Tests và notebook mới được kiểm cú pháp. Smoke run trên Kaggle là bước xác minh đầu tiên.
- Version Ultralytics và OpenCLIP chưa pin chính xác. `expand_head` phụ thuộc cấu trúc Detect (`cv3`) và sẽ báo lỗi rõ nếu cấu trúc khác.
