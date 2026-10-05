# Đồng bộ với P-026

`src/rav` là bản copy của `P-026/src/rav` (pipeline thuật toán, cùng contract `docs/design/CONTRACTS.md` của P-026).
Đường import giống P-026 (`src.rav...`) nên file chép qua lại được nguyên văn.

| | Trạng thái |
|---|---|
| Copy từ P-026 commit `192ff03` (2026-10-05) | `core/` (trừ `bulk_execution.py`), `components/` (trừ file mới bên dưới), `pipeline/{prepare,propose,evaluate}.py`, `store/field_store.py` — **không sửa** |
| Không copy (chỉ của sản phẩm) | `api/`, `commands/`, `service/`, `store/db.py`, `cli/` |
| Mới ở repo này, theo contract P-026 (có thể chép sang P-026) | `components/sources/bdd_images.py`, `components/decoders/image_file.py`, `components/features/clip_b32.py` (`emb.clip_b32x3`, `emb.clip_b32` — CONTRACTS §8), `components/objectives/text_match.py` (CONTRACTS §8), `components/selectors/farthest_history.py` (Algo 1), `components/judges/accept_all.py`, `components/routers/fixed_budget.py`, `components/exporters/csv_ids.py`, `pipeline/workflow.py` (`run_workflow`, CONTRACTS §12.4), `core/bulk_execution.py`, `store/packed_field_store.py` |

Test của P-026 chạy lại ở đây: `tests/p026/test_core.py`, `tests/p026/test_p2.py` (copy nguyên văn).

Khi P-026 đổi core/contract: chép lại các file "không sửa", chạy `pytest tests/`, cập nhật commit ở bảng trên.
Ghi chú gửi P-026: `core/execution.py` tách kết quả theo video bằng quét `unit_ids` cho từng video (O(N × số video)) —
`core/bulk_execution.py` là bản O(N) cùng hành vi; nên gộp vào `Execution` khi có nhiều video.
