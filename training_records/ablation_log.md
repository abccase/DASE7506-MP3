# DASE7506 MP3 SQL Repair Agent — Controlled Ablation & Compliance Log

Maintained independently by `verifier` in `training_records/`.
All design, model, LoRA, and harness ablations were driven strictly by **`train` (480) + `validation` (120)**; official **`test` (240)** reproductions ran ONLY on approved frozen milestones (`v13`, `v14`).

## Promoted Final Submission Artifacts
- **Final Source Export (11 files, `0.176 MiB` ≤ `16 MiB`):** [`submission/`](submission/) *(old baseline preserved in [`submission_v7_backup/`](submission_v7_backup); Round 1 archive preserved in [`submission_v13/`](submission_v13))*
- **Final Clean 2-File LoRA Adapter (`77.05 MiB` ≤ `512 MiB`):** [`adapter/`](adapter/) (`adapter_config.json` + `adapter_model.safetensors`, `Qwen/Qwen3-0.6B` rev `c1899de289a04d12100db370d81485cdf75e47ca`, strictly LoRA A/B keys only)
- **Official Final Test Outputs (`190/240 = 79.17%`, `31/31 PASS`, `0 WARN`, `0 FAIL`):**
  - Predictions: [`outputs/test.json`](outputs/test.json) *(copy of `test-v14.json`)*
  - Official Score JSON: [`outputs/test-score.json`](outputs/test-score.json) *(copy of `test-v14-score.json`)*
  - Complete LLM Usage Log: [`outputs/test-llm.jsonl`](outputs/test-llm.jsonl) *(copy of `test-v14-llm.jsonl`)*
  - Compliance Audits: [`test_v14_compliance.json`](training_records/test_v14_compliance.json), [`final_submission_compliance.json`](training_records/final_submission_compliance.json), [`test_v14_audit.json`](training_records/test_v14_audit.json)

---

## 1. Official `test` (N=240) Milestone Reproduction Table (SOLO 4 CPU Threads, 3-Instance Grading)

| System | Route | 3-Inst Acc | Δ vs `v7` (+gain/-loss) | Δ vs No-LLM | Pub Pass | FP | Pub Fail | `rows==0` (49) | `rows==1` (117) | `rows>1` (74) | `ordered=T` (60) | `ordered=F` (180) | Tools (mean/max) | LLM Calls (tasks / max / max_tok) | Predict / Score Time | Peak RSS | Compliance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `test-baseline` (draft) | `no_llm` | 88/240 (36.67%) | -96 | — | — | — | — | — | — | — | — | — | 0.00 / 0 | 0 | 0.0s / 37.1s | — | PASS (starter) |
| `test-v7` (prior ref) | `live_llm` | 184/240 (76.67%) | 0 (ref) | — | 215/240 (89.6%) | 31 | 25 | 28/49 (57.1%) | 96/117 (82.1%) | 60/74 (81.1%) | 49/60 (81.7%) | 135/180 (75.0%) | 3.65 / 8 | *unlogged* (LB ≥89 / 74t) | 1224.3s / — | — | GAPS (no log; 3 loaders in old `submission/`) |
| `test-v13-nomodel` (`A3`) | `no_llm` | 177/240 (73.75%) | -7 (+6/-13) | 0 | 209/240 (87.1%) | 32 | 31 | 27/49 (55.1%) | 89/117 (76.1%) | 61/74 (82.4%) | 45/60 (75.0%) | 132/180 (73.3%) | 4.28 / 8 (`exec=90`) | 0 | 29.8s / 47.9s | 0.11 GiB | **PASS (`7/7`, 0W/0F)** |
| `test-v14-nomodel` (`A3-R2`) | `no_llm` | 178/240 (74.17%) | -6 (+7/-13) | 0 | 208/240 (86.7%) | 30 | 32 | 29/49 (59.2%) | 89/117 (76.1%) | 60/74 (81.1%) | 46/60 (76.7%) | 132/180 (73.3%) | 4.38 / 8 | 0 | **27.4s / 41.2s** (`68.6s` ≤ 180s) | **0.11 GiB** (≤ 2 GiB) | **PASS (`7/7`, 0W/0F)** |
| `test-v13` (`submission_v13` + `v13_r32`) | `live_llm` | 189/240 (78.75%) | +5 (+10/-5) | +12 (+12/-0 vs `A3`) | 220/240 (91.7%) | 31 | 20 *(0 exec err)* | 30/49 (61.2%) | 96/117 (82.1%) | 63/74 (85.1%) | 47/60 (78.3%) | 142/180 (78.9%) | 4.48 / 8 (`exec=92`) | 63 (50t / max 2 / 187 tok) | 1304.6s (21.7m) / 41.6s | 4.27 GiB | **PASS (`31/31`, 0W/0F)** |
| **`test-v14` (FINAL PROMOTED: `submission/` + `v13_r32`)** | **`live_llm`** | **190/240 (79.17%)** | **+6 (+15/-9)** | **+12 (+17/-5 vs `A3-R2`)** | **224/240 (93.3%)** | **34** | **16** *(0 exec err)* | **28/49 (57.1%)** | **97/117 (82.9%)** | **65/74 (87.8%)** | **48/60 (80.0%)** | **142/180 (78.9%)** | **4.62 / 8** (`exec=114`) | **170 (79t / max 5 / 187 tok, mean 39.8)** | **2472.8s (41.2m) / 42.9s** | **5.01 GiB** (≤ 16 GiB) | **PASS (`31/31`, 0W/0F)** |

---

## 2. Validation (N=120) Complete Controlled Ablation Ladder

| Run ID | Owner | Description | Route | 3-Inst Acc | Δ vs `v7` (+gain/-loss) | Δ vs `A3-R2` | Pub Pass | FP | Pub Fail | `rows==0` (33) | `rows==1` (41) | `rows>1` (46) | Tools (mean/max) | LLM Calls (tasks / max / max_tok) | Predict / Score Time | Compliance |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-draft` | baseline | Original `issue_sql` submitted verbatim | `no_llm` | 47/120 (39.17%) | -51 | -56 | 63/120 | 16 | 57 | 7/33 (21.2%) | 18/41 (43.9%) | 22/46 (47.8%) | 0.00 / 0 | 0 | 0.0s / 18.4s | PASS |
| `B1-v7-nomodel` | baseline | `v7` harness without live LLM | `no_llm` | 87/120 (72.50%) | -11 | -16 | 99/120 | 12 | 21 | 22/33 (66.7%) | 30/41 (73.2%) | 35/46 (76.1%) | 3.01 / 6 | 0 | 11.5s / 31.9s | PASS |
| `B2-v7` (Ref) | baseline | `v7` harness + `Qwen3-0.6B` + `lora-e2` (r=16) | `live_llm` | 98/120 (81.67%) | 0 (ref) | -5 | 112/120 | 14 | 8 | 23/33 (69.7%) | 35/41 (85.4%) | 40/46 (87.0%) | 3.60 / 8 | *unlogged* (LB ≥40) | 759.7s / 36.7s | GAPS |
| `A1-variants` | `harness` | Upgraded harness + `proj_variants.py` | `no_llm` | 94/120 (78.33%) | -4 | -9 | 109/120 | 15 | 11 | 24/33 (72.7%) | 33/41 (80.5%) | 37/46 (80.4%) | 3.28 / 7 | 0 | 13.6s / 21.7s | **PASS** |
| `A2-probe` | `harness` | `A1` + `db_probe.py` (1-call `execute` literal repair) | `no_llm` | 97/120 (80.83%) | -1 | -6 | 112/120 | 15 | 8 | 24/33 (72.7%) | 34/41 (82.9%) | 39/46 (84.8%) | 3.41 / 8 | 0 | 13.6s / 24.6s | **PASS** |
| `A3-full-nomodel` | `harness` | `A2` + `static_audit.py` + train-fitted `reranker` FP Firewall | `no_llm` | 102/120 (85.00%) | +4 (+9/-5) | -1 | 112/120 | 10 | 8 | 27/33 (81.8%) | 36/41 (87.8%) | 39/46 (84.8%) | 4.27 / 8 | 0 | 16.2s / 29.4s *(repro 29.4s, 0.11 GiB)* | **PASS** |
| `A3-R2-nomodel` | `harness` | Round 2 deterministic harness (`r0` reorder, alpha-id FK, pre-prune) | `no_llm` | 103/120 (85.83%) | +5 (+10/-5) | 0 (ref) | 112/120 | 9 | 8 | 27/33 (81.8%) | 36/41 (87.8%) | 40/46 (87.0%) | 4.22 / 8 | 0 | 17.2s / 43.4s | **PASS** |
| `H2b-lora-e2` | `harness` | `A3` + **same** `train/lora-e2` (`thr=0.35`, legacy raw) | `live_llm` | 107/120 (89.17%) | +9 (+9/-0) | +4 | 117/120 | 10 | 3 | 27/33 (81.8%) | 38/41 (92.7%) | 42/46 (91.3%) | 4.40 / 8 | 22 (17t / 2 / 68) | 328.1s / 32.3s | **PASS** |
| `v12_r32_ep1` | `lora_trainer` | `A3` + `v12_r32_ep1` (Stage 1 ep1, `v2-rich-chat`, greedy) | `live_llm` | 105/120 (87.50%) | +7 (+9/-2) | +2 | 115/120 | 10 | 5 | 27/33 (81.8%) | 37/41 (90.2%) | 41/46 (89.1%) | 4.40 / 8 | 27 (20t / 2 / 98) | 767.1s / 33.1s | **PASS** |
| `v12_r32` | `lora_trainer` | `A3` + `v12_r32` (Stage 1 ep2, `v2-rich-chat`, greedy) | `live_llm` | 107/120 (89.17%) | +9 (+10/-1) | +4 | 116/120 | 9 | 4 | 28/33 (84.8%) | 38/41 (92.7%) | 41/46 (89.1%) | 4.42 / 8 | 26 (20t / 2 / 98) | 630.4s / 28.7s | **PASS (`27/27`, 0W/0F)** |
| `R1-v13_r32` | `lora_trainer` | `A3` + `v13_r32` (Stage 2 surgical ep, greedy) | `live_llm` | 108/120 (90.00%) | +10 (+11/-1) | +5 | 116/120 | 8 | 4 | 29/33 (87.9%) | 38/41 (92.7%) | 41/46 (89.1%) | 4.41 / 8 | 26 (20t / 2 / 78) | 519.8s / 24.3s | **PASS (`28/28`, 0W/0F)** |
| `R2-U1 + v13_r32` | `harness` | `R2` KV-shared multi-temp `Bo4` TTS + pre-check autofix | `live_llm` | 110/120 (91.67%) | +12 (+13/-1) | +7 (+7/-0) | 117/120 | 7 | 3 | 29/33 (87.9%) | 38/41 (92.7%) | 43/46 (93.5%) | 4.34 / 8 | 31 (19t / 5 / 76) | 578.6s / 32.2s | **PASS** |
| **`R2b-Full + v13_r32` (FINAL PROMOTED)** | **`harness` + `lora_trainer`** | **`R2-U1` + broader calibrated `fp_audit` w/ `Bo2` self-consistency guard** | **`live_llm`** | **111/120 (92.50%)** | **+13 (+14/-1)** | **+8 (+8/-0)** | **117/120** | **6** | **3** | **29/33 (87.9%)** | **39/41 (95.1%)** | **43/46 (93.5%)** | **4.56 / 8** | **64 (35t / 5 / 97)** | **1146.2s (19.1m) / 33.2s** | **PASS (`27/27`, 0W/0F)** |
| `R2b-Full + v14_clean_r32` | `lora_trainer` | Equal-depth 2-ep sibling (`train_rich_v4_calibrated`, `seed=777`) | `live_llm` | 111/120 (92.50%) | +13 (+15/-2) | +8 (+9/-1) | 118/120 | 7 | 2 | 28/33 (84.8%) | 39/41 (95.1%) | 44/46 (95.7%) | 4.58 / 8 | 70 (35t / 5 / 91) | 1816.2s *(contended)* / 44.4s | **PASS (`27/27`, 0W/0F)** |
| `R2b-Full + soup_r64` | `lora_trainer` | Exact QR-SVD LoRA Soup `0.5*v13 + 0.5*v14_clean` (`r=64`, `154.05 MiB`) | `live_llm` | 109/120 (90.83%) | +11 (+13/-2) | +6 (+7/-1) | 117/120 | 8 | 3 | 27/33 (81.8%) | 39/41 (95.1%) | 43/46 (93.5%) | 4.58 / 8 | 70 (35t / 5 / 94) | 1873.8s *(contended)* / 21.6s | **PASS (`27/27`, 0W/0F)** |

---

## 3. Base Model & Test-Time Scaling Scout (`model_scout` Phase 1 Summary)

Evaluated on the 22 `v7` validation failures (`14 FP` + `8 PubFail`, official 3-instance grading) + 4-thread CPU speed benchmark (`mp3_sql/.venv`):

| Model (Actual Unique Params) | Mode | 22-Fail Solved (Oracle) | FP14 Fixed | Pub8 Fixed | Avg New Tok | Est. 80-Task 4-CPU-Thread Time | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3-0.6B` (`596.05M`) | `think=False`, BoN-6 (`t=0.7`) | 2/22 (3/22) | 2/14 (3) | 0/8 | 66.1 | `41.5m` (1×64 fp32) / `88.0m` (Bo3 bf16) | **Selected base for LoRA A/B** |
| `Qwen/Qwen3-0.6B` (`596.05M`) | `think=True`, `max=1536` (`t=0.6`) | 2/22 (2/22) | 2/14 | 0/8 | 781.3 | `261m` bf16 / `433m` fp32 | **Rejected** (0 gain, >120m CPU cap) |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`494.03M`) | Greedy / BoN-6 (`t=0.7`) | 0/22 / 3/22 | 0 / 2 | 0 / 1 | 103 / 134 | `56.2m` (1×64 fp32) / `94.8m` (Bo3 bf16) | **Rejected** (no LoRA allowed; FP verifier regresses -2/15 ctrl) |

---

## 4. Exact Offline Reproduction Commands (Linux / Windows, 4 CPU Threads)

```bash
# Run from MP3_SQL_student_package/ with official .venv activated:
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export MP3_BASE_MODEL="/path/to/Qwen3-0.6B"          # Qwen/Qwen3-0.6B rev c1899de289a04d12100db370d81485cdf75e47ca
export MP3_ADAPTER_DIR="/path/to/v13_r32"            # 2-file LoRA ZIP: adapter_config.json + adapter_model.safetensors
export MP3_DTYPE="float32"                           # default; fast AVX2 prefill on 4 CPU threads
export MP3_LLM_LOG="outputs/test-llm.jsonl"          # defaults to outputs/llm_usage.jsonl if omitted

PYTHONPATH="$PWD/../submission" python -m mp3_sql.predict \
  --split test --method ../submission/submission.py \
  --checkpoint ../submission/config.json --workers 1 \
  --output outputs/test.json

python -m mp3_sql.score \
  --split test --predictions outputs/test.json \
  --workers 4 --output outputs/test-score.json
```

---

## 5. Final Compliance Sign-Off (`OVERALL=PASS`: `31/31 PASS`, `0 WARN`, `0 FAIL`)

- [x] **Official package integrity:** All 21 official `mp3_sql` code/data files match `PACKAGE_MANIFEST.json`.
- [x] **Official replay & 3-instance grading:** `validation` (`111/120 = 92.50%`) and `test` (`190/240 = 79.17%`) independently replayed with exact SHA-256 & 3-instance parity.
- [x] **Tool trace rules:** `max_calls=8/8` (`mean=4.625` on test, `4.56` on val), exact `issue_sql` checked before `repair_hint`, `0` duplicates, `0` forged observations.
- [x] **Single live LLM & offline safetensors loading:** Exactly `1` `AutoModelForCausalLM.from_pretrained` loader in `submission/llm_runner.py` (`local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`, `PeftModel.from_pretrained(..., is_trainable=False)`).
- [x] **Parameter budget & LoRA A/B rules:** Base `Qwen/Qwen3-0.6B` rev `c1899de289a04d12100db370d81485cdf75e47ca` (SHA-256 `f47f7117...` verified, `596.05M` unique / `751.63M` stored) + `v13_r32` (`r=32, alpha=64, bias="none"`, `392` strictly LoRA A/B tensors = `20.19M` params, `0` embed/lm_head/bias/modules_to_save, clean 2-file export `77.05 MiB` ≤ `512 MiB`) = **`616.24M` unique / `771.82M` stored** (≤ `1,000,000,000`).
- [x] **LLM usage log:** `outputs/test-llm.jsonl` verified (`170` records across `79` tasks, max `5`/task ≤ `6`, max `187` new tokens ≤ `2048`, single model `Qwen/Qwen3-0.6B` + adapter `v13_r32`).
- [x] **Export size, deps & zero leakage:** `submission/` = `0.176 MiB` (11 files ≤ `16 MiB`), stdlib + `requirements.txt` + `requirements-model.txt` only, `0` test/val task IDs, `0` test fingerprints, `0` grading file references; all LoRA datasets (`train_rich_v2_curated`, `train_rich_v3`, `train_dagger_v13`, `train_rich_v4_calibrated`) programmatically verified 100% strictly `train`-only.
- [x] **Runtime & memory (SOLO 4 CPU threads):** Predict `2472.8s` (`41.2 min` ≤ `120 min`) + Score `42.9s` (≤ `180s`), Peak RSS **`5.01 GiB`** (≤ `16.0 GiB`).
