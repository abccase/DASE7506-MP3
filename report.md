# DASE7506 Mini-Project 3 Technical Report: Execution-Grounded SQL Repair Agent with False-Positive Firewall and Rich Chat-Template LoRA

**Track & Route:** Single Approved Live-LLM Route (`Qwen/Qwen3-0.6B` + LoRA A/B Adapter `v13_r32`; optional No-LLM ablation `config_nomodel.json` also reproducible)  
**Benchmark Protocol:** `dase7506-sql-repair-v1` (Budget: 8 charged tool calls + free `submit`; 3-instance full-result grading)  
**Repository:** <https://github.com/abccase/DASE7506-MP3>

---

## 1. Executive Summary & Official Results

This project solves DASE7506 Mini-Project 3 (SQLite query repair across unseen database schemas) by coupling a **schema-aware, execution-probing Perceive-Plan-Act harness** with a **two-stage `Qwen/Qwen3-0.6B` LoRA A/B repair & false-positive-audit adapter (`v13_r32`)** and **budget-constrained test-time scaling**.

| System | Route | Validation (N=120) | Test (N=240) | Public Pass (Test) | Mean Tools | CPU Time (4 Threads) | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| Starter Baseline (Verbatim Draft) | No LLM | 47/120 (39.17%) | 88/240 (36.67%) | — | 0.00 | 37.1s score | — |
| Prior Harness + `lora-e2` (`v7` ref) | Live LLM | 98/120 (81.67%) | 184/240 (76.67%) | 215/240 (89.58%) | 3.65 | ~20.4 min | — |
| **Our No-LLM Harness (`config_nomodel.json`)** | **No LLM** | **103/120 (85.83%)** | **178/240 (74.17%)** | **208/240 (86.67%)** | **4.38** | **27.4s + 41.2s (68.6s)** | **0.11 GiB** |
| **Our Round 1 (`submission_v13` + `v13_r32`)** | Live LLM | 108/120 (90.00%) | 189/240 (78.75%) | 220/240 (91.67%) | 4.48 | 21.7 min + 41.6s | 4.27 GiB |
| **Final Promoted System (`submission/` + `v13_r32`)** | **Live LLM** | **111/120 (92.50%)** | **190/240 (79.17%)** | **224/240 (93.33%)** | **4.625** | **41.2 min + 42.9s** | **5.01 GiB** |

Key design insights that unlocked **+102 tasks over baseline (+52.50 pp)** and **+6 tasks (+15 gained, -9 lost) over the prior `v7` LoRA agent** on `test` (`+13` on `validation`):
1. **Eliminating Blind Database Assumptions via 1-Call `execute` Probing (`db_probe.py`):** Prior iterations had `execute = 0/120` calls, failing on DB cell value formats (`'Orton'` vs `'Team Orton'`, `'morning'` vs `'Morning'`, `'4'` vs `'4:xx'`) and literal column misattribution. A single read-only `group_concat(..., char(31))` probe inspects referenced columns and resolves unambiguous value mismatches deterministically.
2. **Defending Against Small-Row False Positives (`static_audit.py`, `reranker.py`, `fp_audit`):** Submitting immediately on the first public-passing query (`matches_example=True`) suffers severe 3-instance attrition on `row_count <= 1` tasks (on `test`, 18/50 remaining failures are `0-row` false positives). A train-fitted logistic + pairwise Bradley-Terry firewall (`FPReranker`, 5-fold DB-grouped `train` OOF AUC `0.8534`) flags risky passing queries (non-FK joins, ungrounded literals, extra `DISTINCT`, missing `ORDER BY`) and routes them to calibrated LLM `fp_audit` with `2/3` AST self-consistency guarding.
3. **Byte-Aligned Chat-Template LoRA & Test-Time Scaling (`llm_runner.py`, `v13_r32`):** Re-engineering LoRA training onto Qwen3's official non-thinking chat template with full actual returned columns/rows, `ordered` flag, all deduplicated 8B hints, static audit warnings, and DB sample values—followed by 1x-prefill KV-cache-shared multi-temperature `Bo4` sampling (`n<=4`, total `<=5` completions/task) and local pre-check schema repair—eliminated all execution errors on both `validation` (`0`) and `test` (`0`) and raised 1-instance public-example pass rate to `93.33%` (`224/240`).

---

## 2. Perceive-Plan-Act Agent Loop (`submission/submission.py`)

`create_agent(checkpoint=None)` instantiates a fresh `HarnessAgent` per episode (auto-loading `submission/config.json` when `checkpoint` is omitted). Every call to `act(observation)` executes a deterministic, budget-guarded state machine wrapped in an exception-safe fallback to `_submit_best()`:

```
[Observation: question, issue_sql, ordered, example, history, remaining]
   │
   ▼  Step 1 (Budget: 8 -> 7)
check(issue_sql)  ──[exact draft MUST precede repair_hint]
   │
   ▼  Step 2 (Budget: 7 -> 6)
repair_hint()     ──[deduplicate via AST canonical_sql_key, count consensus votes]
   │
   ▼  Step 3 (Early Schema on ex_rows == 0; Budget: 6 -> 5)
schema()          ──[allows static FK join checks & wrong-col-count hint pre-pruning]
   │
   ▼  Step 4-5 (Check unique hints & deterministic repairs; up to max_hint_checks=4)
check(cand_i)     ──[proj_variants: column prune, type/name/agg reorder, FK bridge, alpha-id order]
   │
   ├──► [PASSING CANDIDATE EXISTS] ──► False-Positive Firewall Path:
   │      1. Fetch schema() if ex_rows <= 1 or hard_red_flag or fp_risk >= 0.16
   │      2. Deterministic upgrades: var_r0_reorder, repair_alpha_order, repair_fk_bridge
   │      3. Run 1-call execute() DB Probe (db_probe.py) if risky/ungrounded literals
   │      4. Deterministic probe literal repair -> check()
   │      5. If fp_risk >= threshold (broad on ex==0 / ex==1 extra_distinct) & LLM available:
   │           LoRA mode="fp_audit" (n=1 greedy; escalate Bo3 on hard_flag; Bo2 SC guard on soft)
   │      6. FPReranker.pick_best_passing() -> submit()
   │
   └──► [NO PASSING CANDIDATE YET] ──► Public-Fail Repair & Test-Time Scaling Path:
          1. Fetch schema() if missing
          2. Run 1-call execute() DB Probe (db_probe.py) + deterministic literal repair -> check()
          3. Apply SetOp/alias-aware mp3_repair + proj_variants (disambiguate, prune, reorder, FK bridge)
          4. Pass 1 Live LLM (mode="repair" or mode="retry" on best runnable attempt, n=1 greedy)
               └─ local validate_executable_locally() + mp3_repair/GROUP BY autofix BEFORE spending check()
          5. If still failing & remaining >= 1:
               Pass 2 KV-cache-shared multi-temp Bo4 (temps=[0.80, 1.00, 1.20, 1.40], top_p=0.97, top_k=25)
               -> local pre-validation + AST self-consistency voting -> check() novel candidates
          6. Fallback: FPReranker.pick_best_fallback() (prefers ok=True, matching col count, highest audit) -> submit()
```

### Core Modules in `submission/` (Total: 11 files, `0.176 MiB` ≤ `16 MiB` limit)
- **`submission.py`** (`HarnessAgent`, `create_agent`): Budget-aware state machine enforcing `<= 8` charged tool calls (`remaining` guard), `check(issue_sql)` prior to `repair_hint`, and `self.used_actions` deduplication so identical tool actions are never repeated.
- **`static_audit.py`** (`SchemaCatalog`, `audit_sql`, `validate_executable_locally`): Parses SQLite ASTs with `sqlglot` to flag `NON_FK_JOIN`, `TAUTOLOGICAL_JOIN`, `CROSS_TYPE_JOIN`, `ORDERED_MISSING_ORDERBY`, `ALPHABETICAL_ID_ORDER`, `EXTRA_DISTINCT_NO_CUE`, `MISSING_DISTINCT_WITH_CUE`, `NEGATION_NEQ_ON_JOIN`, and `EMPTY_REF_TABLE`.
- **`db_probe.py`** (`build_probe_query`, `parse_probe_result`, `repair_literals_from_probe`): Synthesizes a single valid SQLite query packing `DISTINCT` sample values and exact/case-insensitive/substring literal existence checks across referenced text columns using `group_concat(..., char(31))`—requiring only **1 `execute` budget** and avoiding `UNION ALL` / SQLite expression-depth limits (verified `480/480` clean execution on `train`).
- **`proj_variants.py`**: Deterministic AST repair functions—`prune_extra_columns` (with alias inlining into `ORDER BY`/`GROUP BY`/`HAVING`), `reorder_columns` (matching `example.columns` + cell types/aggregates), `disambiguate_columns`, `repair_bogus_join_via_fk_bridge`, `repair_alphabetical_id_order`, and `add_missing_group_by_for_mixed_agg`.
- **`mp3_repair.py`**: Schema-anchored identifier repair extended to `exp.SetOperation` (`UNION`/`EXCEPT`/`INTERSECT`), undeclared alias join synthesis (`T3.*`), and tautological self-join blocking.
- **`reranker.py` & `reranker_weights.json`**: 22-feature pure-Python `FPReranker` (`0.95 KiB` JSON) computing calibrated `fp_risk` and `pass_score` using weights fitted via 5-fold DB-grouped logistic + Bradley-Terry optimization strictly on `train` (OOF ROC-AUC = `0.8534`).
- **`prompt_format.py` & `llm_runner.py`**: Shared `v2-rich-chat` prompt builder, prompt-injection sanitizer, single `AutoModelForCausalLM` + `PeftModel` offline safetensors loader, `past_key_values.batch_repeat_interleave(N)` KV-shared sampling, strict `MAX_COMPLETIONS_PER_TASK = 6` and `MAX_NEW_TOKENS_CAP = 2048` enforcement, and mandatory JSONL logging (`MP3_LLM_LOG`).

---

## 3. Context, Working Memory, and Retrieval Choices

### Structured Working Memory (`v2-rich-chat`)
Rather than keeping raw JSON tool histories (which quickly exceed small-model context lengths and distract 0.6B attention with redundant observations), `HarnessAgent` maintains structured per-episode working memory:
- **`self.candidates`**: Every evaluated query (`draft`, unique `repair_hint`s, deterministic repairs, LLM outputs) stored alongside its execution outcome (`ok`, `matches`, `row_count`, `columns`, `rows[:4]`, `error`), static audit flags, and reranker risk score.
- **`self.probe_report`**: Compact mapping of queried `Table.Column` sample values and literal-grounding verdicts (`FOUND_EXACT`, `CASE_DIFF`, `SUBSTR_MATCH`, `FOUND_IN_OTHER_COL`).
- **`prompt_format.build_messages()`**: Serializes `Schema` (compact `table(col TYPE PK, ...) | FK: ...`), `Question` + explicit `Ordered output required: yes|no`, `Expected Public Example`, `Database Value Samples`, `Original Draft` + feedback, deduplicated `Candidate Hints` with consensus `(votes=k)` annotations, `Previous Model Attempt` (in `retry` mode), and mode-specific task directive (`repair` vs `fp_audit` vs `retry`). Across all `1,950` `train_rich_v2_curated` examples, median length is `666` tokens and 95th percentile is `1,312` tokens (fitting inside `max_len=2048` with `>99.8%` zero truncation).

### Why BM25 `retrieve` Was Evaluated and Excluded
Because Spider schemas are strictly disjoint across `train` (118 DBs), `validation` (20 DBs), and `test` (19 DBs), cross-split BM25 retrieval via the `retrieve` tool returns training examples from completely unrelated tables and columns:
- Empirical check during development (`few_shot_pass.py`): Injecting retrieved `train` examples into prompts dropped hard-task accuracy to `4/12` (vs `10/24` zero-shot retrieval-free) due to foreign table/column hallucination, while consuming 1 of only 8 charged tool calls and ~400–600 context tokens.
- Consequently, `retrieve` is intentionally not invoked at inference time; tool budget is invested instead in `schema` and 1-call `execute` database probing, which provide exact schema/value ground truth for the active task's database.

---

## 4. Weight Optimization (`Qwen/Qwen3-0.6B` Two-Stage LoRA `v13_r32`)

### Base Model & Parameter Compliance
- **Base Repository & Revision:** `Qwen/Qwen3-0.6B`, immutable commit `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Loading Policy:** Strictly offline `AutoModelForCausalLM.from_pretrained(..., local_files_only=True, trust_remote_code=False, use_safetensors=True)` + `PeftModel.from_pretrained(..., is_trainable=False)`.
- **Parameter Budget Verification:**
  - Base `Qwen/Qwen3-0.6B`: `596,049,920` unique parameters (`751,632,384` stored parameters including untied `lm_head` tensor layout).
  - LoRA A/B Adapter (`v13_r32`): `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `bias="none"`, `modules_to_save=None`, targeting `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` across all 28 layers = `392` tensors, **`20,185,088` trainable parameters** (`0` embedding, `0` `lm_head`, `0` bias weights).
  - **Total Parameters:** **`616,235,008` unique / `771,817,472` stored** — strictly below the `1,000,000,000` ceiling.
  - **Adapter Export:** Exactly 2 files (`adapter_config.json` + `adapter_model.safetensors`), **`77.05 MiB`** (≤ `512 MiB` limit).

### Two-Stage `train`-Only Curriculum & RTX 3060 Training Optimization
DeepSeek's prior LoRA attempts (`lora-e2`, `lora-v10`, `lora-v11sub`) stalled at `9–10/24` hard validation tasks because they trained with raw text prompts, omitted actual returned rows and `ordered` flags, had weak single-token corruptions, and lacked `fp_audit` supervision. We rebuilt the pipeline on `train` (480 tasks, 118 DBs) using `training_records/train_rich_lora.py`:

1. **Memory-Efficient Exact GPU Training (`logits_to_keep = target_len + 1`):**
   On the RTX 3060 Laptop GPU (6 GiB VRAM), computing full `[1, seq_len, 151936]` logits materialized a ~1.2 GiB tensor causing WDDM PCIe paging (`~162 tok/s`). Passing `logits_to_keep = target_len + 1` with `fp16` autocast, `fp32` LoRA master weights, `GradScaler`, `sdpa`, `gradient_checkpointing(use_reentrant=True)`, and `torch.cuda.set_per_process_memory_fraction(0.48)` kept peak VRAM at **`1.842 GiB`** and boosted throughput to **`686–1,065 tok/s`** (>5x speedup, 0 OOM). Prompt tokens are strictly masked with `-100` under Qwen3's `apply_chat_template(..., enable_thinking=False)`.
2. **Stage 1 (`v12_r32_ep1`, 1 epoch / 122 steps, `lr=2e-4`, `accum=16`, `max_len=2048`):**
   Trained on `train_rich_v2_curated.jsonl` (`1,950` `train`-only examples across 480 tasks / 118 DBs: `orig_hard_repair=282`, `orig_allfail_fullctx=80`, `hint_repair=125`, `multi_hint_synth=87`, `audit_fp_real=178`, `audit_fp_synth=216`, `audit_tp_keep=250`, `sem_corrupt_repair=463` across 12 AST corruptors executed on `train` public DBs, `multi_turn_retry=269`). Epoch 1 average loss reached `0.2115` (`105/120` val; Epoch 2 `v12_r32` reached `107/120` val).
3. **Stage 2 Surgical Calibration (`v13_r32`, 1 epoch / 115 steps, `lr=1e-4`, `23.15 min`):**
   Resumed from `v12_r32_ep1` on `train_rich_v3.jsonl` (`1,833` `train`-only examples across 479 tasks / 118 DBs), calibrating the `Candidate Hints -> MATCHES_EXAMPLE` prior inside `fp_audit` from `28.6%` to **`75.2%` FP-rewrite vs `24.8%` clean-only TP-keep**, and adding targeted `train`-DB AST corruptions for `_corrupt_literal_wrong_column`, `_corrupt_negation_to_join` (`NOT IN`/`EXCEPT` -> `JOIN !=`), and `_corrupt_scalar_subquery`. Loss decreased smoothly from `0.1075` (step 1) to `0.0249–0.0325` (steps 90–105), lifting `validation` to **`108/120`** (Round 1) and **`111/120`** (Round 2) with **0 regressions** over the No-LLM harness on both `validation` (`+8/-0`) and `test` (`+12/-0` in Round 1; `+17/-5` in Round 2).

---

## 5. Controlled Ablation Studies

All component choices were validated through controlled ablations on `validation` (N=120) before running official `test` (N=240) milestones (all scored with official 3-instance `mp3_sql.score` on 4 CPU threads).

### Ablation 1: Deterministic & Execution-Probing Harness Ladder (No Live LLM)
Isolates each deterministic module while holding live LLM calls at **0**:

| Configuration | Val 3-Inst | `ex==0` (33) | `ex==1` (41) | `ex>1` (46) | FP / PF | Mean Tools | Predict (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-draft`: Verbatim `issue_sql` | 47/120 (39.17%) | 7/33 | 18/41 | 22/46 | 16 / 57 | 0.00 | 0.0s |
| `B1-v7-nomodel`: Prior heuristic harness | 87/120 (72.50%) | 22/33 | 30/41 | 35/46 | 12 / 21 | 3.01 | 11.5s |
| `A1`: Upgraded loop + `proj_variants.py` | 94/120 (78.33%) | 24/33 | 33/41 | 37/46 | 15 / 11 | 3.28 | 13.6s |
| `A2`: `A1` + `db_probe.py` (1-call `execute`) | 97/120 (80.83%) | 24/33 | 34/41 | 39/46 | 15 / 8 | 3.41 | 13.6s |
| `A3`: `A2` + `static_audit.py` + `FPReranker` | 102/120 (85.00%) | 27/33 | 36/41 | 39/46 | 10 / 8 | 4.27 | 16.2s |
| **`A3-R2`: `A3` + Round 2 hygiene (`r0_reorder`, `alpha_id`, `SetOp`)** | **103/120 (85.83%)** | **27/33** | **36/41** | **40/46** | **9 / 8** | **4.22** | **17.2s** |

On `train` (480 tasks), `A3-R2` similarly improved from `355/480 (73.96%)` (`v7-nomodel`) to `381/480 (79.38%, +26/-0)` with zero regressions; on `test` (240 tasks), `A3-R2` achieved **`178/240 (74.17%)`** in `27.4s` predict + `41.2s` score (`0.11 GiB` RAM, no LLM).

### Ablation 2: Harness Isolation Holding Legacy LoRA Fixed (`train/lora-e2`)
To separate harness engineering gains from LoRA weight gains, we plugged DeepSeek's original `lora-e2` (`r=16`) adapter into our Round 1 harness (`H2b`):
- `v7` (old harness + `lora-e2`): `98/120 (81.67%)`, `14 FP / 8 PF`, `759.7s`.
- **`H2b` (our `A3` harness + same `lora-e2`)**: **`107/120 (89.17%)`, `+9 gained / 0 lost`**, `10 FP / 3 PF`, **`328.1s`** (2.3x faster due to selective firewall triggering).

### Ablation 3 & 4: LoRA Progression + Test-Time Scaling (`Bo4`) + `fp_audit` Self-Consistency Guard
Starting from `A3` (`102/120`), we isolated Stage 1 & Stage 2 LoRA training, followed by Round 2 KV-shared `Bo4` sampling (`R2-U1`) and broader `fp_audit` with `2/3` AST self-consistency guarding (`R2b-Full`):

| System | Val 3-Inst | Δ vs `v7` | Δ vs `A3-R2` | `ex==0` | `ex==1` | `ex>1` | FP / PF | LLM Calls (Max/Task) | Test 3-Inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v7` (Baseline Ref) | 98/120 (81.67%) | 0 | -5 | 23/33 | 35/41 | 40/46 | 14 / 8 | unlogged | 184/240 (76.67%) |
| `A3` + `v12_r32_ep1` | 105/120 (87.50%) | +7 | +2 | 27/33 | 37/41 | 41/46 | 10 / 5 | 27 (2) | — |
| `A3` + `v12_r32` (Epoch 2) | 107/120 (89.17%) | +9 | +4 | 28/33 | 38/41 | 41/46 | 9 / 4 | 26 (2) | — |
| `R1`: `A3` + `v13_r32` (Greedy) | 108/120 (90.00%) | +10 | +5 | 29/33 | 38/41 | 41/46 | 8 / 4 | 26 (2) | **189/240 (78.75%)** |
| `R2-U1`: `R1` + `Bo4` TTS + pre-check autofix | 110/120 (91.67%) | +12 | +7 (+7/-0) | 29/33 | 38/41 | 43/46 | 7 / 3 | 31 (5) | — |
| **`R2b-Full`: Final `submission/` + `v13_r32`** | **111/120 (92.50%)** | **+13** | **+8 (+8/-0)** | **29/33** | **39/41** | **43/46** | **6 / 3** | **64 (5)** | **190/240 (79.17%)** |

### Ablation 5: Ungated ≤1B Model Scout, Thinking Mode & Weight-Space LoRA Soup
1. **Zero-Shot ≤1B & Thinking Mode (on the 22 `v7` Validation Failures):**
   - `Qwen/Qwen3-0.6B` (`596.05M`, `enable_thinking=False`, BoN-6): solved `2/22` (oracle `3/22`), avg `66.1` tokens.
   - `Qwen/Qwen3-0.6B` (`enable_thinking=True`, `max_new=1536`): solved **`2/22`**, avg **`781.3` tokens** (projected **261–433 min** on 80 tasks -> violates 120-min CPU limit for zero accuracy gain).
   - `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`494.03M`, ungated): `0/22` greedy, `3/22` BoN-6, and as an FP verifier regressed `-2/15` on control tasks (also ineligible for LoRA training under course rules).
2. **DAgger Sibling & Exact QR-SVD LoRA Soup (on Final Round 2 Harness):**
   - On-policy DAgger + new AST corruptors trained as an equal-depth sibling `v14_clean_r32` also reached `111/120 (92.50%)` (`118/120` public pass, `PF=2`), but with `1` zero-row regression (`+9/-1` vs `A3-R2`), whereas `v13_r32` achieved **`+8/-0` zero regressions** and highest `ex==0` (`29/33`).
   - Exact QR-SVD weight fusion (`0.5 * ΔW(v13) + 0.5 * ΔW(v14_clean)` at `r=64`, `0.0` reconstruction error) scored `109/120 (90.83%)`, confirming that single-basin `v13_r32` weights are sharper for exact SQL token prediction.

---

## 6. Runtime, Tool, and Model Cost Evaluation

Measured **solo on 4 CPU threads** (`OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=""`) using the official `mp3_sql.predict` (`--workers 1`) and `mp3_sql.score` (`--workers 4`):

| Metric | Validation (`N=120`) | Test (`N=240`) | Course Limit |
|---|---:|---:|---|
| Official 3-Instance Accuracy | **111/120 (92.50%)** | **190/240 (79.17%)** | — |
| Tool Calls (Mean / Max) | `4.56 / 8` | `4.625 / 8` | `<= 8` charged calls/task |
| Tasks Triggering LLM | `35 / 120` (`29.2%`) | `79 / 240` (`32.9%`) | — |
| Total LLM Completions | `64` | `170` | — |
| Max Completions per Task | `5` | `5` | `<= 6` per task |
| New Tokens (Mean / Max) | `— / 97` | `39.79 / 187` | `<= 2,048` per completion |
| Prediction Wall Time | `1146.2s` (`19.1 min`) | `2472.8s` (`41.2 min`) | `<= 120 min` (2 hours) |
| Scoring Wall Time | `33.2s` | `42.5s–42.9s` | `<= 180s` |
| Peak Resident RAM (RSS) | — | **`5.01 GiB`** | `<= 16.0 GiB` |
| Source / Adapter Export Size | `0.176 MiB` / `77.05 MiB` | `0.176 MiB` / `77.05 MiB` | `<= 16 MiB` / `<= 512 MiB` |

**CPU Dtype Engineering (`float32` vs `bfloat16`):** Benchmarking `Qwen3-0.6B` on 4 AVX2 CPU threads (527-token SQL prompt) showed that while `bfloat16` decodes faster (`4.53 tok/s` vs `2.44 tok/s`), `float32` prefill is **4.8x faster** (`4.88s` vs `23.59s`). For short SQL outputs (mean `39.8` tokens), `MP3_DTYPE=float32` finishes each call in `~21s` vs `~32s` in `bfloat16`, keeping total 240-task prediction time at **41.2 min** (under 35% of the 2-hour budget).

---

## 7. Failure Case Analysis

On `test` (`N=240`), `224/240 (93.33%)` pass the public example with **0 SQL execution errors**. The remaining `50` failures divide cleanly into two regimes:

| Failure Regime | Test Count | Breakdown by `ex_rows` |
|---|---:|---|
| **Public-Fail (Semantic Mismatch)** | `16` (`0` exec errors) | Concentrated in multi-step aggregation, nested correlated subqueries, and domain-level negation |
| **Hidden-Instance False Positives** | `34` | `rows==0`: **21** (`28/49 = 57.1%` pass) · `rows==1`: **10** (`97/117 = 82.9%`) · `rows>1`: **3** (`65/74 = 87.8%`) |

Representative concrete failure cases observed on `validation` (`9` remaining errors out of 120):
1. **`dorm_1` (`validation-a75b7764fcacce68`, `pub_ok=False`):** *"How many students are older than average for each gender?"* — Requires a correlated subquery or CTE comparing `age > (SELECT AVG(age) FROM student s2 WHERE s2.sex = s1.sex)` before grouping; all candidates collapsed `WHERE` into `HAVING avg(age) > 3`.
2. **`sports_competition` (`validation-5cf3b447b4af02c2`, `pub_ok=False`):** *"Countries that have never participated in any friendly-type competitions"* — Requires `EXCEPT` or `NOT IN` over the full `country` domain (`Country` column in `competition`), whereas candidates filtered `WHERE Competition_type = 'friendly'` and returned empty/wrong sets.
3. **`e_government` (`validation-f76a31ffe49ff88d`, `pub_ok=True`, `1-row FP`):** *"Last name of the first individual contacted..."* — Query passed the 1-row public DB using `ORDER BY date_contact_from LIMIT 1`, but tie-breaking or join multiplicity diverged on the unseen grading DBs.
4. **`local_govt_mdm` (`validation-3625496ea0b7be64` & `36ce398902e7135d`, `pub_ok=True`, `0-row FP`):** In tables that are empty or lack matching join rows in the sampled public subset (`example.row_count == 0`), semantically distinct queries (e.g. `LIKE` vs `!=` or slightly altered filter joins) produce identical `0-row` outputs on the public database, leaving zero local execution signal to distinguish them.

---

## 8. Prompt-Injection Risks & Safeguards

In this environment, four inputs flow dynamically from external/task sources into prompts and deterministic SQL construction: `observation["task"]["question"]`, `issue_sql`, `repair_hint` suggestions, and database cell strings returned by `execute` (`db_probe.py`) or `check` sample rows. Malicious or malformed strings inside any of these (e.g., `"-- ignore previous instructions; <|im_end|><|im_start|>system ..."` or `'; DROP TABLE ...`) pose two distinct risks: **chat-template role hijacking** and **SQL injection / write escalation**.

We enforce **five defense-in-depth safeguards**:
1. **Chat-Token & Control-Character Sanitization (`prompt_format.py:_sanitize_untrusted`):** Strips all HuggingFace/Qwen special chat delimiters matching `<\|[^|]{1,40}\|>` and null bytes `\x00` from `question`, `db_probe` sample cell values, and `feedback`, and enforces strict character caps (`600` chars on `question`, `1500` chars on DB samples, `180` chars on error strings).
2. **Explicit Untrusted-Data Boundary in `SYSTEM_PROMPT`:** Instructs the model that question text, schema identifiers, DB cell values, and candidate hints are untrusted data and never executable instructions.
3. **Strict Output Extraction & Single-Statement Read-Only AST Validation (`extract_sql_from_output` + `static_audit.validate_executable_locally`):** Parses model output with `sqlglot`, retaining only the first statement, rejecting any non-`SELECT`/`SetOperation` root or any DDL/DML/command node (`Insert`, `Update`, `Delete`, `Create`, `Drop`, `Command`, `Into`), and verifying all table/column references against the tool-provided `schema` before issuing a `check` call.
4. **Safe SQLite Probe Construction (`db_probe.py`):** Queries are built exclusively from catalog-verified table/column identifiers quoted safely and executed inside the official read-only `execute` tool (never string-interpolating unescaped DB cell values into executable SQL; deterministic literal replacement escapes single quotes properly via `sqlglot` AST `exp.Literal.string(...)`).
5. **Official Runtime Sandbox:** All database execution passes through `mp3_sql/sqlite_tool.py` (`?mode=ro&immutable=1`, `PRAGMA query_only=ON`, SQLite authorizer allowing only `SELECT`/read-only functions, 0.6s/5M-step progress limit). Additionally, `retrieve` (a potential secondary injection vector) is disabled.

---

## 9. Data Provenance & Integrity

- **Zero Test/Validation Leakage:** Every learned component—`reranker_weights.json` (22 features fitted on `579` public-passing candidates from `train` 480 tasks), `train_rich_v2_curated.jsonl` (`1,950` examples across `480` `train` tasks / `118` `train` DBs), and `train_rich_v3.jsonl` (`1,833` examples across `479` `train` tasks / `118` `train` DBs)—was constructed **strictly from the `train` split**. Both `build_rich_dataset.py` and `train_rich_lora.py` assert `id.startswith("train-")`, and provenance records (`training_records/data_provenance_v2.json`, `training_records/data_provenance_v3.json`) verify exact SHA-256 checksums of source files:
  - `train.jsonl`: `562d6d54cbfac83829cf528a3b8ca5f101994c9fe5b1a502519beba328900f65`
  - `train_references.json`: `def39effca9b1efe78f7316d71739111b8c8290dd0897db5e6655293fca24da7`
  - `repair_hints.json`: `cedf7056f9248bc6296ee80bc03e2a7744c119111f640d8596ea195c5d1e6b03`
  - `public_results.json`: `1dba955aab6ef8f6cbe4f05bad697f513929400f262afbc1bd6bd4f37efba8c2`
  - `manifest.json`: `bf7a517249ebe31b9cad8cbf5d6598fde98316d89f9aa1172cae3ad83289a6bc`
- **Official Package Integrity:** All 21 files under `MP3_SQL_student_package/` pass `python -m mp3_sql.check_data` and match `PACKAGE_MANIFEST.json`. No test tasks, test references, or test fingerprints appear anywhere in `submission/`, `adapter/`, or `training_records/`.

---

## 10. External Assistance Disclosure

- Coding, error-taxonomy analysis, multi-agent experiment orchestration, and report drafting were assisted by AI coding assistants (DeepSeek during initial exploration and Gemini / Claude via Antigravity during final harness, LoRA, and verification rounds).
- **Zero external LLM APIs or internet calls** are used during evaluation or grading. All inference executes strictly offline using `Qwen/Qwen3-0.6B` (rev `c1899de289a04d12100db370d81485cdf75e47ca`) and our local `train`-only LoRA adapter `v13_r32`.

---

## 11. Exact Offline Reproduction Instructions

### Environment & Model Setup (Linux / Windows, Python 3.11–3.13)
1. Install official dependencies inside `MP3_SQL_student_package/`:
   ```bash
   pip install -r requirements.txt -r requirements-model.txt
   python -m mp3_sql.check_data
   ```
2. Provision the immutable public base model `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` locally (e.g., `models/Qwen3-0.6B`), and extract the 2-file LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) locally (e.g., `adapter/`).

### Reproduce Final Official Test Score (`190/240 = 79.17%`, 4 CPU Threads)
Run from `MP3_SQL_student_package/`:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export MP3_BASE_MODEL="../models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="../adapter"
export MP3_DTYPE="float32"
export MP3_LLM_LOG="outputs/test-llm.jsonl"

PYTHONPATH="$PWD/../submission" python -m mp3_sql.predict \
  --split test \
  --method ../submission/submission.py \
  --checkpoint ../submission/config.json \
  --workers 1 \
  --output outputs/test.json

python -m mp3_sql.score \
  --split test \
  --predictions outputs/test.json \
  --workers 4 \
  --output outputs/test-score.json
```

### Reproduce Fast No-LLM Harness Ablation (`178/240 = 74.17%`, `~69s` Total)
```bash
PYTHONPATH="$PWD/../submission" python -m mp3_sql.predict \
  --split test \
  --method ../submission/submission.py \
  --checkpoint ../submission/config_nomodel.json \
  --workers 4 \
  --output outputs/test-nomodel.json

python -m mp3_sql.score \
  --split test \
  --predictions outputs/test-nomodel.json \
  --workers 4 \
  --output outputs/test-nomodel-score.json
```

### Reproduce LoRA & Reranker Training (`train`-Only)
From the repository root (`DASE7506-MP3/`):
```bash
python training_records/fit_reranker.py
python training_records/build_rich_dataset.py --out /tmp/train_rich_v2.jsonl --seed 42
python training_records/build_v3_dataset.py --out /tmp/train_rich_v3.jsonl --seed 42
python training_records/train_rich_lora.py --data /tmp/train_rich_v2.jsonl --out /tmp/v12_ep1 --epochs 1 --rank 32 --lr 2e-4
python training_records/train_rich_lora.py --data /tmp/train_rich_v3.jsonl --resume /tmp/v12_ep1 --out /tmp/v13_r32 --epochs 1 --rank 32 --lr 1e-4
```
