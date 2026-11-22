# DASE7506 Mini-Project 3: Execution-Grounded SQL Repair Agent

An execution-grounded, false-positive-aware SQLite repair agent combining a deterministic Perceive-Plan-Act harness (`submission/`) with a two-stage `Qwen/Qwen3-0.6B` LoRA A/B adapter (`v13_r32`) trained strictly on the `train` split.

## Official Results (4 CPU Threads, 3-Instance Grading)

| System | Route | Validation (N=120) | Test (N=240) | Public Pass (Test) | 4-CPU Time | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| Starter Baseline (`issue_sql` draft) | No LLM | 47/120 (39.17%) | 88/240 (36.67%) | — | 37.1s score | — |
| Prior Baseline (`v7`, legacy LoRA) | Live LLM | 98/120 (81.67%) | 184/240 (76.67%) | 215/240 (89.58%) | ~20.4 min | — |
| **Our Deterministic + Probing Harness (`config_nomodel.json`)** | **No LLM** | **103/120 (85.83%)** | **178/240 (74.17%)** | **208/240 (86.67%)** | **68.6s total** | **0.11 GiB** |
| **Final Agent (`submission/` + `v13_r32`)** | **Live LLM** | **111/120 (92.50%)** | **190/240 (79.17%)** | **224/240 (93.33%)** | **41.2m + 42.9s** | **5.01 GiB** |

- **Zero SQL execution errors** on both `validation` (120) and `test` (240).
- **Strict budget & parameter compliance:** `<= 8` charged tool calls/task (`mean=4.625`), `<= 5` LLM completions/task (`limit 6`), `<= 187` new tokens (`limit 2048`), single offline safetensors loader (`trust_remote_code=False`), `616.24M` unique / `771.82M` stored total parameters (`<= 1B`), `0.81 MiB` source export with `report.pdf` (`<= 16 MiB`), `77.05 MiB` 2-file LoRA export (`<= 512 MiB`).

## Repository Layout

```text
DASE7506-MP3/
├── README.md                  # Reproduction & submission guide
├── report.md / report.pdf     # Complete technical report (architecture, 5 ablations, safety, provenance)
├── requirements.txt           # Core harness dependency (sqlglot==27.29.0)
├── requirements-model.txt     # Live-LLM runtime dependencies (torch, transformers, peft, safetensors)
├── submission/                # Self-contained agent export (<= 16 MiB)
│   ├── submission.py          # create_agent(checkpoint=None) & HarnessAgent state machine
│   ├── config.json            # Final Live-LLM configuration
│   ├── config_nomodel.json    # Fast No-LLM ablation configuration (68.6s total)
│   ├── db_probe.py            # Single-call execute() DB value prober & literal repairer
│   ├── static_audit.py        # Schema/FK join validator & local pre-check SQL validator
│   ├── proj_variants.py       # Column pruning, type/agg reordering & FK bridge repair
│   ├── mp3_repair.py          # SetOperation & alias-aware schema anchoring
│   ├── reranker.py            # Pure-Python logistic + Bradley-Terry FP risk ranker
│   ├── reranker_weights.json  # Train-only fitted weights (OOF AUC 0.8534)
│   ├── prompt_format.py       # Shared v2-rich-chat prompt builder & injection sanitizer
│   ├── llm_runner.py          # Offline single-LLM safetensors loader, KV-shared BoN & MP3_LLM_LOG
│   └── report.pdf             # Bundled copy of the technical report
├── outputs/                   # Official final test predictions, score & LLM call log
│   ├── test.json
│   ├── test-score.json
│   └── test-llm.jsonl
└── training_records/          # Train-only dataset builders, LoRA/reranker trainers, provenance & ablation logs
    ├── build_rich_dataset.py
    ├── build_v3_dataset.py
    ├── train_rich_lora.py
    ├── fit_reranker.py
    ├── data_provenance_v2.json
    ├── data_provenance_v3.json
    ├── v12_stage1_train_summary.json
    ├── v13_stage2_train_summary.json
    └── ablation_log.md
```

## Immutable Release Artifacts (`v1.0.0`)

- **Source Export ZIP (`<= 16 MiB`):**  
  <https://github.com/abccase/DASE7506-MP3/releases/download/v1.0.0/source-submission-v1.0.0.zip>
- **2-File LoRA Adapter ZIP (`77.05 MiB <= 512 MiB`, `adapter_config.json` + `adapter_model.safetensors`):**  
  <https://github.com/abccase/DASE7506-MP3/releases/download/v1.0.0/lora-adapter-v13_r32.zip>

## Offline Reproduction (4 CPU Threads)

### 1. Provision Dependencies, Base Model & Adapter
Place the official course package at `MP3_SQL_student_package/`, provision the exact base model revision `Qwen/Qwen3-0.6B` (`c1899de289a04d12100db370d81485cdf75e47ca`) into `models/Qwen3-0.6B/`, and unzip `lora-adapter-v13_r32.zip` into `adapter/`.

```bash
cd MP3_SQL_student_package
python -m pip install -r ../requirements.txt -r ../requirements-model.txt
python -m mp3_sql.check_data
```

### 2. Run Official Test Evaluation (Live-LLM Route: `190/240 = 79.17%`)

**Linux (bash):**
```bash
cd MP3_SQL_student_package
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export MP3_BASE_MODEL="$PWD/../models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/../adapter"
export MP3_DTYPE="float32"
export MP3_LLM_LOG="outputs/test-llm.jsonl"

PYTHONPATH="$PWD/../submission" python -m mp3_sql.predict \
  --split test --method ../submission/submission.py \
  --checkpoint ../submission/config.json --workers 1 \
  --output outputs/test.json

python -m mp3_sql.score \
  --split test --predictions outputs/test.json \
  --workers 4 --output outputs/test-score.json
```

**Windows (PowerShell):**
```powershell
Set-Location MP3_SQL_student_package
$env:OMP_NUM_THREADS="4"; $env:MKL_NUM_THREADS="4"; $env:OPENBLAS_NUM_THREADS="4"; $env:TORCH_NUM_THREADS="4"
$env:CUDA_VISIBLE_DEVICES=""
$env:MP3_BASE_MODEL="$PWD\..\models\Qwen3-0.6B"
$env:MP3_ADAPTER_DIR="$PWD\..\adapter"
$env:MP3_DTYPE="float32"
$env:MP3_LLM_LOG="outputs\test-llm.jsonl"
$env:PYTHONPATH="$PWD\..\submission"

.\.venv\Scripts\python.exe -m mp3_sql.predict --split test --method ..\submission\submission.py --checkpoint ..\submission\config.json --workers 1 --output outputs\test.json
.\.venv\Scripts\python.exe -m mp3_sql.score --split test --predictions outputs\test.json --workers 4 --output outputs\test-score.json
```

### 3. Run Fast No-LLM Ablation (`178/240 = 74.17%`, `~69s` Total)
```bash
cd MP3_SQL_student_package
PYTHONPATH="$PWD/../submission" python -m mp3_sql.predict \
  --split test --method ../submission/submission.py \
  --checkpoint ../submission/config_nomodel.json --workers 4 \
  --output outputs/test-nomodel.json

python -m mp3_sql.score \
  --split test --predictions outputs/test-nomodel.json \
  --workers 4 --output outputs/test-nomodel-score.json
```

### 4. Reproducing `train`-Only Reranker & LoRA Training
From the repository root (GPU recommended for LoRA fine-tuning):
```bash
python training_records/fit_reranker.py
python training_records/build_rich_dataset.py --out /tmp/train_rich_v2.jsonl --seed 42
python training_records/build_v3_dataset.py --out /tmp/train_rich_v3.jsonl --seed 42
python training_records/train_rich_lora.py --data /tmp/train_rich_v2.jsonl --out /tmp/v12_ep1 --epochs 1 --rank 32 --lr 2e-4
python training_records/train_rich_lora.py --data /tmp/train_rich_v3.jsonl --resume /tmp/v12_ep1 --out /tmp/v13_r32 --epochs 1 --rank 32 --lr 1e-4
```
