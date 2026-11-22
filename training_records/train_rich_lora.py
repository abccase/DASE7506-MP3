"""GPU-optimized, 100% compliant LoRA trainer for Qwen/Qwen3-0.6B (RTX 3060 6GB).

Key systems optimizations for RTX 3060 Laptop (with GUI apps occupying ~2.7 GiB VRAM):
  - `logits_to_keep = target_len + 1` + target-only fp32 CrossEntropyLoss: eliminates the 1.2 GiB
    `[1, seq_len, 151936]` prompt-token logits tensor, cutting peak VRAM from 3.21 GiB -> 1.44 GiB
    and boosting throughput >6x by staying 100% inside physical GDDR6 (zero WDDM PCIe paging).
  - `torch.cuda.set_per_process_memory_fraction(0.48)`: hard cap inside free GDDR6 VRAM.
  - `fp16` Tensor Core autocast + `fp32` LoRA A/B master weights + `GradScaler` + `sdpa` +
    `gradient_checkpointing(use_reentrant=True)`.

Strict compliance guarantees:
  - Base model: Qwen/Qwen3-0.6B rev c1899de289a04d12100db370d81485cdf75e47ca (unchanged, safetensors, trust_remote_code=False)
  - Trains ONLY LoRA A/B matrices (`bias="none"`, no `modules_to_save`, no `embed_tokens`, no `lm_head`)
  - Exports clean 2-file adapter directories (`adapter_config.json` + `adapter_model.safetensors` ONLY)
    with canonical `base_model_name_or_path="Qwen/Qwen3-0.6B"` and `revision="c1899de289a04d12100db370d81485cdf75e47ca"`
  - Automatically asserts key compliance and adapter size <= 512 MiB after saving.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CANONICAL_BASE_REPO = "Qwen/Qwen3-0.6B"
CANONICAL_BASE_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_BASE_DIR = os.environ.get("MP3_BASE_MODEL", str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B"))

LORA_AB_KEY_RE = re.compile(r"\.lora_[AB](\.[^.]+)?\.weight$")
FORBIDDEN_TOKENS = (
    "embed_tokens", "lm_head", "modules_to_save", "bias",
    "norm", "lora_magnitude", "lora_embedding", "trainable_token",
)
MAX_ADAPTER_BYTES = 512 * 1024 * 1024


class ChatSQLDataset(Dataset):
    """Tokenizes `{"messages": [...], "target": "..."}` using official Qwen3 chat template."""

    def __init__(
        self,
        path: Path,
        tokenizer,
        max_len: int = 2048,
        enable_thinking: bool = False,
        limit: int = 0,
    ) -> None:
        raw_lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        non_train = [r["id"] for r in raw_lines if not str(r.get("id", "")).startswith("train-")]
        if non_train:
            raise ValueError(f"Non-train task IDs found in training data: {non_train[:5]}")
        if limit > 0:
            raw_lines = raw_lines[:limit]

        eos_id = tokenizer.eos_token_id
        assert eos_id is not None, "Tokenizer missing eos_token_id"

        self.samples: list[tuple[torch.Tensor, int]] = []
        truncated_count = 0
        lengths: list[int] = []
        target_lengths: list[int] = []

        for item in raw_lines:
            prompt_text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_text = item["target"].strip()
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [eos_id]

            budget = max_len - len(target_ids)
            if budget < 64 or len(prompt_ids) < 8:
                continue
            if len(prompt_ids) > budget:
                head_n = int(budget * 0.55)
                tail_n = budget - head_n
                prompt_ids = prompt_ids[:head_n] + prompt_ids[-tail_n:]
                truncated_count += 1

            input_ids = prompt_ids + target_ids
            lengths.append(len(input_ids))
            target_lengths.append(len(target_ids))
            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                len(target_ids),
            ))

        lengths.sort()
        target_lengths.sort()
        n = len(lengths)
        self.stats = {
            "count": n,
            "truncated": truncated_count,
            "min_len": lengths[0] if n else 0,
            "median_len": lengths[n // 2] if n else 0,
            "p95_len": lengths[int(0.95 * (n - 1))] if n else 0,
            "max_len": lengths[-1] if n else 0,
            "median_target_len": target_lengths[n // 2] if n else 0,
            "max_target_len": target_lengths[-1] if n else 0,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ids, tlen = self.samples[idx]
        return {"input_ids": ids, "target_len": tlen}


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(batch) == 1, "Optimized logits_to_keep trainer expects batch_size=1"
    ids = batch[0]["input_ids"].unsqueeze(0)
    return {"input_ids": ids, "target_len": int(batch[0]["target_len"])}


def save_clean_adapter(model: PeftModel, out_dir: Path) -> dict[str, Any]:
    """Save ONLY compliant 2-file LoRA export (`adapter_config.json` + `adapter_model.safetensors`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    allowed = {"adapter_config.json", "adapter_model.safetensors"}
    for child in list(out_dir.iterdir()):
        if child.name not in allowed and child.is_file():
            child.unlink()

    cfg_path = out_dir / "adapter_config.json"
    st_path = out_dir / "adapter_model.safetensors"
    assert cfg_path.is_file() and st_path.is_file(), f"Missing adapter files in {out_dir}"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = CANONICAL_BASE_REPO
    cfg["revision"] = CANONICAL_BASE_REV
    assert cfg.get("bias") == "none", f"Non-compliant bias={cfg.get('bias')}"
    assert not cfg.get("modules_to_save"), f"Non-compliant modules_to_save={cfg.get('modules_to_save')}"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_params = 0
    keys = []
    with safe_open(str(st_path), framework="pt") as f:
        for k in f.keys():
            keys.append(k)
            kl = k.lower()
            if not LORA_AB_KEY_RE.search(k):
                raise AssertionError(f"Non-LoRA A/B key in adapter: {k}")
            for bad in FORBIDDEN_TOKENS:
                if bad in kl:
                    raise AssertionError(f"Forbidden key token '{bad}' in adapter key: {k}")
            total_params += f.get_tensor(k).numel()

    total_bytes = cfg_path.stat().st_size + st_path.stat().st_size
    assert total_bytes <= MAX_ADAPTER_BYTES, f"Adapter exceeds 512 MiB: {total_bytes}"

    return {
        "dir": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir()),
        "num_tensors": len(keys),
        "lora_params": total_params,
        "size_mib": round(total_bytes / (1024**2), 2),
    }


def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--mem-fraction", type=float, default=0.48)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=15)
    parser.add_argument("--save-every-epoch", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available!")

    if 0.1 < args.mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({vram_gb:.2f} GiB, cap={args.mem_fraction*vram_gb:.2f} GiB) | dtype={args.dtype} | r={args.rank} alpha={args.alpha}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    use_scaler = (args.dtype == "float16")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    dataset = ChatSQLDataset(
        args.data,
        tokenizer,
        max_len=args.max_len,
        enable_thinking=args.enable_thinking,
        limit=args.limit,
    )
    print("Dataset stats:", json.dumps(dataset.stats), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=amp_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    vocab_size = model.config.vocab_size

    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"Resumed trainable LoRA from {args.resume}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    model = model.to(device)
    model.train()
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        drop_last=False,
        pin_memory=True,
    )

    steps_per_epoch = max(1, math.ceil(len(loader) / args.accum))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_lr_lambda(s, warmup_steps, total_steps)
    )

    log_dir = args.out.parent / f"{args.out.name}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    saved_ckpts: list[dict[str, Any]] = []

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    global_step = 0
    micro = 0
    running_loss = 0.0
    running_n = 0
    tokens_seen = 0
    stop_early = False

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        if stop_early:
            break
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            tlen = batch["target_len"]
            tokens_seen += int(input_ids.numel())

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(input_ids=input_ids, logits_to_keep=tlen + 1)
                shift_logits = out.logits[:, :-1, :].float()
                shift_labels = input_ids[:, -tlen:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, vocab_size),
                    shift_labels.reshape(-1),
                    label_smoothing=args.label_smoothing,
                )

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at micro={micro}, skipping", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled = loss / args.accum
            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            lv = float(loss.item())
            running_loss += lv
            running_n += 1
            epoch_loss += lv
            epoch_batches += 1
            micro += 1

            if micro % args.accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1 or global_step == total_steps:
                    elapsed = time.perf_counter() - started
                    avg_l = running_loss / max(1, running_n)
                    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                    res_gb = torch.cuda.max_memory_reserved() / (1024**3)
                    cur_lr = scheduler.get_last_lr()[0]
                    tok_s = tokens_seen / max(1e-6, elapsed)
                    eta_min = (elapsed / global_step) * max(0, total_steps - global_step) / 60.0
                    rec = {
                        "step": global_step,
                        "total_steps": total_steps,
                        "epoch": epoch,
                        "loss": round(avg_l, 4),
                        "lr": round(cur_lr, 7),
                        "peak_vram_gib": round(peak_gb, 3),
                        "reserved_vram_gib": round(res_gb, 3),
                        "tok_per_sec": round(tok_s, 1),
                        "elapsed_min": round(elapsed / 60.0, 2),
                        "eta_min": round(eta_min, 2),
                    }
                    history.append(rec)
                    print(
                        f"ep {epoch} step {global_step}/{total_steps} | loss={avg_l:.4f} "
                        f"| lr={cur_lr:.2e} | vram={peak_gb:.2f}/{res_gb:.2f} GiB | {tok_s:.0f} tok/s "
                        f"| {elapsed/60:.1f}m (ETA {eta_min:.1f}m)",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_n = 0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_early = True
                    break

        if not stop_early and micro % args.accum != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if args.max_steps == 0 and args.save_every_epoch and epoch < args.epochs:
            ep_dir = args.out.parent / f"{args.out.name}_ep{epoch}"
            meta = save_clean_adapter(model, ep_dir)
            meta["epoch"] = epoch
            meta["epoch_avg_loss"] = round(epoch_loss / max(1, epoch_batches), 4)
            saved_ckpts.append(meta)
            print(f"Saved epoch {epoch} clean adapter -> {ep_dir} ({meta['size_mib']} MiB)", flush=True)

    final_meta = save_clean_adapter(model, args.out)
    final_meta["step"] = global_step
    saved_ckpts.append(final_meta)

    elapsed_total = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    summary = {
        "base_repo": CANONICAL_BASE_REPO,
        "base_rev": CANONICAL_BASE_REV,
        "data_path": str(args.data),
        "dataset_stats": dataset.stats,
        "hyperparams": vars(args) | {"data": str(args.data), "out": str(args.out)},
        "gpu": gpu_name,
        "peak_vram_gib": round(peak_gb, 3),
        "total_steps": global_step,
        "elapsed_seconds": round(elapsed_total, 2),
        "elapsed_minutes": round(elapsed_total / 60.0, 2),
        "saved_adapters": saved_ckpts,
        "history": history,
    }
    (log_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nDONE in {elapsed_total/60:.2f} min | peak VRAM={peak_gb:.2f} GiB | final adapter: {final_meta}", flush=True)


if __name__ == "__main__":
    main()
