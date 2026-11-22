"""Compliant single-LLM inference runner for harness ablations and final submission.

Strict compliance:
  - Exactly ONE live LLM loader (`AutoModelForCausalLM.from_pretrained(..., local_files_only=True,
    trust_remote_code=False, use_safetensors=True)` + optional `PeftModel.from_pretrained`).
  - Hard caps: <= 6 completions per task, <= 2048 new tokens per completion.
  - Mandatory JSONL logging via `MP3_LLM_LOG` (`task_id`, `prompt`, `generated_text`,
    `prompt_tokens`, `completion_tokens`, `model_id`, `adapter_id`).
  - Supports TWO prompt styles (via config, NOT two models):
      * `"v2_rich_chat"`: Qwen3 chat template with `prompt_format.build_messages`
        (for new rich LoRA v12+ or zero-shot Qwen3-0.6B with thinking=False/True).
      * `"legacy_raw"`: DeepSeek's raw `### Schema ... ### Corrected SQL` template
        (solely for apples-to-apples controlled harness ablation with `train/lora-e2`).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from prompt_format import extract_sql_from_output

CANONICAL_MODEL_ID = "Qwen/Qwen3-0.6B"
CANONICAL_MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_COMPLETIONS_PER_TASK = 6
MAX_NEW_TOKENS_CAP = 2048

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}
_TASK_CALLS: dict[str, int] = defaultdict(int)


def _base_dir() -> str:
    env_val = os.environ.get("MP3_BASE_MODEL", "").strip()
    if env_val:
        return env_val
    for candidate in ("models/Qwen3-0.6B", "../models/Qwen3-0.6B"):
        if Path(candidate).is_dir():
            return candidate
    return "models/Qwen3-0.6B"


def _adapter_dir() -> str:
    return os.environ.get("MP3_ADAPTER_DIR", "").strip()


def available(require_adapter: bool = False) -> bool:
    bdir = _base_dir()
    if not bdir or not Path(bdir).is_dir():
        return False
    adir = _adapter_dir()
    if require_adapter:
        if not adir:
            return False
        ap = Path(adir)
        return (ap / "adapter_config.json").is_file() and (ap / "adapter_model.safetensors").is_file()
    if adir:
        ap = Path(adir)
        return (ap / "adapter_config.json").is_file() and (ap / "adapter_model.safetensors").is_file()
    return True


def remaining_completions(task_id: str) -> int:
    return max(0, MAX_COMPLETIONS_PER_TASK - _TASK_CALLS[task_id])


def reset_task_budget(task_id: str | None = None) -> None:
    with _LOCK:
        if task_id is None:
            _TASK_CALLS.clear()
        else:
            _TASK_CALLS.pop(task_id, None)


def _load(threads: int = 4):
    bdir = _base_dir()
    adir = _adapter_dir()
    dname = os.environ.get("MP3_DTYPE", "float32").lower()
    cache_key = (bdir, adir, dname)
    if _CACHE.get("key") == cache_key:
        return _CACHE["model"], _CACHE["tokenizer"]

    with _LOCK:
        if _CACHE.get("key") == cache_key:
            return _CACHE["model"], _CACHE["tokenizer"]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dname, torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(
            bdir, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            bdir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch_dtype,
        )
        adapter_id = None
        if adir:
            from peft import PeftModel
            ap = Path(adir)
            model = PeftModel.from_pretrained(model, str(ap), is_trainable=False)
            adapter_id = ap.name
        model.eval()

        model_id = os.environ.get("MP3_MODEL_ID", CANONICAL_MODEL_ID)
        _CACHE.clear()
        _CACHE.update({
            "key": cache_key,
            "model": model,
            "tokenizer": tokenizer,
            "torch": torch,
            "model_id": model_id,
            "adapter_id": adapter_id,
        })
    return _CACHE["model"], _CACHE["tokenizer"]


def _log_record(record: dict[str, Any]) -> None:
    log_path = os.environ.get("MP3_LLM_LOG", "").strip() or "outputs/llm_usage.jsonl"
    try:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _legacy_render_schema(schema: list[dict]) -> str:
    lines = []
    for table in schema:
        parts = []
        for col in table.get("columns", []):
            piece = f"{col['name']} {col.get('type', '')}".strip()
            if col.get("pk"):
                piece += " PK"
            parts.append(piece)
        line = f"{table['name']}({', '.join(parts)})"
        fks = [f"{fk['from']}->{fk['table']}.{fk['to']}" for fk in (table.get("foreign_keys") or [])]
        if fks:
            line += "  FK: " + ", ".join(fks)
        lines.append(line)
    return "\n".join(lines)


def build_legacy_raw_prompt(
    question: str,
    schema: list[dict],
    source_sql: str,
    feedback: dict | None,
    example: dict,
) -> str:
    fb = feedback or {}
    if not fb.get("ok"):
        err = fb.get("error") or {}
        if isinstance(err, dict):
            fb_text = f"Execution error: {err.get('type', 'Error')}: {str(err.get('message', ''))[:200]}"
        else:
            fb_text = f"Execution error: {str(err)[:200]}"
    elif fb.get("matches_example"):
        fb_text = "The draft already returns the expected result."
    else:
        fb_text = f"The draft runs but returns the wrong result ({fb.get('row_count', 0)} rows)."

    columns = example.get("columns") or []
    rows = (example.get("rows") or [])[:5]
    example_text = (
        f"columns={columns} rows={json.dumps(rows, ensure_ascii=False)} "
        f"row_count={example.get('row_count')}"
    )
    return (
        "You are a SQLite expert. Fix the draft query so that it answers the question "
        "on the given database schema. Reply with one SQLite query only, no explanation.\n\n"
        f"### Schema\n{_legacy_render_schema(schema)}\n\n"
        f"### Question\n{question}\n\n"
        f"### Draft\n{source_sql.strip()}\n\n"
        f"### Execution feedback\n{fb_text}\n\n"
        f"### Expected output\n{example_text}\n\n"
        f"### Corrected SQL\n"
    )


def _trim_eos_tokens(gen_ids, eos_id: int | None):
    gen_list = gen_ids.tolist()
    if eos_id is not None and eos_id in gen_list:
        comp_tokens = gen_list.index(eos_id) + 1
    else:
        comp_tokens = len(gen_list)
    return gen_ids[:comp_tokens], comp_tokens


def _clean_greedy_gen_config(model) -> None:
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(gc, attr):
                setattr(gc, attr, None)


def generate_legacy_raw(
    *,
    task_id: str,
    prompt: str,
    max_new_tokens: int = 112,
    n: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> list[str]:
    rem = remaining_completions(task_id)
    if rem <= 0:
        return []
    n_run = min(max(1, n), rem)
    max_new = min(max(16, max_new_tokens), MAX_NEW_TOKENS_CAP)

    model, tokenizer = _load()
    torch = _CACHE["torch"]
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(inputs["input_ids"].shape[1])

    t0 = time.perf_counter()
    with torch.inference_mode():
        if n_run == 1:
            _clean_greedy_gen_config(model)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            torch.manual_seed(42 + _TASK_CALLS[task_id])
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                num_return_sequences=n_run,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
    dt = time.perf_counter() - t0

    results: list[str] = []
    for i in range(output.shape[0]):
        trimmed, comp_tokens = _trim_eos_tokens(output[i][prompt_len:], tokenizer.eos_token_id)
        raw_text = tokenizer.decode(trimmed, skip_special_tokens=True)
        with _LOCK:
            _TASK_CALLS[task_id] += 1
        _log_record({
            "task_id": task_id,
            "mode": "legacy_raw",
            "model_id": _CACHE.get("model_id", CANONICAL_MODEL_ID),
            "model_revision": CANONICAL_MODEL_REV,
            "adapter_id": _CACHE.get("adapter_id"),
            "prompt": prompt,
            "generated_text": raw_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": comp_tokens,
            "elapsed_sec": round(dt / output.shape[0], 3),
        })
        sql = extract_sql_from_output(raw_text)
        if sql and sql not in results:
            results.append(sql)
    return results


def _sample_next_token(logits, temperature, top_p: float, top_k: int, torch):
    if isinstance(temperature, (int, float)):
        scaled = logits / max(0.05, float(temperature))
    else:
        scaled = logits / temperature
    if top_k > 0 and top_k < scaled.shape[-1]:
        topk_vals, _ = torch.topk(scaled, top_k, dim=-1)
        min_topk = topk_vals[:, -1:].expand_as(scaled)
        scaled = torch.where(scaled < min_topk, torch.full_like(scaled, -float("inf")), scaled)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled_in_sorted = torch.multinomial(probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_in_sorted)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _kv_shared_sample(model, input_ids, n_run: int, max_new: int, eos_id: int, temperature: float, top_p: float, top_k: int, torch):
    """True 1x-prefill KV-cache-shared batched decoding for N sequences on CPU.

    When temperature >= 0.6 and n_run >= 2, uses a diverse temperature ladder across the
    batch (e.g. [0.80, 1.00, 1.20, 1.38] with top_p=0.97, top_k=25) so sharp r=32 LoRA
    distributions explore distinct valid SQL structures instead of collapsing to 4 greedy copies.
    """
    pref = model(input_ids=input_ids, use_cache=True)
    pkv = pref.past_key_values
    if hasattr(pkv, "batch_repeat_interleave"):
        pkv.batch_repeat_interleave(n_run)
    else:
        return model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=max(0.8, temperature),
            top_p=0.97,
            top_k=25,
            num_return_sequences=n_run,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )[:, input_ids.shape[1]:]

    if temperature >= 0.6 and n_run >= 2:
        base_t = max(0.80, float(temperature))
        step_t = 0.20
        temp_vec = torch.tensor(
            [[min(1.45, base_t + i * step_t)] for i in range(n_run)],
            dtype=pref.logits.dtype,
            device=pref.logits.device,
        )
        eff_top_p = 0.97
        eff_top_k = 25
    else:
        temp_vec = max(0.05, float(temperature))
        eff_top_p = top_p
        eff_top_k = top_k

    first_logits = pref.logits[:, -1, :].expand(n_run, -1).contiguous()
    nxt = _sample_next_token(first_logits, temp_vec, eff_top_p, eff_top_k, torch)
    generated = [nxt]
    done = nxt.squeeze(-1) == eos_id

    for _ in range(max_new - 1):
        if bool(done.all()):
            break
        step_out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv = step_out.past_key_values
        nxt = _sample_next_token(step_out.logits[:, -1, :], temp_vec, eff_top_p, eff_top_k, torch)
        nxt = torch.where(done.unsqueeze(-1), torch.full_like(nxt, eos_id), nxt)
        generated.append(nxt)
        done = done | (nxt.squeeze(-1) == eos_id)

    return torch.cat(generated, dim=1)


def generate_chat_raw_batch(
    *,
    task_id: str,
    messages: list[dict[str, str]],
    mode: str = "repair",
    max_new_tokens: int = 192,
    n: int = 1,
    temperature: float = 0.65,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> list[str]:
    """Generate `n` SQL strings (preserving duplicates so caller can count self-consistency votes!)."""
    rem = remaining_completions(task_id)
    if rem <= 0:
        return []
    n_run = min(max(1, n), rem)
    max_new = min(max(16, max_new_tokens), MAX_NEW_TOKENS_CAP)

    model, tokenizer = _load()
    torch = _CACHE["torch"]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(inputs["input_ids"].shape[1])

    t0 = time.perf_counter()
    with torch.inference_mode():
        if n_run == 1 and temperature <= 0.0:
            _clean_greedy_gen_config(model)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_matrix = output[:, prompt_len:]
        elif n_run == 1:
            torch.manual_seed(42 + _TASK_CALLS[task_id])
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_matrix = output[:, prompt_len:]
        else:
            torch.manual_seed(42 + _TASK_CALLS[task_id])
            gen_matrix = _kv_shared_sample(
                model,
                inputs["input_ids"],
                n_run=n_run,
                max_new=max_new,
                eos_id=tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                torch=torch,
            )
    dt = time.perf_counter() - t0

    sqls: list[str] = []
    for i in range(gen_matrix.shape[0]):
        trimmed, comp_tokens = _trim_eos_tokens(gen_matrix[i], tokenizer.eos_token_id)
        raw_text = tokenizer.decode(trimmed, skip_special_tokens=True)
        with _LOCK:
            _TASK_CALLS[task_id] += 1
        _log_record({
            "task_id": task_id,
            "mode": mode,
            "enable_thinking": enable_thinking,
            "model_id": _CACHE.get("model_id", CANONICAL_MODEL_ID),
            "model_revision": CANONICAL_MODEL_REV,
            "adapter_id": _CACHE.get("adapter_id"),
            "prompt": prompt,
            "generated_text": raw_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": comp_tokens,
            "elapsed_sec": round(dt / gen_matrix.shape[0], 3),
        })
        sql = extract_sql_from_output(raw_text)
        if sql:
            sqls.append(sql)
    return sqls


def generate_chat(
    *,
    task_id: str,
    messages: list[dict[str, str]],
    mode: str = "repair",
    max_new_tokens: int = 192,
    n: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> list[str]:
    eff_temp = 0.0 if n == 1 else temperature
    batch = generate_chat_raw_batch(
        task_id=task_id,
        messages=messages,
        mode=mode,
        max_new_tokens=max_new_tokens,
        n=n,
        temperature=eff_temp,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )
    return list(dict.fromkeys(batch))
