"""Fit FPReranker logistic weights strictly on `train` (480 tasks, 118 DBs).

Uses 5-fold DB-grouped cross-validation strictly inside `train` to select L2
regularization and report out-of-DB AUC / FP detection metrics. NEVER touches
`validation` or `test`.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(os.environ.get("MP3_PACKAGE_DIR", str(ROOT / "MP3_SQL_student_package")))
HARNESS_DIR = ROOT / "submission"

sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HARNESS_DIR))

from mp3_sql.common import load_bank, result_fingerprint
from db_probe import build_probe_query, parse_probe_result
from reranker import FEATURE_NAMES, extract_reranker_features
from static_audit import SchemaCatalog

DATA_DIR = PACKAGE / "mp3_sql" / "data"
OUT_WEIGHTS = HARNESS_DIR / "reranker_weights.json"


def fast_exec(db_path: Path, sql: str) -> dict:
    try:
        con = sqlite3.connect(db_path.as_uri() + "?mode=ro&immutable=1", uri=True)
        con.execute("PRAGMA query_only=ON")
        cur = con.execute(sql)
        rows = cur.fetchmany(5001)
        cols = [d[0] for d in (cur.description or [])]
        con.close()
        if len(rows) > 5000:
            return {"ok": False, "columns": [], "rows": [], "row_count": 0, "error": "Result exceeds 5000 rows"}
        return {"ok": True, "columns": cols, "rows": [list(r) for r in rows], "row_count": len(rows), "error": None}
    except Exception as e:
        return {"ok": False, "columns": [], "rows": [], "row_count": 0, "error": str(e)[:200]}


def eval_all3(row: dict, cases_for_task: list[dict], sql: str):
    pub_res = None
    pub_pass = False
    all3 = True
    for i, c in enumerate(cases_for_task):
        res = fast_exec(DATA_DIR / c["path"], sql)
        ok = bool(res["ok"] and result_fingerprint(res, row["ordered"]) == c["expected"])
        if i == 0:
            pub_res = res
            pub_res["matches_example"] = ok
            pub_pass = ok
        if not ok:
            all3 = False
    return pub_pass, all3, pub_res


def fit_logistic_and_pairwise(
    X: list[list[float]],
    y: list[int],
    tids: list[str],
    l2: float = 1.0,
    pair_weight: float = 1.5,
    steps: int = 3000,
    lr: float = 0.25,
):
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    # Pre-index mixed (TP, FP) pairs within the same task
    by_tid = defaultdict(list)
    for idx, tid in enumerate(tids):
        by_tid[tid].append(idx)
    pairs = []
    for idxs in by_tid.values():
        pos = [i for i in idxs if y[i] == 1]
        neg = [i for i in idxs if y[i] == 0]
        for i in pos:
            for j in neg:
                pairs.append((i, j))
    n_pairs = max(1, len(pairs))

    for step in range(steps):
        gb = 0.0
        gw = [0.0] * d
        # Pointwise binary cross-entropy
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            z = max(-20.0, min(20.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = (p - yi) / n
            gb += err
            for j in range(d):
                gw[j] += err * xi[j]
        # Pairwise Bradley-Terry loss: maximize P(score(pos) > score(neg))
        if pairs and pair_weight > 0:
            for i, k in pairs:
                xi, xk = X[i], X[k]
                dz = sum(w[j] * (xi[j] - xk[j]) for j in range(d))
                dz = max(-20.0, min(20.0, dz))
                p_pair = 1.0 / (1.0 + math.exp(-dz))
                perr = pair_weight * (p_pair - 1.0) / n_pairs
                for j in range(d):
                    gw[j] += perr * (xi[j] - xk[j])
        step_lr = lr / (1.0 + 0.0005 * step)
        b -= step_lr * gb
        for j in range(d):
            w[j] -= step_lr * (gw[j] + (l2 / n) * w[j])
    return b, w


def roc_auc(probs: list[float], labels: list[int]) -> float:
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main():
    rows, manifest, folder = load_bank("train", DATA_DIR)
    cases = json.loads((folder / "train_references.json").read_text(encoding="utf-8"))
    hints = json.loads((folder / "repair_hints.json").read_text(encoding="utf-8"))

    samples = []  # (db_id, tid, source, feat_vec, y_true)
    for row in rows:
        tid = row["id"]
        db_id = row["db_id"]
        meta = manifest["databases"][db_id]
        schema = meta["schema"]
        cat = SchemaCatalog(schema)
        pub_db = DATA_DIR / meta["public_path"]
        hs = [h.strip() for h in hints.get(tid, {}).get("suggestions", []) if isinstance(h, str) and h.strip()]

        # Run single-call db_probe once per task (exact inference behavior!)
        probe_sql, plan = build_probe_query(schema, row["question"], [row["issue_sql"]] + hs[:2], row["example"]["columns"])
        probe_rep = parse_probe_result(fast_exec(pub_db, probe_sql), plan) if probe_sql else None

        seen = set()
        cands = [("draft", row["issue_sql"])] + [(f"hint_{i}", h) for i, h in enumerate(hs)]
        for src, sql in cands:
            if sql in seen:
                continue
            seen.add(sql)
            pp, a3, pres = eval_all3(row, cases["tasks"][tid], sql)
            if not pp:
                continue
            feats, _ = extract_reranker_features(
                sql=sql,
                source=src,
                cat=cat,
                question=row["question"],
                ordered=row["ordered"],
                example=row["example"],
                feedback=pres,
                probe_report=probe_rep,
                hint_sqls=hs,
            )
            vec = [feats[name] for name in FEATURE_NAMES]
            samples.append((db_id, tid, src, vec, 1 if a3 else 0))

    print(f"Collected {len(samples)} public-passing train candidates: TP={sum(s[4] for s in samples)}, FP={sum(1-s[4] for s in samples)}")

    # 5-fold DB-grouped CV
    dbs = sorted({s[0] for s in samples})
    db_fold = {db: i % 5 for i, db in enumerate(dbs)}

    best_l2 = 2.0
    best_score = -1.0
    best_auc = -1.0
    for l2 in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        oof_probs, oof_y = [], []
        oof_by_tid = defaultdict(list)
        for fold in range(5):
            tr = [s for s in samples if db_fold[s[0]] != fold]
            va = [s for s in samples if db_fold[s[0]] == fold]
            b, w = fit_logistic_and_pairwise([s[3] for s in tr], [s[4] for s in tr], [s[1] for s in tr], l2=l2)
            for s in va:
                z = b + sum(wj * xj for wj, xj in zip(w, s[3]))
                prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
                oof_probs.append(prob)
                oof_y.append(s[4])
                oof_by_tid[s[1]].append((prob, s[4]))
        auc = roc_auc(oof_probs, oof_y)
        mixed_total = 0
        mixed_wins = 0
        for tid, items in oof_by_tid.items():
            ys = {y for _, y in items}
            if ys == {0, 1}:
                mixed_total += 1
                if max(items, key=lambda t: t[0])[1] == 1:
                    mixed_wins += 1
        comb = auc + 0.5 * (mixed_wins / max(1, mixed_total))
        print(f"  l2={l2:5.1f} | 5-fold DB-grouped OOF ROC-AUC={auc:.4f} | mixed-task pick TP={mixed_wins}/{mixed_total}")
        if comb > best_score:
            best_score = comb
            best_auc = auc
            best_l2 = l2

    # Fit final weights on all train samples using best_l2
    b, w = fit_logistic_and_pairwise([s[3] for s in samples], [s[4] for s in samples], [s[1] for s in samples], l2=best_l2)
    weight_map = {name: round(val, 5) for name, val in zip(FEATURE_NAMES, w)}
    # Ensure positive prior weight for is_repaired_or_llm (0 occurrences in raw draft+hint train set)
    if weight_map.get("is_repaired_or_llm", 0.0) == 0.0:
        weight_map["is_repaired_or_llm"] = round(max(0.25, weight_map.get("is_hint", 0.3)), 5)

    payload = {
        "fitted_on": "train_480_only",
        "n_candidates": len(samples),
        "n_tp": sum(s[4] for s in samples),
        "n_fp": sum(1 - s[4] for s in samples),
        "l2": best_l2,
        "oof_db_grouped_auc": round(best_auc, 4),
        "bias": round(b, 5),
        "weights": weight_map,
    }
    OUT_WEIGHTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {OUT_WEIGHTS} (best l2={best_l2}, OOF AUC={best_auc:.4f}):")
    print(f"  bias = {b:.4f}")
    for k, v in sorted(weight_map.items(), key=lambda kv: kv[1]):
        print(f"  {k:28s}: {v:+.4f}")


if __name__ == "__main__":
    main()
