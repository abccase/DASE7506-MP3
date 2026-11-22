"""Build `train_rich_v3.jsonl` (strictly train-only, 480 tasks / 118 DBs) with calibrated
`fp_audit` Candidate-Hints prior and targeted compositional corruptors:
  1. `fp_audit` Candidate-Hints alignment: places synthetic & real FPs inside `### Candidate Hints`
     (`[Hint 1 (votes=3..4)] ... -> MATCHES_EXAMPLE`) matching exact `harness_agent.py` inference layout,
     with ~72% FP-rewrite vs ~28% clean-only TP-keep (zero red-flag TPs!).
  2. `_corrupt_literal_wrong_column`: moves string literal to sibling or FK-neighbor text column so
     `db_probe` + `static_audit` emit authentic `LITERAL_WRONG_COLUMN` warnings -> gold_sql.
  3. `_corrupt_negation_to_join`: rewrites `NOT IN (SELECT ...)` and `EXCEPT` train gold SQLs into
     realistic `JOIN ... WHERE col != 'val'` and `JOIN ... WHERE col NOT LIKE '%val%'` anti-patterns -> gold_sql.
  4. `_corrupt_scalar_subquery`: rewrites `WHERE col > (SELECT AVG/MAX/MIN(col) FROM T)` into realistic
     broken `HAVING` / self-join / unaggregated subquery forms -> gold_sql.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(os.environ.get("MP3_PACKAGE_DIR", str(ROOT / "MP3_SQL_student_package")))
HARNESS_DIR = ROOT / "submission"
LORA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(LORA_DIR))

from mp3_sql.common import result_fingerprint  # noqa: E402
from db_probe import build_probe_query, parse_probe_result  # noqa: E402
from prompt_format import build_messages  # noqa: E402
from static_audit import SchemaCatalog, audit_sql  # noqa: E402
from build_rich_dataset import (  # noqa: E402
    FastSQLitePool,
    clip_feedback,
    ensure_semicolon,
    generate_corruptions as base_generate_corruptions,
    sha256_file,
)


def _corrupt_literal_wrong_column(
    tree: exp.Expression, schema: list[dict[str, Any]], rng: random.Random
) -> list[tuple[str, str]]:
    """Move a WHERE string literal onto a wrong sibling text column OR a joined FK-neighbor text column."""
    out: list[tuple[str, str]] = []
    if not isinstance(tree, exp.Select):
        return out
    where = tree.args.get("where")
    if where is None:
        return out

    tbl_map = {t["name"].lower(): t for t in schema}
    alias_to_tbl = {(t.alias or t.name).lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    used_tbls = set(alias_to_tbl.values())

    for eq in list(where.find_all(exp.EQ)):
        col_node, lit_node = None, None
        if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Literal) and eq.expression.is_string:
            col_node, lit_node = eq.this, eq.expression
        elif isinstance(eq.expression, exp.Column) and isinstance(eq.this, exp.Literal) and eq.this.is_string:
            col_node, lit_node = eq.expression, eq.this
        if col_node is None or lit_node is None:
            continue

        val = str(lit_node.this).strip()
        if len(val) < 2:
            continue

        # Resolve owning table of col_node
        t_low = alias_to_tbl.get((col_node.table or "").lower(), "")
        if not t_low:
            owners = [tl for tl in used_tbls if any(c["name"].lower() == col_node.name.lower() for c in tbl_map.get(tl, {}).get("columns", []))]
            if len(owners) == 1:
                t_low = owners[0]
        if not t_low or t_low not in tbl_map:
            continue

        # Variant A: Sibling text column in the same table (e.g. ResTypeName vs ResName, LName vs FName)
        sib_cols = [
            c["name"] for c in tbl_map[t_low]["columns"]
            if c["name"].lower() != col_node.name.lower()
            and any(k in (c.get("type") or "text").lower() for k in ("text", "char", "var", ""))
            and not c.get("pk")
            and not c["name"].lower().endswith("id")
        ]
        if sib_cols:
            v = copy.deepcopy(tree)
            for m in v.args["where"].find_all(exp.EQ):
                if m.sql() == eq.sql():
                    target_c = m.this if isinstance(m.this, exp.Column) else m.expression
                    target_c.set("this", exp.to_identifier(rng.choice(sib_cols)))
                    out.append(("literal_wrong_col_sibling", v.sql(dialect="sqlite")))
                    break

        # Variant B: Move literal to an FK-connected neighbor table's text column (adding JOIN if needed!)
        for tb in schema:
            nb_low = tb["name"].lower()
            if nb_low == t_low:
                continue
            # Check FK in either direction between t_low and nb_low
            fk_pair = None
            for fk in tb.get("foreign_keys") or []:
                if fk["table"].lower() == t_low:
                    fk_pair = (tb["name"], fk["from"], fk["to"])
                    break
            if fk_pair is None:
                for fk in tbl_map[t_low].get("foreign_keys") or []:
                    if fk["table"].lower() == nb_low:
                        fk_pair = (tb["name"], fk["to"], fk["from"])
                        break
            if fk_pair is None:
                continue

            nb_txt_cols = [
                c["name"] for c in tb["columns"]
                if any(k in (c.get("type") or "text").lower() for k in ("text", "char", "var"))
                and not c.get("pk")
                and not c["name"].lower().endswith("id")
            ]
            if not nb_txt_cols:
                continue

            nb_name, nb_fk_col, orig_fk_col = fk_pair
            pick_col = rng.choice(nb_txt_cols)
            v2 = copy.deepcopy(tree)
            orig_alias = next((a for a, r in alias_to_tbl.items() if r == t_low), tbl_map[t_low]["name"])
            if nb_low not in used_tbls:
                v2.join(
                    exp.table_(nb_name),
                    on=exp.EQ(
                        this=exp.column(nb_fk_col, table=nb_name),
                        expression=exp.column(orig_fk_col, table=orig_alias),
                    ),
                    copy=False,
                )
                nb_alias = nb_name
            else:
                nb_alias = next((a for a, r in alias_to_tbl.items() if r == nb_low), nb_name)

            for m in v2.args["where"].find_all(exp.EQ):
                if m.sql() == eq.sql():
                    target_c = m.this if isinstance(m.this, exp.Column) else m.expression
                    target_c.set("this", exp.to_identifier(pick_col))
                    target_c.set("table", exp.to_identifier(nb_alias))
                    out.append(("literal_wrong_col_joined", v2.sql(dialect="sqlite")))
                    break
            break
        break
    return out


def _corrupt_negation_to_join(gold: str, tree: exp.Expression) -> list[tuple[str, str]]:
    """Rewrite `NOT IN (SELECT ...)` and `EXCEPT` into realistic `JOIN ... WHERE !=` and `NOT LIKE`."""
    out: list[tuple[str, str]] = []

    # Pattern 1: SELECT <cols> FROM <T1> WHERE <k1> NOT IN (SELECT <k2> FROM <T2> WHERE <cond>)
    m_notin = re.search(
        r"^SELECT\s+(.*?)\s+FROM\s+(\w+)(?:\s+AS\s+\w+)?\s+WHERE\s+(?:\w+\.)?(\w+)\s+NOT\s+IN\s*\(\s*SELECT\s+(?:\w+\.)?(\w+)\s+FROM\s+(\w+)(?:\s+AS\s+\w+)?(?:\s+WHERE\s+(.*?))?\s*\)\s*;?$",
        gold.strip(),
        flags=re.I | re.S,
    )
    if m_notin:
        sel_cols, t1, k1, k2, t2, sub_where = m_notin.groups()
        # Clean T1. prefix in sel_cols if any
        clean_sel = re.sub(r"\bT\d+\.", f"{t1}.", sel_cols)
        if sub_where:
            clean_w = re.sub(r"\bT\d+\.", f"{t2}.", sub_where.strip())
            neq_w = re.sub(r"=", "!=", clean_w, count=1)
            bad1 = f"SELECT DISTINCT {clean_sel} FROM {t1} JOIN {t2} ON {t1}.{k1} = {t2}.{k2} WHERE {neq_w}"
            out.append(("notin_to_join_neq", bad1))
            m_str = re.search(r"([A-Za-z0-9_.]+)\s*=\s*['\"]([^'\"]+)['\"]", clean_w)
            if m_str:
                wcol, wval = m_str.groups()
                if "." not in wcol:
                    wcol = f"{t2}.{wcol}"
                bad2 = f"SELECT DISTINCT {clean_sel} FROM {t1} JOIN {t2} ON {t1}.{k1} = {t2}.{k2} WHERE {wcol} NOT LIKE '%{wval}%'"
                out.append(("notin_to_join_notlike", bad2))
        else:
            bad_left = f"SELECT {clean_sel} FROM {t1} JOIN {t2} ON {t1}.{k1} != {t2}.{k2}"
            out.append(("notin_to_neq_join", bad_left))

    # Pattern 2: Any EXCEPT (top-level or nested) with a right SELECT having WHERE
    for exc_node in tree.find_all(exp.Except):
        right = exc_node.expression
        if isinstance(right, exp.Select) and right.args.get("where"):
            v1 = copy.deepcopy(right)
            v1.set("distinct", exp.Distinct())
            replaced = False
            for eq in v1.args["where"].find_all(exp.EQ):
                eq.replace(exp.NEQ(this=eq.left.copy(), expression=eq.right.copy()))
                replaced = True
                break
            if replaced:
                out.append(("except_to_join_neq", v1.sql(dialect="sqlite")))

            v2 = copy.deepcopy(right)
            v2.set("distinct", exp.Distinct())
            for eq in v2.args["where"].find_all(exp.EQ):
                if isinstance(eq.expression, exp.Literal) and eq.expression.is_string:
                    val = str(eq.expression.this)
                    eq.replace(exp.Not(this=exp.Like(this=eq.left.copy(), expression=exp.Literal.string(f"%{val}%"))))
                    out.append(("except_to_join_notlike", v2.sql(dialect="sqlite")))
                    break
        elif isinstance(right, exp.Select):
            # Simple 2-table EXCEPT without WHERE -> corrupt to LEFT JOIN without IS NULL or !=
            out.append(("except_to_union", exp.Union(this=copy.deepcopy(exc_node.this), expression=copy.deepcopy(right), distinct=True).sql(dialect="sqlite")))
    return out


def _corrupt_scalar_subquery(gold: str, tree: exp.Expression) -> list[tuple[str, str]]:
    """Corrupt `WHERE col > (SELECT AVG/MAX/MIN(col) FROM ...)` into realistic HAVING / unaggregated bugs."""
    out: list[tuple[str, str]] = []
    m = re.search(
        r"^(SELECT\s+.*?\s+FROM\s+(\w+).*?)\s+WHERE\s+([A-Za-z0-9_.]+)\s*([<>]=?)\s*\(\s*SELECT\s+(AVG|MAX|MIN)\s*\(\s*([A-Za-z0-9_.]+)\s*\)\s+FROM\s+\w+(?:\s+WHERE\s+.*?)?\s*\)(.*)$",
        gold.strip().rstrip(";"),
        flags=re.I | re.S,
    )
    if m:
        head, tbl, col, op, agg, agg_col, tail = m.groups()
        if "group by" in tail.lower():
            out.append(("subq_agg_to_having", f"{head} {tail.strip()} HAVING {col} {op} {agg}({agg_col})"))
        else:
            out.append(("subq_agg_to_having", f"{head} GROUP BY {col} HAVING {col} {op} {agg}({agg_col}) {tail.strip()}"))
        # Unaggregated subquery (classic scalar subquery bug!)
        bad_noagg = re.sub(r"\(\s*SELECT\s+(?:AVG|MAX|MIN)\s*\(\s*([^)]+)\s*\)", r"(SELECT \1", gold.strip().rstrip(";"), count=1, flags=re.I)
        if bad_noagg != gold.strip().rstrip(";"):
            out.append(("subq_drop_agg", bad_noagg))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"E:\MP3\exp\lora\data\train_rich_v3.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = PACKAGE / "mp3_sql" / "data"

    train_path = data_dir / "train.jsonl"
    refs_path = data_dir / "train_references.json"
    hints_path = data_dir / "repair_hints.json"
    pub_path = data_dir / "public_results.json"
    manifest_path = data_dir / "manifest.json"

    train_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_ids = {r["id"] for r in train_rows}
    train_dbs = {r["db_id"] for r in train_rows}
    assert all(tid.startswith("train-") for tid in train_ids) and len(train_rows) == 480

    train_refs = json.loads(refs_path.read_text(encoding="utf-8"))
    gold_map = {k: v for k, v in train_refs["gold_sql"].items() if k in train_ids}
    ref_cases = {k: v for k, v in train_refs["tasks"].items() if k in train_ids}
    hints_map = {k: v for k, v in json.loads(hints_path.read_text(encoding="utf-8")).items() if k in train_ids}
    pub_map = {k: v for k, v in json.loads(pub_path.read_text(encoding="utf-8")).items() if k in train_ids}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    db_meta_map = {k: v for k, v in manifest["databases"].items() if k in train_dbs}

    pool = FastSQLitePool()

    def eval_sql(row: dict[str, Any], sql: str) -> tuple[bool, bool, dict[str, Any]]:
        meta = db_meta_map[row["db_id"]]
        pub_out = pool.execute(data_dir / meta["public_path"], sql)
        pub_ok = bool(pub_out["ok"] and result_fingerprint(pub_out, row["ordered"]) == pub_map[row["id"]])
        pub_out["matches_example"] = pub_ok
        all3 = pub_ok
        if all3:
            for c in ref_cases[row["id"]]:
                res = pool.execute(data_dir / c["path"], sql)
                if not (res["ok"] and result_fingerprint(res, row["ordered"]) == c["expected"]):
                    all3 = False
                    break
        return pub_ok, all3, clip_feedback(pub_out)

    def run_db_probe(db_path: Path, schema: list[dict], question: str, sqls: list[str], ex_cols: list[str]) -> dict | None:
        psql, plan = build_probe_query(schema, question, sqls, ex_cols)
        if not psql:
            return None
        fb = pool.execute(db_path, psql)
        rep = parse_probe_result(fb, plan)
        return rep if rep.get("ok") else None

    def get_audit(sql: str, cat: SchemaCatalog, question: str, ordered: bool, ex_cols: list[str], fb: dict | None, prep: dict | None) -> dict:
        actual_cols = (fb.get("columns") if (fb and fb.get("ok")) else None)
        return audit_sql(sql, cat, question, ordered, ex_cols, actual_cols=actual_cols, probe_report=prep)

    records: list[dict[str, Any]] = []
    regime_counts: Counter[str] = Counter()
    corrupt_counts: Counter[str] = Counter()
    clean_tp_pool: list[tuple[int, dict[str, Any]]] = []

    def make_rec(
        tid: str, db_id: str, regime: str, question: str, schema: list[dict],
        ordered: bool, example: dict, issue_sql: str, draft_fb: dict | None,
        draft_warns: list[str] | None, hints_list: list[dict] | None,
        probe_rep: dict | None, mode: str, target: str,
        previous_attempt: dict | None = None, extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        use_probe = probe_rep if (probe_rep and rng.random() > 0.08) else None
        msgs = build_messages(
            question=question, schema=schema, ordered=ordered, example=example,
            issue_sql=issue_sql, draft_feedback=draft_fb, draft_warnings=draft_warns,
            hints=hints_list, probe_report=use_probe, mode=mode,
            previous_attempt=previous_attempt,
        )
        return {
            "id": tid, "db_id": db_id, "regime": regime, "mode": mode,
            "messages": msgs, "target": ensure_semicolon(target), "meta": extra_meta or {},
        }

    t0 = time.perf_counter()
    for row in train_rows:
        tid = row["id"]
        db_id = row["db_id"]
        gold = gold_map[tid].strip()
        meta = db_meta_map[db_id]
        schema = meta["schema"]
        cat = SchemaCatalog(schema)
        pub_db = data_dir / meta["public_path"]
        question = row["question"]
        example = row["example"]
        ex_cols = list(example.get("columns") or [])
        ordered = row["ordered"]
        rc = example.get("row_count", 0)
        draft = row["issue_sql"].strip()

        d_pub, d_3, d_fb = eval_sql(row, draft)
        raw_sug = [h.strip() for h in hints_map.get(tid, {}).get("suggestions", []) if h.strip()]
        vote_counts = Counter(raw_sug)
        uniq_hints = [h for h in dict.fromkeys(raw_sug) if h != draft][:4]
        h_evals = [(h, vote_counts[h], *eval_sql(row, h)) for h in uniq_hints]

        probe_rep = run_db_probe(pub_db, schema, question, [draft] + uniq_hints, ex_cols)
        d_aud = get_audit(draft, cat, question, ordered, ex_cols, d_fb, probe_rep)
        d_warns = d_aud["warnings"] or None

        def hint_ent(h: str, v: int, hfb: dict, checked: bool, prep: dict | None) -> dict:
            fb_u = hfb if checked else None
            w = get_audit(h, cat, question, ordered, ex_cols, fb_u, prep)["warnings"] or None
            return {"sql": h, "feedback": fb_u, "warnings": w, "votes": v}

        pub_failing_hints = [x for x in h_evals if not x[2]]
        failing_hints = [x for x in h_evals if not x[3]]

        is_compositional = bool(re.search(r"\b(EXCEPT|INTERSECT|NOT\s+IN)\b|WHERE\b.*\(\s*SELECT\s+(AVG|MAX|MIN)\b", gold, re.I | re.S))

        # 1. Authentic hard repair + compositional boost
        if not d_pub:
            h_list = [
                hint_ent(h, v, hfb, checked=(i == 0 or rng.random() < 0.6), prep=probe_rep)
                for i, (h, v, hp, h3, hfb) in enumerate(pub_failing_hints[:3])
            ]
            rec_r = make_rec(tid, db_id, "orig_hard_repair", question, schema, ordered, example,
                             draft, d_fb, d_warns, h_list or None, probe_rep, "repair", gold, {"rc": rc})
            records.append(rec_r)
            regime_counts["orig_hard_repair"] += 1
            if is_compositional:
                # Extra pass for hard compositional SQL (NOT IN / EXCEPT / INTERSECT / scalar agg subquery)
                records.append(make_rec(tid, db_id, "orig_compositional_boost", question, schema, ordered, example,
                                        draft, d_fb, d_warns, [hint_ent(h, v, hfb, True, probe_rep) for h, v, _, _, hfb in pub_failing_hints[:2]] or None,
                                        probe_rep, "repair", gold, {"rc": rc}))
                regime_counts["orig_compositional_boost"] += 1

            if not any(hp for _, _, hp, _, _ in h_evals) and pub_failing_hints:
                full_h = [hint_ent(h, v, hfb, checked=(i < 3), prep=probe_rep) for i, (h, v, _, _, hfb) in enumerate(pub_failing_hints[:4])]
                records.append(make_rec(tid, db_id, "orig_allfail_fullctx", question, schema, ordered, example,
                                        draft, d_fb, d_warns, full_h, probe_rep, "repair", gold, {"rc": rc}))
                regime_counts["orig_allfail_fullctx"] += 1

        # 2. Failing 8B Hint Repair
        for h, v, hp, h3, hfb in pub_failing_hints[:2]:
            records.append(make_rec(tid, db_id, "hint_repair", question, schema, ordered, example,
                                    draft, d_fb, d_warns, [hint_ent(h, v, hfb, True, probe_rep)], probe_rep, "repair", gold, {"rc": rc}))
            regime_counts["hint_repair"] += 1

        # 3. Real 3-Instance False Positives (EXACT harness_agent.py Candidate Hints layout!)
        if d_pub and not d_3:
            h_list = [hint_ent(h, v, hfb, checked=(i == 0), prep=probe_rep) for i, (h, v, _, _, hfb) in enumerate(failing_hints[:3])]
            records.append(make_rec(tid, db_id, "audit_fp_real", question, schema, ordered, example,
                                    draft, d_fb, d_warns, h_list or None, probe_rep, "fp_audit", gold, {"src": "draft", "rc": rc}))
            regime_counts["audit_fp_real"] += 1

        for h, v, hp, h3, hfb in h_evals:
            if hp and not h3:
                # Present BOTH with real votes AND with votes=4 (matches 4/4 consensus hint FPs!)
                h_ent1 = hint_ent(h, v, hfb, True, probe_rep)
                records.append(make_rec(tid, db_id, "audit_fp_real", question, schema, ordered, example,
                                        draft, d_fb, d_warns, [h_ent1], probe_rep, "fp_audit", gold, {"src": "hint", "rc": rc}))
                regime_counts["audit_fp_real"] += 1
                if v < 4:
                    h_ent4 = {**h_ent1, "votes": 4}
                    records.append(make_rec(tid, db_id, "audit_fp_real_v4", question, schema, ordered, example,
                                            draft, d_fb, d_warns, [h_ent4], probe_rep, "fp_audit", gold, {"src": "hint_v4", "rc": rc}))
                    regime_counts["audit_fp_real_v4"] += 1

        # 4. Collect ONLY CLEAN True-Positive Keep Examples (ZERO hard red flags!)
        def is_clean_tp(aud: dict) -> bool:
            f = aud["features"]
            return (
                f["suspicious_non_fk_joins"] == 0
                and f["cross_type_joins"] == 0
                and f["ungrounded_literals"] == 0
                and not f["ordered_missing_orderby"]
                and not f["alphabetical_id_order"]
                and not f["negation_uses_neq_on_join"]
                and f["col_count_match"]
            )

        prio = 0 if rc == 0 else (1 if rc == 1 else 2)
        tp_hints = [(h, v, hfb) for (h, v, hp, h3, hfb) in h_evals if hp and h3]
        if tp_hints:
            h_tp, v_tp, hfb_tp = tp_hints[0]
            tp_aud = get_audit(h_tp, cat, question, ordered, ex_cols, hfb_tp, probe_rep)
            if is_clean_tp(tp_aud):
                h_ent_tp = {"sql": h_tp, "feedback": hfb_tp, "warnings": tp_aud["warnings"] or None, "votes": v_tp}
                clean_tp_pool.append((prio, make_rec(
                    tid, db_id, "audit_tp_keep", question, schema, ordered, example,
                    draft, d_fb, d_warns, [h_ent_tp], probe_rep, "fp_audit", h_tp, {"src": "hint", "rc": rc},
                )))
        elif d_pub and d_3 and is_clean_tp(d_aud):
            clean_tp_pool.append((prio, make_rec(
                tid, db_id, "audit_tp_keep", question, schema, ordered, example,
                draft, d_fb, d_warns, None, probe_rep, "fp_audit", draft, {"src": "draft", "rc": rc},
            )))

        # 5. Targeted High-Signal Corruptions (including NEW wrong-column literal, negation!=, scalar-subquery!)
        try:
            g_tree = sqlglot.parse_one(gold, read="sqlite")
        except Exception:
            g_tree = None

        special_corrupts: list[tuple[str, str]] = []
        if g_tree is not None:
            special_corrupts.extend(_corrupt_literal_wrong_column(g_tree, schema, rng))
            special_corrupts.extend(_corrupt_negation_to_join(gold, g_tree))
            special_corrupts.extend(_corrupt_scalar_subquery(gold, g_tree))

        base_corrupts = base_generate_corruptions(gold, schema, ordered, rng)
        base_prio = {
            "literal_mismatch": 0, "except_to_neq": 0, "in_to_dup_join": 0,
            "sqlite_all_syntax": 0, "missing_subq_agg": 0, "drop_not_in": 0,
            "intersect_to_union": 0, "bogus_join_key": 1, "drop_bridge_join": 1,
            "extra_join": 1, "having_in_where": 1, "swap_agg": 1,
            "swap_table": 2, "swap_column": 2, "drop_group_by": 2,
            "drop_order_by": 2, "flip_order_dir": 2, "spurious_where": 2,
            "drop_distinct": 2, "spurious_distinct": 2, "flip_op": 3,
            "proj_swap": 4, "proj_drop": 4, "proj_extra_col": 5, "proj_extra_agg": 5,
        }
        base_corrupts.sort(key=lambda kv: (base_prio.get(kv[0], 4), rng.random()))
        all_corrupts = special_corrupts + base_corrupts

        # Deduplicate while preserving priority of special_corrupts first!
        seen_c = {gold.strip(), gold.strip().rstrip(";")}
        dedup_corrupts: list[tuple[str, str]] = []
        for k, s in all_corrupts:
            sc = s.strip()
            if sc and sc not in seen_c and sc.rstrip(";") not in seen_c:
                seen_c.add(sc)
                dedup_corrupts.append((k, sc))

        special_kinds = {
            "literal_wrong_col_sibling", "literal_wrong_col_joined",
            "notin_to_join_neq", "notin_to_join_notlike", "notin_to_neq_join",
            "except_to_join_neq", "except_to_join_notlike",
            "subq_agg_to_having", "subq_agg_flip_op", "literal_mismatch",
        }

        fail_added = 0
        fp_added = 0
        retry_cand = None

        for kind, bad_sql in dedup_corrupts:
            c_pub, c_3, c_fb = eval_sql(row, bad_sql)
            if c_3:
                continue
            is_spec = kind in special_kinds
            c_probe = run_db_probe(pub_db, schema, question, [draft, bad_sql], ex_cols)
            c_warns = get_audit(bad_sql, cat, question, ordered, ex_cols, c_fb, c_probe)["warnings"] or None

            # Use a non-passing draft in `issue_sql` when placing `bad_sql` inside `Candidate Hints`!
            base_draft_sql = draft if not d_3 else bad_sql
            base_draft_fb = d_fb if not d_3 else c_fb
            base_draft_w = d_warns if not d_3 else c_warns

            if c_pub and not c_3:
                max_fp = 3 if is_spec else 1
                if fp_added < max_fp:
                    # Place FP inside `### Candidate Hints` with votes=3..4 (85% of time) to match inference!
                    if rng.random() < 0.85:
                        votes_sim = rng.choice([2, 3, 4, 4])
                        h_fp = [{"sql": bad_sql, "feedback": c_fb, "warnings": c_warns, "votes": votes_sim}]
                        records.append(make_rec(
                            tid, db_id, "audit_fp_synth", question, schema, ordered, example,
                            base_draft_sql, base_draft_fb, base_draft_w, h_fp, c_probe, "fp_audit", gold,
                            {"kind": kind, "rc": rc, "in_hints": True},
                        ))
                    else:
                        records.append(make_rec(
                            tid, db_id, "audit_fp_synth", question, schema, ordered, example,
                            bad_sql, c_fb, c_warns, None, c_probe, "fp_audit", gold,
                            {"kind": kind, "rc": rc, "in_hints": False},
                        ))
                    regime_counts["audit_fp_synth"] += 1
                    corrupt_counts[f"fp:{kind}"] += 1
                    fp_added += 1
            else:
                max_fail = 3 if is_spec else 1
                if fail_added < max_fail:
                    # Put bad_sql as checked Hint 1 (70% of time) or Draft (30%) so model repairs bad Hints too!
                    if not d_pub and rng.random() < 0.70 and bad_sql != draft:
                        h_bad = [{"sql": bad_sql, "feedback": c_fb, "warnings": c_warns, "votes": rng.choice([1, 2, 3, 4])}]
                        records.append(make_rec(
                            tid, db_id, "sem_corrupt_repair", question, schema, ordered, example,
                            draft, d_fb, d_warns, h_bad, c_probe, "repair", gold, {"kind": kind, "rc": rc},
                        ))
                    else:
                        records.append(make_rec(
                            tid, db_id, "sem_corrupt_repair", question, schema, ordered, example,
                            bad_sql, c_fb, c_warns, None, c_probe, "repair", gold, {"kind": kind, "rc": rc},
                        ))
                    regime_counts["sem_corrupt_repair"] += 1
                    corrupt_counts[f"fail:{kind}"] += 1
                    fail_added += 1
                    if retry_cand is None and bad_sql != draft:
                        retry_cand = (kind, bad_sql, c_fb, c_warns, c_probe)

        # 6. Multi-Turn Retry (never repeating previous broken SQL!)
        if not d_pub and retry_cand is not None:
            k2, s2, fb2, w2, pr2 = retry_cand
            h_list = [hint_ent(h, v, hfb, checked=(i == 0), prep=pr2 or probe_rep) for i, (h, v, _, _, hfb) in enumerate(pub_failing_hints[:2])]
            records.append(make_rec(
                tid, db_id, "multi_turn_retry", question, schema, ordered, example,
                draft, d_fb, d_warns, h_list or None, pr2 or probe_rep, "retry", gold,
                previous_attempt={"sql": s2, "feedback": fb2, "warnings": w2},
                extra_meta={"kind2": k2, "rc": rc},
            ))
            regime_counts["multi_turn_retry"] += 1

    pool.close()

    # Calibrate TP-keep to ~28% of total `fp_audit` (~130 clean TPs vs ~340 FPs -> 72% FP-rewrite prior!)
    fp_total = regime_counts["audit_fp_real"] + regime_counts["audit_fp_real_v4"] + regime_counts["audit_fp_synth"]
    tp_target = min(135, max(90, int(fp_total * 0.38)))
    by_p: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for p, r in clean_tp_pool:
        by_p[p].append(r)
    for p in by_p:
        rng.shuffle(by_p[p])
    chosen_tp = by_p[0][: int(tp_target * 0.50)] + by_p[1][: int(tp_target * 0.40)] + by_p[2][: int(tp_target * 0.10)]
    for r in chosen_tp:
        records.append(r)
        regime_counts["audit_tp_keep"] += 1

    out_ids = {r["id"] for r in records}
    out_dbs = {r["db_id"] for r in records}
    assert out_ids.issubset(train_ids) and out_dbs.issubset(train_dbs), "LEAKAGE!"

    rng.shuffle(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Audit warning & Candidate-Hints balance right here!
    warn_c = Counter()
    cand_hint_fp = Counter()
    for r in records:
        u = r["messages"][1]["content"]
        for tag in ("LITERAL_WRONG_COLUMN", "LITERAL_MISMATCH", "UNGROUNDED_LITERAL", "NON_FK_JOIN", "NEGATION_NEQ_ON_JOIN", "ORDERED_REQUIRED"):
            if tag in u:
                warn_c[tag] += 1
        if r["mode"] == "fp_audit" and "### Candidate Hints" in u and "MATCHES_EXAMPLE" in u.split("### Candidate Hints")[1]:
            cand_hint_fp["tp_keep" if r["regime"] == "audit_tp_keep" else "fp_rewrite"] += 1

    prov = {
        "prompt_contract": "v2-rich-chat (v3 surgical balance)",
        "split_used": "train_only",
        "unique_task_ids": len(out_ids),
        "unique_db_ids": len(out_dbs),
        "total_examples": len(records),
        "fp_audit_cand_hints_balance": dict(cand_hint_fp),
        "warning_tag_counts": dict(warn_c),
        "regime_counts": dict(regime_counts),
        "top_corrupt_counts": dict(corrupt_counts.most_common(25)),
        "source_files_sha256": {
            "train.jsonl": sha256_file(train_path),
            "train_references.json": sha256_file(refs_path),
            "repair_hints.json": sha256_file(hints_path),
            "public_results.json": sha256_file(pub_path),
            "manifest.json": sha256_file(manifest_path),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (args.out.parent / "data_provenance_v3.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(prov, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
