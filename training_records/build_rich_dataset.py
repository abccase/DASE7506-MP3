"""Build the rich, strictly train-only multi-regime dataset aligned 100% byte-for-byte
with `submission/{prompt_format.py,db_probe.py,static_audit.py}` (`v2-rich-chat`).

HARD COMPLIANCE & PROVENANCE GUARANTEE:
  - Reads ONLY `train.jsonl`, `train_references.json`, and train-ID/DB-filtered entries from
    `repair_hints.json`, `public_results.json`, `manifest.json`, plus the 118 `train` DBs.
  - NEVER touches `validation` or `test` labels, tasks, or databases.
  - Asserts every output task ID starts with `train-` and records exact SHA-256 provenance.

Regimes generated:
  1. `orig_hard_repair` (`mode="repair"`): Draft + failing-only hints (zero TP-hint leakage!)
  2. `orig_allfail_fullctx` (`mode="repair"`): Full context on the 83 hardest all-pub-fail tasks
  3. `hint_repair` (`mode="repair"`): Failing 8B hints with full execution + static/probe feedback
  4. `multi_hint_synth` (`mode="repair"`): Multi-hint synthesis (capped TP-hint presence to prevent copying)
  5. `audit_fp_real` / `_single` / `_ctx` (`mode="fp_audit"`): ALL 105 real 3-instance FPs -> gold_sql
  6. `audit_fp_synth` (`mode="fp_audit"`): Synthetic 3-instance FPs (corrupted gold SQLs passing public DB, failing 3-inst) -> gold_sql
  7. `audit_tp_keep` (`mode="fp_audit"`): Balanced True Positives (weighted to 0/1-row) -> keep passing SQL
  8. `sem_corrupt_repair` (`mode="repair"`): 12 targeted AST semantic corruptions executed on train public DB -> gold_sql
  9. `multi_turn_retry` (`mode="retry"`): 2-turn self-correction with `previous_attempt` -> gold_sql
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import random
import re
import sqlite3
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
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HARNESS_DIR))

from mp3_sql.common import result_fingerprint  # noqa: E402
from db_probe import build_probe_query, parse_probe_result  # noqa: E402
from prompt_format import build_messages  # noqa: E402
from static_audit import SchemaCatalog, audit_sql  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_semicolon(sql: str) -> str:
    s = sql.strip()
    if s and not s.endswith(";"):
        s += ";"
    return s


class FastSQLitePool:
    def __init__(self) -> None:
        self._conns: dict[str, sqlite3.Connection] = {}

    def conn(self, db_path: Path) -> sqlite3.Connection:
        key = str(db_path)
        con = self._conns.get(key)
        if con is None:
            con = sqlite3.connect(db_path.as_uri() + "?mode=ro&immutable=1", uri=True)
            con.execute("PRAGMA query_only=ON")
            self._conns[key] = con
        return con

    def execute(self, db_path: Path, sql: str) -> dict[str, Any]:
        con = self.conn(db_path)
        deadline = time.monotonic() + 0.6
        steps = [0]

        def progress() -> int:
            steps[0] += 1000
            return 1 if (steps[0] > 4_000_000 or time.monotonic() > deadline) else 0

        con.set_progress_handler(progress, 1000)
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(5001)
            if len(rows) > 5000:
                return {
                    "ok": False, "columns": [], "rows": [], "row_count": 0, "truncated": False,
                    "error": {"type": "ValueError", "message": "Result exceeds 5000 rows"},
                }
            cols = [d[0] for d in cur.description] if cur.description else []
            out_rows = [
                [("blob:" + v.hex()) if isinstance(v, bytes) else v for v in r]
                for r in rows
            ]
            return {
                "ok": True, "columns": cols, "rows": out_rows,
                "row_count": len(out_rows), "truncated": len(out_rows) > 8, "error": None,
            }
        except Exception as err:
            return {
                "ok": False, "columns": [], "rows": [], "row_count": 0, "truncated": False,
                "error": {"type": type(err).__name__, "message": str(err)[:220]},
            }
        finally:
            con.set_progress_handler(None, 0)

    def close(self) -> None:
        for con in self._conns.values():
            try:
                con.close()
            except Exception:
                pass
        self._conns.clear()


def clip_feedback(fb: dict[str, Any]) -> dict[str, Any]:
    """Match exact mp3_sql environment observation format (rows[:8])."""
    return {
        "ok": fb["ok"],
        "columns": list(fb.get("columns") or []),
        "rows": (fb.get("rows") or [])[:8],
        "row_count": fb.get("row_count", len(fb.get("rows") or [])),
        "truncated": fb.get("row_count", 0) > 8,
        "error": fb.get("error"),
        "matches_example": bool(fb.get("matches_example", False)),
    }


# ---------- 12 High-Signal Semantic Corruptors ----------

def _corrupt_string_literal(tree: exp.Expression, rng: random.Random) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    lits = [n for n in tree.find_all(exp.Literal) if n.is_string and str(n.this).strip()]
    if not lits:
        return out
    idx = rng.randrange(len(lits))
    val = str(lits[idx].this)
    variants: list[str] = []
    if val != val.lower():
        variants.append(val.lower())
    elif val != val.title():
        variants.append(val.title())
    words = val.split()
    if len(words) >= 2:
        variants.append(words[-1])
        variants.append(words[0])
    if ":" in val:
        variants.append(val.split(":")[0])
    if val != val.strip():
        variants.append(val.strip())
    if len(val) >= 4 and val[0].isalpha():
        variants.append(val[0])
    for new_val in variants[:2]:
        if new_val and new_val != val:
            v = copy.deepcopy(tree)
            v_lits = [n for n in v.find_all(exp.Literal) if n.is_string and str(n.this).strip()]
            v_lits[idx].set("this", new_val)
            out.append(("literal_mismatch", v.sql(dialect="sqlite")))
    return out


def _corrupt_projection(tree: exp.Expression, schema: list[dict[str, Any]], rng: random.Random) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(tree, exp.Select):
        return out
    exprs = list(tree.expressions)
    if not exprs:
        return out
    if len(exprs) >= 2:
        v = copy.deepcopy(tree)
        ve = list(v.expressions)
        ve[0], ve[1] = ve[1], ve[0]
        v.set("expressions", ve)
        out.append(("proj_swap", v.sql(dialect="sqlite")))
        v2 = copy.deepcopy(tree)
        ve2 = list(v2.expressions)
        ve2.pop(1 if len(ve2) == 2 else rng.randrange(len(ve2)))
        v2.set("expressions", ve2)
        out.append(("proj_drop", v2.sql(dialect="sqlite")))
    v3 = copy.deepcopy(tree)
    if v3.args.get("group"):
        v3.append("expressions", exp.Count(this=exp.Star()))
        out.append(("proj_extra_agg", v3.sql(dialect="sqlite")))
    else:
        from_tbls = [t.name.lower() for t in v3.find_all(exp.Table) if t.name]
        used_cols = {c.name.lower() for c in exprs if isinstance(c, exp.Column)}
        for tb in schema:
            if tb["name"].lower() in from_tbls:
                extra = [c["name"] for c in tb["columns"] if c["name"].lower() not in used_cols]
                if extra:
                    v3.append("expressions", exp.column(rng.choice(extra)))
                    out.append(("proj_extra_col", v3.sql(dialect="sqlite")))
                    break
    return out


def _corrupt_distinct(tree: exp.Expression) -> list[tuple[str, str]]:
    if not isinstance(tree, exp.Select):
        return []
    v = copy.deepcopy(tree)
    if v.args.get("distinct"):
        v.set("distinct", None)
        return [("drop_distinct", v.sql(dialect="sqlite"))]
    if not any(isinstance(e, exp.AggFunc) for e in v.expressions):
        v.set("distinct", exp.Distinct())
        return [("spurious_distinct", v.sql(dialect="sqlite"))]
    return []


def _corrupt_join(tree: exp.Expression, schema: list[dict[str, Any]], rng: random.Random) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(tree, exp.Select):
        return out
    joins = list(tree.args.get("joins") or [])
    tbl_map = {t["name"].lower(): t for t in schema}
    alias_to_tbl: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        if t.name:
            alias_to_tbl[(t.alias or t.name).lower()] = t.name.lower()

    if joins:
        v = copy.deepcopy(tree)
        v_joins = list(v.args.get("joins") or [])
        on_expr = v_joins[0].args.get("on")
        if isinstance(on_expr, exp.EQ):
            cols = [c for c in [on_expr.left, on_expr.right] if isinstance(c, exp.Column)]
            if cols:
                target_col = cols[-1]
                t_key = alias_to_tbl.get((target_col.table or "").lower(), "")
                tb = tbl_map.get(t_key)
                if tb:
                    alts = [c["name"] for c in tb["columns"] if c["name"].lower() != target_col.name.lower()]
                    if alts:
                        id_alts = [c for c in alts if "id" in c.lower()]
                        pick = rng.choice(id_alts or alts)
                        target_col.set("this", exp.to_identifier(pick))
                        out.append(("bogus_join_key", v.sql(dialect="sqlite")))

    used_tbls = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    for tb in schema:
        tn = tb["name"].lower()
        if tn in used_tbls:
            continue
        for fk in tb.get("foreign_keys") or []:
            ref = fk["table"].lower()
            if ref in used_tbls:
                ref_alias = next((a for a, real in alias_to_tbl.items() if real == ref), fk["table"])
                v = copy.deepcopy(tree)
                v.join(
                    exp.table_(tb["name"]),
                    on=exp.EQ(
                        this=exp.column(fk["from"], table=tb["name"]),
                        expression=exp.column(fk["to"], table=ref_alias),
                    ),
                    copy=False,
                )
                out.append(("extra_join", v.sql(dialect="sqlite")))
                break
        if any(k == "extra_join" for k, _ in out):
            break

    if len(joins) >= 2:
        v = copy.deepcopy(tree)
        v_joins = list(v.args.get("joins") or [])
        v_joins.pop(0)
        v.set("joins", v_joins)
        out.append(("drop_bridge_join", v.sql(dialect="sqlite")))

    return out


def _corrupt_setop_and_subquery(gold: str, tree: exp.Expression) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    m = re.search(r"([<>]=?)\s*\(\s*SELECT\s+(?:MAX|MIN)\s*\(\s*([^)]+?)\s*\)", gold, flags=re.I)
    if m:
        op, inner = m.group(1), m.group(2)
        out.append(("sqlite_all_syntax", gold[: m.start()] + f"{op} ALL (SELECT {inner}" + gold[m.end():]))
        out.append(("missing_subq_agg", gold[: m.start()] + f"{op} (SELECT {inner}" + gold[m.end():]))

    if re.search(r"\bNOT\s+IN\b", gold, flags=re.I):
        out.append(("drop_not_in", re.sub(r"\bNOT\s+IN\b", "IN", gold, count=1, flags=re.I)))

    if isinstance(tree, exp.Except):
        right = tree.expression
        if isinstance(right, exp.Select) and right.args.get("where"):
            v = copy.deepcopy(right)
            for eq in v.args["where"].find_all(exp.EQ):
                eq.replace(exp.NEQ(this=eq.left.copy(), expression=eq.right.copy()))
                break
            out.append(("except_to_neq", v.sql(dialect="sqlite")))
    elif isinstance(tree, exp.Intersect):
        v = exp.Union(this=copy.deepcopy(tree.this), expression=copy.deepcopy(tree.expression), distinct=True)
        out.append(("intersect_to_union", v.sql(dialect="sqlite")))

    m_in = re.search(
        r"^(SELECT\s+(?:AVG|SUM|COUNT)\s*\([^)]+\)\s+FROM\s+(\w+))\s+WHERE\s+(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)\s*\)\s*;?$",
        gold.strip(),
        flags=re.I,
    )
    if m_in:
        head, t1, c1, c2, t2 = m_in.groups()
        out.append(("in_to_dup_join", f"{head} JOIN {t2} ON {t1}.{c1} = {t2}.{c2}"))

    return out


def _corrupt_order_limit(tree: exp.Expression, ordered: bool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(tree, exp.Select):
        return out
    order = tree.args.get("order")
    if order and order.expressions:
        v = copy.deepcopy(tree)
        first_ord = v.args["order"].expressions[0]
        first_ord.set("desc", not bool(first_ord.args.get("desc")))
        out.append(("flip_order_dir", v.sql(dialect="sqlite")))
        if ordered:
            v2 = copy.deepcopy(tree)
            v2.set("order", None)
            out.append(("drop_order_by", v2.sql(dialect="sqlite")))
    if tree.args.get("limit"):
        v3 = copy.deepcopy(tree)
        v3.set("limit", None)
        out.append(("drop_limit", v3.sql(dialect="sqlite")))
    return out


def _corrupt_group_having_agg(tree: exp.Expression) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(tree, exp.Select):
        return out
    if tree.args.get("group"):
        v = copy.deepcopy(tree)
        v.set("group", None)
        out.append(("drop_group_by", v.sql(dialect="sqlite")))
    if tree.args.get("having"):
        v2 = copy.deepcopy(tree)
        hav = v2.args.get("having")
        v2.set("having", None)
        if v2.args.get("where"):
            v2.where(hav.this.copy(), copy=False)
        else:
            v2.set("where", exp.Where(this=hav.this.copy()))
        out.append(("having_in_where", v2.sql(dialect="sqlite")))
    swap_agg = {exp.Max: exp.Min, exp.Min: exp.Max, exp.Avg: exp.Sum, exp.Sum: exp.Avg}
    for node in tree.find_all(exp.Max, exp.Min, exp.Avg, exp.Sum):
        v3 = copy.deepcopy(tree)
        for m in v3.find_all(type(node)):
            m.replace(swap_agg[type(node)](this=m.this.copy()))
            break
        out.append(("swap_agg", v3.sql(dialect="sqlite")))
        break
    return out


def _corrupt_col_table_op(
    gold: str, tree: exp.Expression, schema: list[dict[str, Any]], rng: random.Random
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(tree, exp.Select):
        return out
    table_cols = {tb["name"].lower(): [c["name"] for c in tb["columns"]] for tb in schema}
    all_cols = [c["name"] for tb in schema for c in tb["columns"]]
    alias_to_tbl = {(t.alias or t.name).lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.name}

    cols = list(tree.find_all(exp.Column))
    rng.shuffle(cols)
    for node in cols[:4]:
        t_real = alias_to_tbl.get((node.table or "").lower(), "")
        pool = table_cols.get(t_real, all_cols)
        cands = [c for c in pool if c.lower() != node.name.lower()]
        near = difflib.get_close_matches(node.name, cands, n=1, cutoff=0.35)
        if near:
            v = copy.deepcopy(tree)
            for c in v.find_all(exp.Column):
                if c.name == node.name and (c.table or "") == (node.table or ""):
                    c.set("this", exp.to_identifier(near[0]))
                    out.append(("swap_column", v.sql(dialect="sqlite")))
                    break
            break

    table_names = [tb["name"] for tb in schema]
    for node in tree.find_all(exp.Table):
        cands = [t for t in table_names if t.lower() != node.name.lower()]
        near = difflib.get_close_matches(node.name, cands, n=1, cutoff=0.35)
        if near:
            v = copy.deepcopy(tree)
            for t in v.find_all(exp.Table):
                if t.name == node.name:
                    t.set("this", exp.to_identifier(near[0]))
                    out.append(("swap_table", v.sql(dialect="sqlite")))
                    break
            break

    if list(tree.find_all(exp.And)):
        v = copy.deepcopy(tree)
        for node in v.find_all(exp.And):
            node.replace(exp.Or(this=node.left.copy(), expression=node.right.copy()))
            break
        out.append(("and_to_or", v.sql(dialect="sqlite")))
    elif list(tree.find_all(exp.Or)):
        v = copy.deepcopy(tree)
        for node in v.find_all(exp.Or):
            node.replace(exp.And(this=node.left.copy(), expression=node.right.copy()))
            break
        out.append(("or_to_and", v.sql(dialect="sqlite")))
    else:
        swap_op = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.GT, exp.LTE: exp.LT, exp.EQ: exp.NEQ, exp.NEQ: exp.EQ}
        where = tree.args.get("where")
        if where:
            for node in where.find_all(exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ):
                v = copy.deepcopy(tree)
                for m in v.args["where"].find_all(type(node)):
                    m.replace(swap_op[type(node)](this=m.left.copy(), expression=m.right.copy()))
                    break
                out.append(("flip_op", v.sql(dialect="sqlite")))
                break

    from_tbls = [t for t in tree.find_all(exp.Table) if t.name]
    if from_tbls and not tree.args.get("where"):
        t0 = from_tbls[0]
        tb = next((x for x in schema if x["name"].lower() == t0.name.lower()), None)
        if tb:
            txt_cols = [c["name"] for c in tb["columns"] if "text" in c.get("type", "").lower() or "char" in c.get("type", "").lower()]
            if txt_cols:
                v = copy.deepcopy(tree)
                col_pick = rng.choice(txt_cols)
                alias = t0.alias or t0.name
                v.where(exp.EQ(this=exp.column(col_pick, table=alias), expression=exp.Literal.string("Active")), copy=False)
                out.append(("spurious_where", v.sql(dialect="sqlite")))

    return out


def generate_corruptions(
    gold: str,
    schema: list[dict[str, Any]],
    ordered: bool,
    rng: random.Random,
) -> list[tuple[str, str]]:
    try:
        tree = sqlglot.parse_one(gold, read="sqlite")
    except Exception:
        return []
    raw: list[tuple[str, str]] = []
    raw.extend(_corrupt_string_literal(tree, rng))
    raw.extend(_corrupt_projection(tree, schema, rng))
    raw.extend(_corrupt_distinct(tree))
    raw.extend(_corrupt_join(tree, schema, rng))
    raw.extend(_corrupt_setop_and_subquery(gold, tree))
    raw.extend(_corrupt_order_limit(tree, ordered))
    raw.extend(_corrupt_group_having_agg(tree))
    raw.extend(_corrupt_col_table_op(gold, tree, schema, rng))

    seen = {gold.strip(), gold.strip().rstrip(";")}
    deduped: list[tuple[str, str]] = []
    for kind, sql in raw:
        s = sql.strip()
        if s and s not in seen and s.rstrip(";") not in seen:
            seen.add(s)
            deduped.append((kind, s))
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"E:\MP3\exp\lora\data\train_rich_v2.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-corrupt-fail-per-task", type=int, default=4)
    parser.add_argument("--max-corrupt-fp-per-task", type=int, default=2)
    parser.add_argument("--max-tp-keep", type=int, default=360)
    parser.add_argument("--max-multi-hint-with-tp", type=int, default=120)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = PACKAGE / "mp3_sql" / "data"

    # Load STRICTLY train-only files and filter shared manifests strictly to train IDs / DBs
    train_path = data_dir / "train.jsonl"
    refs_path = data_dir / "train_references.json"
    hints_path = data_dir / "repair_hints.json"
    pub_path = data_dir / "public_results.json"
    manifest_path = data_dir / "manifest.json"

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_ids = {r["id"] for r in train_rows}
    train_dbs = {r["db_id"] for r in train_rows}
    assert all(tid.startswith("train-") for tid in train_ids), "Non-train ID detected!"
    assert len(train_rows) == 480, f"Expected 480 train tasks, got {len(train_rows)}"

    train_refs = json.loads(refs_path.read_text(encoding="utf-8"))
    gold_map: dict[str, str] = {k: v for k, v in train_refs["gold_sql"].items() if k in train_ids}
    ref_cases: dict[str, list[dict[str, Any]]] = {k: v for k, v in train_refs["tasks"].items() if k in train_ids}
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

    def get_warnings(
        sql: str,
        cat: SchemaCatalog,
        question: str,
        ordered: bool,
        ex_cols: list[str],
        fb: dict | None,
        probe_rep: dict | None,
        drop_prob: float = 0.12,
    ) -> list[str] | None:
        if rng.random() < drop_prob:
            return None
        actual_cols = (fb.get("columns") if (fb and fb.get("ok")) else None)
        res = audit_sql(sql, cat, question, ordered, ex_cols, actual_cols=actual_cols, probe_report=probe_rep)
        return res["warnings"] or None

    records: list[dict[str, Any]] = []
    regime_counts: Counter[str] = Counter()
    corrupt_kind_counts: Counter[str] = Counter()
    tp_keep_pool: list[tuple[int, dict[str, Any]]] = []
    multi_hint_tp_pool: list[dict[str, Any]] = []

    def make_rec(
        tid: str,
        db_id: str,
        regime: str,
        question: str,
        schema: list[dict],
        ordered: bool,
        example: dict,
        issue_sql: str,
        draft_feedback: dict | None,
        draft_warnings: list[str] | None,
        hints_list: list[dict] | None,
        probe_report: dict | None,
        mode: str,
        target: str,
        previous_attempt: dict | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 12% dropout of probe_report so LoRA remains strong even when probe wasn't run
        use_probe = probe_report if (probe_report and rng.random() > 0.12) else None
        msgs = build_messages(
            question=question,
            schema=schema,
            ordered=ordered,
            example=example,
            issue_sql=issue_sql,
            draft_feedback=draft_feedback,
            draft_warnings=draft_warnings,
            hints=hints_list,
            probe_report=use_probe,
            mode=mode,
            previous_attempt=previous_attempt,
        )
        return {
            "id": tid,
            "db_id": db_id,
            "regime": regime,
            "mode": mode,
            "messages": msgs,
            "target": ensure_semicolon(target),
            "meta": extra_meta or {},
        }

    t0 = time.perf_counter()
    for row in train_rows:
        tid = row["id"]
        db_id = row["db_id"]
        gold = gold_map[tid].strip()
        meta = db_meta_map[db_id]
        schema = meta["schema"]
        cat = SchemaCatalog(schema)
        pub_db_path = data_dir / meta["public_path"]
        question = row["question"]
        example = row["example"]
        ex_cols = list(example.get("columns") or [])
        ordered = row["ordered"]
        rc = example.get("row_count", 0)
        draft = row["issue_sql"].strip()

        d_pub, d_3, d_fb = eval_sql(row, draft)

        # Deduplicate hints while preserving vote counts (just like harness!)
        raw_sug = [h.strip() for h in hints_map.get(tid, {}).get("suggestions", []) if h.strip()]
        vote_counts = Counter(raw_sug)
        uniq_hints = [h for h in dict.fromkeys(raw_sug) if h != draft][:4]

        h_evals = []
        for h in uniq_hints:
            hp, h3, hfb = eval_sql(row, h)
            h_evals.append((h, vote_counts[h], hp, h3, hfb))

        # Run exact harness db_probe on [draft] + hints
        probe_rep = run_db_probe(pub_db_path, schema, question, [draft] + uniq_hints, ex_cols)
        d_warns = get_warnings(draft, cat, question, ordered, ex_cols, d_fb, probe_rep)

        def hint_entry(h: str, votes: int, hfb: dict, checked: bool, prep: dict | None) -> dict:
            fb_use = hfb if checked else None
            w = get_warnings(h, cat, question, ordered, ex_cols, fb_use, prep)
            return {"sql": h, "feedback": fb_use, "warnings": w, "votes": votes}

        failing_hints = [x for x in h_evals if not x[3]]        # fail all-3
        pub_failing_hints = [x for x in h_evals if not x[2]]    # fail public

        # ---- Regime 1: Authentic Draft + Failing-Only Hints (NO TP-hint shortcut) ----
        if not d_pub:
            h_list = [
                hint_entry(h, v, hfb, checked=(i == 0 or rng.random() < 0.5), prep=probe_rep)
                for i, (h, v, hp, h3, hfb) in enumerate(pub_failing_hints[:2])
            ]
            records.append(make_rec(
                tid, db_id, "orig_hard_repair", question, schema, ordered, example,
                draft, d_fb, d_warns, h_list or None, probe_rep, "repair", gold, extra_meta={"rc": rc},
            ))
            regime_counts["orig_hard_repair"] += 1

            # Extra full-context view for the 83 hardest tasks (where ALL of draft+hints fail pub)
            if not any(hp for _, _, hp, _, _ in h_evals) and pub_failing_hints:
                full_h = [
                    hint_entry(h, v, hfb, checked=(i < 3), prep=probe_rep)
                    for i, (h, v, hp, h3, hfb) in enumerate(pub_failing_hints[:4])
                ]
                records.append(make_rec(
                    tid, db_id, "orig_allfail_fullctx", question, schema, ordered, example,
                    draft, d_fb, d_warns, full_h, probe_rep, "repair", gold, extra_meta={"rc": rc},
                ))
                regime_counts["orig_allfail_fullctx"] += 1

        # ---- Regime 2: Failing 8B Hint Repair ----
        for h, v, hp, h3, hfb in pub_failing_hints[:2]:
            h_list = [hint_entry(h, v, hfb, checked=True, prep=probe_rep)]
            records.append(make_rec(
                tid, db_id, "hint_repair", question, schema, ordered, example,
                draft, d_fb, d_warns, h_list, probe_rep, "repair", gold, extra_meta={"rc": rc},
            ))
            regime_counts["hint_repair"] += 1

        # ---- Regime 3: Multi-Hint Synthesis (capped TP-hint presence) ----
        if len(h_evals) >= 2:
            has_tp_hint = any(h3 for _, _, _, h3, _ in h_evals)
            h_list = [
                hint_entry(h, v, hfb, checked=(not h3 and i < 2), prep=probe_rep)
                for i, (h, v, hp, h3, hfb) in enumerate(h_evals[:4])
            ]
            rec = make_rec(
                tid, db_id, "multi_hint_synth", question, schema, ordered, example,
                draft, d_fb, d_warns, h_list, probe_rep, "repair", gold,
                extra_meta={"rc": rc, "has_tp": has_tp_hint},
            )
            if has_tp_hint:
                multi_hint_tp_pool.append(rec)
            else:
                records.append(rec)
                regime_counts["multi_hint_synth"] += 1

        # ---- Regime 4: Real 3-Instance False Positives (CRITICAL!) ----
        if d_pub and not d_3:
            # View A: Draft alone passing public DB
            records.append(make_rec(
                tid, db_id, "audit_fp_real", question, schema, ordered, example,
                draft, d_fb, d_warns, None, probe_rep, "fp_audit", gold,
                extra_meta={"src": "draft", "rc": rc},
            ))
            regime_counts["audit_fp_real"] += 1
            # View B: Draft + non-TP hints
            if failing_hints:
                h_list = [
                    hint_entry(h, v, hfb, checked=(i == 0), prep=probe_rep)
                    for i, (h, v, hp, h3, hfb) in enumerate(failing_hints[:2])
                ]
                records.append(make_rec(
                    tid, db_id, "audit_fp_real_ctx", question, schema, ordered, example,
                    draft, d_fb, d_warns, h_list, probe_rep, "fp_audit", gold,
                    extra_meta={"src": "draft+hints", "rc": rc},
                ))
                regime_counts["audit_fp_real_ctx"] += 1

        for h, v, hp, h3, hfb in h_evals:
            if hp and not h3:
                h_ent = hint_entry(h, v, hfb, checked=True, prep=probe_rep)
                records.append(make_rec(
                    tid, db_id, "audit_fp_real", question, schema, ordered, example,
                    draft, d_fb, d_warns, [h_ent], probe_rep, "fp_audit", gold,
                    extra_meta={"src": "hint", "rc": rc},
                ))
                regime_counts["audit_fp_real"] += 1
                # View B: FP hint as primary issue_sql (teaches auditing even if only 1 SQL shown)
                hw = get_warnings(h, cat, question, ordered, ex_cols, hfb, probe_rep)
                records.append(make_rec(
                    tid, db_id, "audit_fp_real_single", question, schema, ordered, example,
                    h, hfb, hw, None, probe_rep, "fp_audit", gold,
                    extra_meta={"src": "hint_single", "rc": rc},
                ))
                regime_counts["audit_fp_real_single"] += 1

        # ---- Regime 6 (collect pool): Balanced True-Positive Keep Examples ----
        prio = 0 if rc == 0 else (1 if rc == 1 else 2)
        if d_pub and d_3:
            tp_keep_pool.append((prio, make_rec(
                tid, db_id, "audit_tp_keep", question, schema, ordered, example,
                draft, d_fb, d_warns, None, probe_rep, "fp_audit", draft,
                extra_meta={"src": "draft", "rc": rc},
            )))
        tp_hints = [(h, v, hfb) for (h, v, hp, h3, hfb) in h_evals if hp and h3]
        if tp_hints:
            h_tp, v_tp, hfb_tp = tp_hints[0]
            h_ent = hint_entry(h_tp, v_tp, hfb_tp, checked=True, prep=probe_rep)
            tp_keep_pool.append((prio, make_rec(
                tid, db_id, "audit_tp_keep", question, schema, ordered, example,
                draft, d_fb, d_warns, [h_ent], probe_rep, "fp_audit", h_tp,
                extra_meta={"src": "hint", "rc": rc},
            )))

        # ---- Regime 5 & 7 & 8: Targeted AST Semantic Corruptions ----
        corrupts = generate_corruptions(gold, schema, ordered, rng)
        rng.shuffle(corrupts)
        # Prioritize hard semantic/join/literal/setop/aggregation bugs that survive deterministic projection variants!
        priority_kinds = {
            "literal_mismatch": 0, "except_to_neq": 0, "in_to_dup_join": 0,
            "sqlite_all_syntax": 0, "missing_subq_agg": 0, "drop_not_in": 0,
            "intersect_to_union": 0, "bogus_join_key": 1, "drop_bridge_join": 1,
            "extra_join": 1, "having_in_where": 1, "swap_agg": 1,
            "drop_group_by": 2, "drop_order_by": 2, "flip_order_dir": 2,
            "swap_column": 2, "swap_table": 2, "spurious_where": 2,
            "drop_distinct": 2, "spurious_distinct": 2, "flip_op": 3,
            "and_to_or": 3, "or_to_and": 3,
            "proj_swap": 4, "proj_drop": 4, "proj_extra_col": 5, "proj_extra_agg": 5,
        }
        corrupts.sort(key=lambda kv: (priority_kinds.get(kv[0], 4), rng.random()))

        fail_added = 0
        fp_added = 0
        failing_corrupts: list[tuple[str, str, dict, list[str] | None, dict | None]] = []

        for kind, bad_sql in corrupts:
            c_pub, c_3, c_fb = eval_sql(row, bad_sql)
            if c_3:
                continue
            c_probe = run_db_probe(pub_db_path, schema, question, [bad_sql], ex_cols)
            c_warns = get_warnings(bad_sql, cat, question, ordered, ex_cols, c_fb, c_probe)

            if c_pub and not c_3:
                if fp_added < args.max_corrupt_fp_per_task:
                    records.append(make_rec(
                        tid, db_id, "audit_fp_synth", question, schema, ordered, example,
                        bad_sql, c_fb, c_warns, None, c_probe, "fp_audit", gold,
                        extra_meta={"kind": kind, "rc": rc},
                    ))
                    regime_counts["audit_fp_synth"] += 1
                    corrupt_kind_counts[f"fp:{kind}"] += 1
                    fp_added += 1
            else:
                if fail_added < args.max_corrupt_fail_per_task:
                    records.append(make_rec(
                        tid, db_id, "sem_corrupt_repair", question, schema, ordered, example,
                        bad_sql, c_fb, c_warns, None, c_probe, "repair", gold,
                        extra_meta={"kind": kind, "rc": rc},
                    ))
                    regime_counts["sem_corrupt_repair"] += 1
                    corrupt_kind_counts[f"fail:{kind}"] += 1
                    fail_added += 1
                    failing_corrupts.append((kind, bad_sql, c_fb, c_warns, c_probe))

        # ---- Regime 9: Multi-Turn Retry (`mode="retry"` with `previous_attempt`) ----
        if not d_pub and failing_corrupts:
            kind2, sql2, fb2, w2, pr2 = failing_corrupts[0]
            if sql2 != draft:
                h_list = [
                    hint_entry(h, v, hfb, checked=(i == 0), prep=pr2 or probe_rep)
                    for i, (h, v, hp, h3, hfb) in enumerate(pub_failing_hints[:2])
                ]
                records.append(make_rec(
                    tid, db_id, "multi_turn_retry", question, schema, ordered, example,
                    draft, d_fb, d_warns, h_list or None, pr2 or probe_rep, "retry", gold,
                    previous_attempt={"sql": sql2, "feedback": fb2, "warnings": w2},
                    extra_meta={"kind2": kind2, "rc": rc},
                ))
                regime_counts["multi_turn_retry"] += 1

    pool.close()

    # Add capped multi-hint-with-TP examples
    rng.shuffle(multi_hint_tp_pool)
    for rec in multi_hint_tp_pool[: args.max_multi_hint_with_tp]:
        records.append(rec)
        regime_counts["multi_hint_synth_tp"] += 1

    # Add balanced True-Positive keep examples (heavily prioritizing rc==0 and rc==1!)
    by_prio: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for p, rec in tp_keep_pool:
        by_prio[p].append(rec)
    for p in by_prio:
        rng.shuffle(by_prio[p])

    fp_total = (
        regime_counts["audit_fp_real"]
        + regime_counts["audit_fp_real_ctx"]
        + regime_counts["audit_fp_real_single"]
        + regime_counts["audit_fp_synth"]
    )
    tp_target = min(args.max_tp_keep, max(int(fp_total * 0.72), 260))
    chosen_tp: list[dict] = []
    chosen_tp.extend(by_prio[0][: int(tp_target * 0.45)])
    chosen_tp.extend(by_prio[1][: int(tp_target * 0.40)])
    rem = tp_target - len(chosen_tp)
    if rem > 0:
        chosen_tp.extend(by_prio[2][: min(rem, 45)])
    rem = tp_target - len(chosen_tp)
    if rem > 0:
        leftovers = by_prio[0][int(tp_target * 0.45):] + by_prio[1][int(tp_target * 0.40):]
        rng.shuffle(leftovers)
        chosen_tp.extend(leftovers[:rem])
    for rec in chosen_tp:
        records.append(rec)
        regime_counts["audit_tp_keep"] += 1

    # Strict provenance check
    out_ids = {r["id"] for r in records}
    out_dbs = {r["db_id"] for r in records}
    assert out_ids.issubset(train_ids), f"LEAKAGE DETECTED in IDs: {out_ids - train_ids}"
    assert out_dbs.issubset(train_dbs), f"LEAKAGE DETECTED in DBs: {out_dbs - train_dbs}"
    assert all(r["target"].strip() for r in records), "Empty target SQL!"

    rng.shuffle(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    provenance = {
        "prompt_contract": "v2-rich-chat (E:/MP3/exp/harness/prompt_format.py)",
        "split_used": "train_only",
        "unique_task_ids": len(out_ids),
        "unique_db_ids": len(out_dbs),
        "total_examples": len(records),
        "audit_fp_total": fp_total,
        "audit_tp_keep_total": regime_counts["audit_tp_keep"],
        "regime_counts": dict(regime_counts),
        "corrupt_kind_counts": dict(corrupt_kind_counts.most_common()),
        "seed": args.seed,
        "source_files_sha256": {
            "train.jsonl": sha256_file(train_path),
            "train_references.json": sha256_file(refs_path),
            "repair_hints.json": sha256_file(hints_path),
            "public_results.json": sha256_file(pub_path),
            "manifest.json": sha256_file(manifest_path),
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    prov_path = args.out.parent / "data_provenance_v2.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(records)} strictly train-only v2-rich-chat examples -> {args.out}")


if __name__ == "__main__":
    main()
