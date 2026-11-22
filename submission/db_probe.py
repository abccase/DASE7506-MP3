"""Single-call SQLite DB value prober & deterministic literal repairer.

Builds ONE safe, fast, single-row read-only SQLite query that obeys every limit
in mp3_sql/sqlite_tool.py (<= 1 row returned so never truncated by rows[:8],
0 UNION ALL branches so never hits SQLITE_LIMIT_COMPOUND_SELECT, <= 32 columns
<< SQLITE_LIMIT_COLUMN=100, <= 12*40 chars per cell << 4000 bytes, only allowed
functions: group_concat, substr, lower, instr, length, trim, count).
"""
from __future__ import annotations

import copy
import re
from collections import defaultdict

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

from static_audit import SchemaCatalog, extract_sql_predicates, resolve_column

SEP_SQL = "char(31)"
SEP_CHAR = "\x1f"
MAX_SAMPLE_COLS = 10
MAX_LIT_CHECKS = 6
MAX_CROSS_CHECKS = 8
MAX_TABLE_COUNTS = 6
MAX_SCAN_ROWS = 3000
MAX_DISTINCT_VALS = 8
MAX_CELL_LEN = 48


def _q_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _q_str(val: str) -> str:
    return "'" + val.replace("'", "''") + "'"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 2}


def _extract_question_quoted_literals(question: str) -> list[str]:
    found = []
    for m in re.finditer(r"['\"]([^'\"]{2,50})['\"]", question or ""):
        val = m.group(1).strip()
        if val and val not in found:
            found.append(val)
    return found


def build_probe_query(
    schema: list[dict],
    question: str,
    candidate_sqls: list[str],
    example_cols: list[str] | None = None,
) -> tuple[str | None, dict]:
    """Build ONE read-only SQLite probe query and a plan dict to parse its 1-row result.

    Returns (sql_or_None, plan_dict).
    """
    cat = SchemaCatalog(schema)
    if not cat.tables:
        return None, {}

    q_toks = _tokens(question)
    ex_toks = _tokens(" ".join(example_cols or []))

    # 1. Gather tables, columns, and string literals referenced by candidate SQLs
    ref_tables: list[str] = []
    pred_cols: list[tuple[str, str]] = []
    lit_checks: list[tuple[str, str, str]] = []  # (tbl_low, col_low, val)
    distinct_lits: list[str] = []

    for sql in candidate_sqls:
        if not sql:
            continue
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except Exception:
            continue
        for t in tree.find_all(exp.Table):
            tl = t.name.lower()
            if tl in cat.tables and tl not in ref_tables:
                ref_tables.append(tl)
        for p in extract_sql_predicates(sql, cat):
            if p["table"] and p["col"]:
                pair = (p["table"], p["col"])
                if pair not in pred_cols:
                    pred_cols.append(pair)
                if p["is_string"] and p["op"] in ("=", "IN", "!=", "LIKE"):
                    raw_val = p["value"].strip("%") if p["op"] == "LIKE" else p["value"]
                    if raw_val and len(raw_val) <= 80:
                        key = (p["table"], p["col"], raw_val)
                        if key not in lit_checks:
                            lit_checks.append(key)
                        if raw_val not in distinct_lits and len(raw_val) >= 2:
                            distinct_lits.append(raw_val)

    for qlit in _extract_question_quoted_literals(question):
        if qlit not in distinct_lits and len(qlit) <= 80:
            distinct_lits.append(qlit)

    # Also find schema tables whose names overlap with the question
    q_tables: list[str] = []
    for tl, info in cat.tables.items():
        ttoks = _tokens(info["name"].replace("_", " "))
        if ttoks & q_toks:
            q_tables.append(tl)

    relevant_tables = list(dict.fromkeys(ref_tables + q_tables))
    if not relevant_tables:
        relevant_tables = list(cat.tables.keys())[:4]

    # 2. Rank text columns to sample distinct values from
    scored_cols: list[tuple[float, str, str]] = []
    seen_cols: set[tuple[str, str]] = set()

    # Priority A: columns directly used in predicates (even if numeric/text ambiguous like duration)
    for t, c in pred_cols:
        if (t, c) in seen_cols or not cat.has_col(t, c):
            continue
        # Skip pure primary keys unless string predicate was applied
        ci = cat.col_info(t, c)
        is_str_pred = any(lt == t and lc == c for lt, lc, _ in lit_checks)
        if not is_str_pred and not cat.is_text_col(t, c):
            continue
        seen_cols.add((t, c))
        scored_cols.append((100.0 if is_str_pred else 60.0, t, c))

    # Priority B: other text columns in relevant tables (great for catching ResTypeName vs ResName, Fname vs LName)
    for t in relevant_tables:
        t_in_ref = t in ref_tables
        for cl, ci in cat.tables[t]["cols"].items():
            if (t, cl) in seen_cols:
                continue
            if not cat.is_text_col(t, cl):
                continue
            ctoks = _tokens(ci["name"].replace("_", " "))
            score = 20.0 if t_in_ref else 10.0
            if ctoks & q_toks:
                score += 25.0
            if ctoks & ex_toks:
                score += 10.0
            if any(k in cl for k in ("name", "type", "status", "code", "state", "country", "city", "category", "team", "gender", "sex", "detail", "title", "time", "duration")):
                score += 12.0
            # Sibling column bonus: same table already has a literal predicate (e.g. LName -> Fname)
            if any(lt == t for lt, _, _ in lit_checks):
                score += 15.0
            seen_cols.add((t, cl))
            scored_cols.append((score, t, cl))

    scored_cols.sort(key=lambda x: (-x[0], x[1], x[2]))
    sample_cols = [(t, c) for _, t, c in scored_cols[:MAX_SAMPLE_COLS]]

    # 3. Select literal checks (up to MAX_LIT_CHECKS)
    lit_checks = lit_checks[:MAX_LIT_CHECKS]

    # 4. Select cross-column literal searches for literals used in predicates (up to MAX_CROSS_CHECKS)
    # Checks whether lit appears (case-insensitive substring) in OTHER candidate text columns.
    cross_checks: list[tuple[str, str, str]] = []  # (tbl_low, col_low, val)
    lit_pairs = {(t, c, v.lower()) for t, c, v in lit_checks}
    for val in distinct_lits[:3]:
        vlow = val.lower()
        for t, c in sample_cols:
            if (t, c, vlow) in lit_pairs:
                continue
            if len(cross_checks) >= MAX_CROSS_CHECKS:
                break
            cross_checks.append((t, c, val))

    # 5. Table row counts for referenced tables (up to MAX_TABLE_COUNTS)
    count_tables = relevant_tables[:MAX_TABLE_COUNTS]

    # 6. Assemble single-row wide SELECT expression list
    exprs: list[str] = []
    plan: dict = {
        "samples": {},       # alias -> (real_tbl, real_col, tbl_low, col_low)
        "lit_exact": {},     # alias -> (real_tbl, real_col, tbl_low, col_low, val)
        "lit_similar": {},   # alias -> (real_tbl, real_col, tbl_low, col_low, val)
        "cross_hits": {},    # alias -> (real_tbl, real_col, tbl_low, col_low, val)
        "table_counts": {},  # alias -> (real_tbl, tbl_low)
    }

    for i, (t, c) in enumerate(sample_cols):
        rt, rc = cat.real_table(t), cat.real_col(t, c)
        al = f"s{i}"
        plan["samples"][al] = (rt, rc, t, c)
        exprs.append(
            f"(SELECT group_concat(v, {SEP_SQL}) FROM ("
            f"SELECT DISTINCT substr(CAST(x AS TEXT), 1, {MAX_CELL_LEN}) AS v FROM ("
            f"SELECT {_q_ident(rc)} AS x FROM {_q_ident(rt)} "
            f"WHERE {_q_ident(rc)} IS NOT NULL LIMIT {MAX_SCAN_ROWS}"
            f") ORDER BY 1 LIMIT {MAX_DISTINCT_VALS})) AS {al}"
        )

    for i, (t, c, val) in enumerate(lit_checks):
        rt, rc = cat.real_table(t), cat.real_col(t, c)
        al_e = f"e{i}"
        al_l = f"l{i}"
        plan["lit_exact"][al_e] = (rt, rc, t, c, val)
        plan["lit_similar"][al_l] = (rt, rc, t, c, val)
        v_esc = _q_str(val)
        v_low = _q_str(val.lower())
        exprs.append(
            f"(SELECT COUNT(*) FROM {_q_ident(rt)} WHERE {_q_ident(rc)} = {v_esc}) AS {al_e}"
        )
        # Match either col contains val (case-insensitive) OR val contains col (len>=2)
        if len(val) >= 2:
            cond = (
                f"instr(lower(CAST({_q_ident(rc)} AS TEXT)), {v_low}) > 0 "
                f"OR (length(trim(CAST({_q_ident(rc)} AS TEXT))) >= 2 AND instr({v_low}, lower(trim(CAST({_q_ident(rc)} AS TEXT)))) > 0)"
            )
        else:
            cond = f"instr(lower(CAST({_q_ident(rc)} AS TEXT)), {v_low}) > 0"
        exprs.append(
            f"(SELECT group_concat(v, {SEP_SQL}) FROM ("
            f"SELECT DISTINCT substr(CAST({_q_ident(rc)} AS TEXT), 1, {MAX_CELL_LEN}) AS v "
            f"FROM {_q_ident(rt)} "
            f"WHERE {_q_ident(rc)} IS NOT NULL AND ({cond}) "
            f"ORDER BY 1 LIMIT 6)) AS {al_l}"
        )

    for i, (t, c, val) in enumerate(cross_checks):
        rt, rc = cat.real_table(t), cat.real_col(t, c)
        al_x = f"x{i}"
        plan["cross_hits"][al_x] = (rt, rc, t, c, val)
        v_low = _q_str(val.lower())
        exprs.append(
            f"(SELECT group_concat(v, {SEP_SQL}) FROM ("
            f"SELECT DISTINCT substr(CAST({_q_ident(rc)} AS TEXT), 1, {MAX_CELL_LEN}) AS v "
            f"FROM {_q_ident(rt)} "
            f"WHERE {_q_ident(rc)} IS NOT NULL AND instr(lower(CAST({_q_ident(rc)} AS TEXT)), {v_low}) > 0 "
            f"ORDER BY 1 LIMIT 5)) AS {al_x}"
        )

    for i, t in enumerate(count_tables):
        rt = cat.real_table(t)
        al_n = f"n{i}"
        plan["table_counts"][al_n] = (rt, t)
        exprs.append(f"(SELECT COUNT(*) FROM {_q_ident(rt)}) AS {al_n}")

    if not exprs:
        return None, {}

    sql = "SELECT " + ", ".join(exprs)
    # Guard SQL length limit (16,000 bytes)
    if len(sql.encode("utf-8")) > 14000:
        # Fallback: drop cross_hits if somehow huge
        keep_n = len(sample_cols) + 2 * len(lit_checks)
        sql = "SELECT " + ", ".join(exprs[:keep_n])
    return sql, plan


def parse_probe_result(feedback: dict, plan: dict) -> dict:
    """Parse the execute tool feedback of build_probe_query into a structured report."""
    report: dict = {
        "ok": False,
        "samples": {},          # "Table.Col" -> list[str]
        "samples_by_key": {},   # (tbl_low, col_low) -> list[str]
        "literal_checks": {},   # (tbl_low, col_low, val) -> {"table": rt, "col": rc, "exact_count": int, "similar_values": list[str]}
        "cross_col_hits": defaultdict(list),  # val_lower -> [(rt, rc, [vals])]
        "table_counts": {},     # "Table" -> int
    }
    if not feedback or not feedback.get("ok") or not feedback.get("rows"):
        return report
    cols = feedback.get("columns") or []
    row = feedback["rows"][0]
    if len(cols) != len(row):
        return report
    by_alias = dict(zip(cols, row))
    report["ok"] = True

    def _split_vals(raw) -> list[str]:
        if raw is None:
            return []
        s = str(raw).strip()
        if not s:
            return []
        return [v for v in s.split(SEP_CHAR) if v != ""]

    for al, (rt, rc, tl, cl) in (plan.get("samples") or {}).items():
        vals = _split_vals(by_alias.get(al))
        if vals:
            report["samples"][f"{rt}.{rc}"] = vals
            report["samples_by_key"][(tl, cl)] = vals

    for al, (rt, rc, tl, cl, val) in (plan.get("lit_exact") or {}).items():
        cnt = by_alias.get(al)
        cnt_int = int(cnt) if isinstance(cnt, (int, float)) else 0
        report["literal_checks"][(tl, cl, val)] = {
            "table": rt,
            "col": rc,
            "value": val,
            "exact_count": cnt_int,
            "similar_values": [],
        }

    for al, (rt, rc, tl, cl, val) in (plan.get("lit_similar") or {}).items():
        sim = _split_vals(by_alias.get(al))
        entry = report["literal_checks"].get((tl, cl, val))
        if entry is not None:
            entry["similar_values"] = sim

    for al, (rt, rc, tl, cl, val) in (plan.get("cross_hits") or {}).items():
        hits = _split_vals(by_alias.get(al))
        if hits:
            report["cross_col_hits"][val.lower()].append((rt, rc, hits))

    for al, (rt, tl) in (plan.get("table_counts") or {}).items():
        cnt = by_alias.get(al)
        if isinstance(cnt, (int, float)):
            report["table_counts"][rt] = int(cnt)

    report["cross_col_hits"] = dict(report["cross_col_hits"])
    return report


def to_prompt_db_samples(report: dict | None, max_cols: int = 12, max_vals: int = 8) -> dict[str, list[str]]:
    """Return `{'Table.Col': [vals...]}` matching `exp/lora/prompt_contract.py` 100%.

    Prioritizes similar literal matches and cross-column literal hits into the
    sampled values list so the LLM sees the exact DB spelling even on large tables.
    """
    if not report or not report.get("ok"):
        return {}
    out: dict[str, list[str]] = {}
    for col_key, vals in (report.get("samples") or {}).items():
        if vals:
            out[col_key] = list(vals[:max_vals])

    # Prepend similar values from literal checks so they are never pushed out by LIMIT
    for info in (report.get("literal_checks") or {}).values():
        sim = info.get("similar_values") or []
        if sim:
            key = f"{info['table']}.{info['col']}"
            merged = list(dict.fromkeys(list(sim) + out.get(key, [])))
            out[key] = merged[:max_vals]

    # Add any cross-column literal hits
    for hits in (report.get("cross_col_hits") or {}).values():
        for rt, rc, vals in hits:
            if not vals:
                continue
            key = f"{rt}.{rc}"
            merged = list(dict.fromkeys(list(vals) + out.get(key, [])))
            out[key] = merged[:max_vals]

    if len(out) > max_cols:
        out = dict(list(out.items())[:max_cols])
    return out


def format_probe_for_prompt(report: dict | None, max_cols: int = 10, max_vals_per_col: int = 8) -> str:
    """Compact, deterministic text representation of DB probe findings for LLM prompts.

    This is the canonical format shared between harness, lora_trainer, and model_scout.
    """
    if not report or not report.get("ok"):
        return "(none)"
    lines: list[str] = []

    tcounts = report.get("table_counts") or {}
    if tcounts:
        tc_str = ", ".join(f"{t}={n}" for t, n in tcounts.items())
        lines.append(f"Table row counts: {tc_str}")

    samples = report.get("samples") or {}
    for col_key, vals in list(samples.items())[:max_cols]:
        shown = [repr(v) for v in vals[:max_vals_per_col]]
        suffix = ", ..." if len(vals) > max_vals_per_col else ""
        lines.append(f"{col_key}: [{', '.join(shown)}{suffix}]")

    # Add explicit literal grounding warnings (high signal for both repair & FP firewall!)
    cross = report.get("cross_col_hits") or {}
    for (tl, cl, val), info in (report.get("literal_checks") or {}).items():
        if info["exact_count"] == 0:
            rt, rc = info["table"], info["col"]
            sim = info.get("similar_values") or []
            if sim:
                lines.append(
                    f"WARN: {rt}.{rc} = {val!r} has 0 exact matches; similar DB values: {sim[:5]}"
                )
            else:
                others = [
                    f"{ot}.{oc} ({ov[:3]})"
                    for ot, oc, ov in (cross.get(val.lower()) or [])
                    if (ot.lower(), oc.lower()) != (tl, cl)
                ]
                if others:
                    lines.append(
                        f"WARN: {rt}.{rc} = {val!r} has 0 exact matches; value appears in: {'; '.join(others[:3])}"
                    )
                else:
                    lines.append(
                        f"WARN: {rt}.{rc} = {val!r} has 0 exact matches in public DB."
                    )

    return "\n".join(lines) if lines else "(none)"


def _best_literal_replacement(old_val: str, candidates: list[str], question: str) -> tuple[str, str] | None:
    """Choose an unambiguous DB value replacement for a 0-match literal, or None.

    Returns (op, new_val) where op is '=' (exact replacement) or 'LIKE' (prefix pattern).
    """
    if not candidates or not old_val:
        return None
    old_low = old_val.lower().strip()
    q_low = (question or "").lower()

    # 1. Exact case-insensitive match (e.g. 'morning' -> 'Morning', 'Bangla' -> 'bangla')
    ci_matches = [c for c in candidates if c.lower().strip() == old_low]
    if len(ci_matches) == 1:
        return ("=", ci_matches[0])

    # 2. Duration / time format match: pure integer N -> 'N:00' if unique, else LIKE 'N:%' if >=2 values use M:SS format
    if re.fullmatch(r"\d{1,2}", old_low):
        time_matches = [c for c in candidates if c.strip() in (f"{old_low}:00", f"0{old_low}:00")]
        if len(time_matches) == 1:
            return ("=", time_matches[0])
        colon_vals = [c for c in candidates if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", c.strip())]
        prefix_vals = [c for c in colon_vals if c.strip().startswith(f"{old_low}:")]
        if len(colon_vals) >= 2 and len(prefix_vals) >= 1:
            return ("LIKE", f"{old_low}:%")

    # 3. Candidate appears verbatim in the question (e.g. 'Orton' -> 'Team Orton' when Q has 'Team Orton')
    in_q = [c for c in candidates if len(c) > len(old_val) and c.lower() in q_low and old_low in c.lower()]
    if len(in_q) == 1:
        return ("=", in_q[0])

    # 4. Unique superstring match containing old_val as a whole word (e.g. 'Orton' -> 'Team Orton')
    word_pat = re.compile(r"(?:^|\b|_)" + re.escape(old_low) + r"(?:$|\b|_)")
    word_matches = [c for c in candidates if word_pat.search(c.lower())]
    if len(word_matches) == 1:
        return ("=", word_matches[0])

    # 5. Unique case-insensitive substring match when old_val is reasonably specific (>=4 chars)
    if len(old_low) >= 4:
        sub_matches = [c for c in candidates if old_low in c.lower()]
        if len(sub_matches) == 1:
            return ("=", sub_matches[0])

    return None


def repair_literals_from_probe(
    sql: str,
    schema: list[dict],
    probe_report: dict | None,
    question: str,
) -> str | None:
    """Deterministically repair unambiguous string literal mismatches using DB probe results.

    Handles:
      - Exact case mismatches ('morning' -> 'Morning')
      - Numeric-to-duration mismatches ('4' -> '4:00')
      - Missing prefix/suffix ('Orton' -> 'Team Orton')
    Returns new SQL string if any literal was unambiguously repaired, else None.
    """
    if not sql or not probe_report or not probe_report.get("ok"):
        return None
    cat = SchemaCatalog(schema)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        root = build_scope(tree)
    except Exception:
        return None
    if root is None:
        return None

    checks = probe_report.get("literal_checks") or {}
    samples_by_key = probe_report.get("samples_by_key") or {}

    clone = copy.deepcopy(tree)
    try:
        clone_root = build_scope(clone)
    except Exception:
        return None
    if clone_root is None:
        return None

    changed = False
    for scope in clone_root.traverse():
        alias_map = {}
        for k, src in scope.sources.items():
            if isinstance(src, exp.Table):
                alias_map[k.lower()] = src.name.lower()
                alias_map[src.name.lower()] = src.name.lower()
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        for clause_name in ("where", "having"):
            clause = sel.args.get(clause_name)
            if clause is None:
                continue

            def _try_replace(col_node: exp.Column, lit_node: exp.Literal, parent_eq: exp.Expression | None = None) -> None:
                nonlocal changed
                if not lit_node.is_string:
                    return
                t, c = resolve_column(col_node, alias_map, cat)
                if not (t and c):
                    return
                old_val = str(lit_node.this)
                info = checks.get((t, c, old_val))
                if info is not None and info["exact_count"] > 0:
                    return  # Already grounded!
                # Gather candidate values from similar_values + column samples
                pool: list[str] = []
                if info is not None:
                    for v in info.get("similar_values") or []:
                        if v not in pool:
                            pool.append(v)
                for v in samples_by_key.get((t, c)) or []:
                    if v not in pool:
                        pool.append(v)
                rep = _best_literal_replacement(old_val, pool, question)
                if rep is None:
                    return
                rep_op, rep_val = rep
                if rep_op == "=" and rep_val != old_val:
                    lit_node.set("this", rep_val)
                    changed = True
                elif rep_op == "LIKE" and isinstance(parent_eq, exp.EQ):
                    parent_eq.replace(
                        exp.Like(this=copy.deepcopy(col_node), expression=exp.Literal.string(rep_val))
                    )
                    changed = True

            for eq in list(clause.find_all((exp.EQ, exp.NEQ))):
                if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Literal):
                    _try_replace(eq.this, eq.expression, parent_eq=eq)
                elif isinstance(eq.expression, exp.Column) and isinstance(eq.this, exp.Literal):
                    _try_replace(eq.expression, eq.this, parent_eq=eq)
            for inp in clause.find_all(exp.In):
                if isinstance(inp.this, exp.Column):
                    for item in inp.expressions or []:
                        if isinstance(item, exp.Literal):
                            _try_replace(inp.this, item, parent_eq=None)

    if not changed:
        return None
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None
