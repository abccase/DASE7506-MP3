"""Static SQL & schema auditor for MP3 SQL Repair Agent.

Pure stdlib + sqlglot. Zero task hardcoding, zero database access.
Provides:
  - SchemaCatalog: indexes tables, columns, types, PKs, FKs, and 2-hop FK graph.
  - audit_sql(...): computes structured features and human-readable warning tags
    for false-positive detection, candidate ranking, and LLM prompt feedback.
"""
from __future__ import annotations

import difflib
import re
from collections import defaultdict, deque

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope


QUESTION_PATTERNS = {
    "distinct": r"\b(distinct|different|unique)\b",
    "count": r"\b(how many|number of|count)\b",
    "avg": r"\b(average|avg|mean)\b",
    "sum": r"\b(total|sum)\b",
    "max": r"\b(most|highest|largest|maximum|latest|newest|biggest|oldest)\b",
    "min": r"\b(least|lowest|smallest|minimum|earliest|youngest|shortest)\b",
    "alphabetical": r"\b(alphabetical|alphabetically|lexicographical|lexicographically)\b",
    "negation": r"\b(never|without|except|neither|nor|not\s+(?:in|living|have|has|had|participat|belong|own|visit|order|buy|bought|enroll|take|taken|attend|work))\b",
    "both": r"\bboth\b",
    "at_least": r"\b(at least|more than|greater than|over|above|no less than)\b",
}


def _norm_ident(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _stem(word: str) -> str:
    w = _norm_ident(word)
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


class SchemaCatalog:
    """Indexed view of the tool-provided schema list."""

    def __init__(self, schema: list[dict]):
        self.raw = schema or []
        self.tables: dict[str, dict] = {}
        self.fk_edges: set[tuple[str, str, str, str]] = set()
        self.fk_adj: dict[str, set[str]] = defaultdict(set)
        self.col_owners: dict[str, list[str]] = defaultdict(list)

        for t in self.raw:
            tname = t["name"]
            tl = tname.lower()
            cols: dict[str, dict] = {}
            pks: set[str] = set()
            for c in t.get("columns") or []:
                cname = c["name"]
                cl = cname.lower()
                ctype = (c.get("type") or "").strip()
                is_pk = bool(c.get("pk"))
                cols[cl] = {"name": cname, "type": ctype, "type_lower": ctype.lower(), "pk": is_pk}
                if is_pk:
                    pks.add(cl)
                self.col_owners[cl].append(tl)
            fks = []
            for fk in t.get("foreign_keys") or []:
                f_col = fk["from"].lower()
                to_tbl = fk["table"].lower()
                to_col = fk["to"].lower()
                fks.append((f_col, to_tbl, to_col))
                self.fk_edges.add((tl, f_col, to_tbl, to_col))
                self.fk_edges.add((to_tbl, to_col, tl, f_col))
                self.fk_adj[tl].add(to_tbl)
                self.fk_adj[to_tbl].add(tl)
            self.tables[tl] = {"name": tname, "cols": cols, "pks": pks, "fks": fks}

    def has_table(self, table: str) -> bool:
        return table.lower() in self.tables

    def real_table(self, table: str) -> str | None:
        info = self.tables.get(table.lower())
        return info["name"] if info else None

    def has_col(self, table: str, col: str) -> bool:
        info = self.tables.get(table.lower())
        return bool(info) and col.lower() in info["cols"]

    def real_col(self, table: str, col: str) -> str | None:
        info = self.tables.get(table.lower())
        if not info:
            return None
        c = info["cols"].get(col.lower())
        return c["name"] if c else None

    def col_info(self, table: str, col: str) -> dict | None:
        info = self.tables.get(table.lower())
        return info["cols"].get(col.lower()) if info else None

    def type_family(self, table: str, col: str) -> str:
        ci = self.col_info(table, col)
        if not ci:
            return "unknown"
        tp = ci["type_lower"]
        if any(k in tp for k in ("int", "num", "real", "floa", "doub", "dec", "bool")):
            return "num"
        if any(k in tp for k in ("char", "text", "clob", "str", "var")):
            return "text"
        if any(k in tp for k in ("date", "time")):
            return "time"
        return "other"

    def is_text_col(self, table: str, col: str) -> bool:
        fam = self.type_family(table, col)
        if fam == "text":
            return True
        if fam == "num":
            return False
        cl = col.lower()
        # Fallback when SQLite type is empty/unspecified
        if cl == "id" or cl.endswith("_id") or cl.endswith("id"):
            return False
        return True

    def fk_distance(self, t1: str, t2: str, max_hops: int = 3) -> int | None:
        t1, t2 = t1.lower(), t2.lower()
        if t1 == t2:
            return 0
        if t1 not in self.fk_adj or t2 not in self.fk_adj:
            return None
        q = deque([(t1, 0)])
        seen = {t1}
        while q:
            cur, d = q.popleft()
            if d >= max_hops:
                continue
            for nxt in self.fk_adj[cur]:
                if nxt == t2:
                    return d + 1
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return None

    def is_explicit_fk_pair(self, t1: str, c1: str, t2: str, c2: str) -> bool:
        return (t1.lower(), c1.lower(), t2.lower(), c2.lower()) in self.fk_edges

    def is_implicit_clean_join(self, t1: str, c1: str, t2: str, c2: str) -> bool:
        t1, c1, t2, c2 = t1.lower(), c1.lower(), t2.lower(), c2.lower()
        if self.is_explicit_fk_pair(t1, c1, t2, c2):
            return True
        # Check if same column name
        if c1 == c2:
            # Red flag: if both tables have explicit FKs elsewhere and a 2-hop bridge exists,
            # and c1 is PK of BOTH tables (e.g., Rent_Arrears.council_tax_id = Council_Tax.council_tax_id),
            # joining directly on their independent PKs is suspicious!
            pk1 = c1 in self.tables.get(t1, {}).get("pks", set())
            pk2 = c2 in self.tables.get(t2, {}).get("pks", set())
            dist = self.fk_distance(t1, t2, max_hops=2)
            if pk1 and pk2 and dist == 2:
                return False
            return True
        # Allow clean implicit FK like trip.start_station_id = station.id
        s1, s2 = _stem(t1), _stem(t2)
        n1, n2 = _norm_ident(c1), _norm_ident(c2)
        if n2 == "id" and (s2 in n1) and n1.endswith("id"):
            return True
        if n1 == "id" and (s1 in n2) and n2.endswith("id"):
            return True
        return False


def resolve_column(col: exp.Column, alias_map: dict[str, str], cat: SchemaCatalog) -> tuple[str | None, str]:
    cname = col.name.lower()
    if col.table:
        t = alias_map.get(col.table.lower(), col.table.lower())
        if cat.has_col(t, cname):
            return t, cname
        return None, cname
    matches = [t for t in set(alias_map.values()) if cat.has_col(t, cname)]
    if len(matches) == 1:
        return matches[0], cname
    return None, cname


def _normalize_col_label(label: str) -> tuple[str, str]:
    """Normalize a result/example column name into (agg_or_empty, core_col_name)."""
    s = (label or "").strip().lower()
    m = re.match(r"^(count|sum|avg|min|max|total)\s*\(\s*(?:distinct\s+)?([^)]*)\)\s*$", s)
    if m:
        agg = m.group(1)
        inner = m.group(2).strip()
        if "." in inner:
            inner = inner.split(".")[-1]
        return agg, inner.strip('"`[] ')
    if "." in s:
        s = s.split(".")[-1]
    return "", s.strip('"`[] ')


def column_alignment_score(actual_cols: list[str], expected_cols: list[str]) -> float:
    """Returns alignment score in [-1.0, 1.0]. -1.0 if column counts mismatch."""
    if not expected_cols:
        return 0.0
    if not actual_cols or len(actual_cols) != len(expected_cols):
        return -1.0
    total = 0.0
    for a, b in zip(actual_cols, expected_cols):
        agg_a, core_a = _normalize_col_label(a)
        agg_b, core_b = _normalize_col_label(b)
        if agg_a == agg_b and core_a == core_b:
            total += 1.0
        elif agg_a and agg_a == agg_b:
            # e.g. count(*) vs count(id) or count(distinct x)
            total += 0.8
        elif core_a and core_b and (core_a == core_b or core_a in core_b or core_b in core_a):
            total += 0.75
        elif agg_b and not agg_a:
            # Expected aggregate but actual was aliased (e.g. 'song_count' vs 'count(*)')
            if agg_b in core_a or (agg_b == "count" and ("num" in core_a or "cnt" in core_a or "total" in core_a)):
                total += 0.7
            else:
                total += 0.2
        else:
            sim = difflib.SequenceMatcher(None, core_a or a.lower(), core_b or b.lower()).ratio()
            total += sim * 0.6
    return total / len(expected_cols)


def extract_sql_predicates(sql: str, cat: SchemaCatalog) -> list[dict]:
    """Extract string/numeric equality, IN, and LIKE predicates with resolved (table, col)."""
    preds: list[dict] = []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        root = build_scope(tree)
    except Exception:
        return preds
    if root is None:
        return preds
    for scope in root.traverse():
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
            for node in clause.find_all((exp.EQ, exp.NEQ)):
                op = "=" if isinstance(node, exp.EQ) else "!="
                col_node, lit_node = None, None
                if isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Literal):
                    col_node, lit_node = node.this, node.expression
                elif isinstance(node.expression, exp.Column) and isinstance(node.this, exp.Literal):
                    col_node, lit_node = node.expression, node.this
                elif isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Column):
                    # SQLite allows double-quoted string literals (e.g. WHERE lname = "Linda");
                    # sqlglot parses quoted non-schema identifiers as quoted Column nodes!
                    t1, c1 = resolve_column(node.this, alias_map, cat)
                    t2, c2 = resolve_column(node.expression, alias_map, cat)
                    if t1 and not t2 and getattr(node.expression.this, "quoted", False):
                        preds.append({
                            "op": op,
                            "table": t1,
                            "col": c1,
                            "value": str(node.expression.name),
                            "is_string": True,
                        })
                        continue
                    if t2 and not t1 and getattr(node.this.this, "quoted", False):
                        preds.append({
                            "op": op,
                            "table": t2,
                            "col": c2,
                            "value": str(node.this.name),
                            "is_string": True,
                        })
                        continue
                if col_node is not None and lit_node is not None:
                    t, c = resolve_column(col_node, alias_map, cat)
                    preds.append({
                        "op": op,
                        "table": t,
                        "col": c,
                        "value": str(lit_node.this),
                        "is_string": bool(lit_node.is_string),
                    })
            for inp in clause.find_all(exp.In):
                if isinstance(inp.this, exp.Column):
                    t, c = resolve_column(inp.this, alias_map, cat)
                    for item in inp.expressions or []:
                        if isinstance(item, exp.Literal):
                            preds.append({
                                "op": "IN",
                                "table": t,
                                "col": c,
                                "value": str(item.this),
                                "is_string": bool(item.is_string),
                            })
            for lk in clause.find_all(exp.Like):
                if isinstance(lk.this, exp.Column) and isinstance(lk.expression, exp.Literal) and lk.expression.is_string:
                    t, c = resolve_column(lk.this, alias_map, cat)
                    preds.append({
                        "op": "LIKE",
                        "table": t,
                        "col": c,
                        "value": str(lk.expression.this),
                        "is_string": True,
                    })
    return preds


def audit_sql(
    sql: str,
    cat: SchemaCatalog | None,
    question: str,
    ordered: bool,
    example_cols: list[str],
    actual_cols: list[str] | None = None,
    probe_report: dict | None = None,
) -> dict:
    """Comprehensive static + optional DB-probe audit of a candidate SQL query."""
    features: dict = {
        "parses": False,
        "suspicious_non_fk_joins": 0,
        "non_fk_joins": 0,
        "cross_type_joins": 0,
        "total_joins": 0,
        "table_count": 0,
        "has_distinct": False,
        "has_top_order_by": False,
        "has_limit": False,
        "has_left_join": False,
        "has_set_op": False,
        "has_subquery": False,
        "ordered_missing_orderby": False,
        "alphabetical_id_order": False,
        "extra_distinct_no_cue": False,
        "missing_distinct_with_cue": False,
        "negation_uses_neq_on_join": False,
        "col_count_match": False,
        "col_align_score": 0.0,
        "ungrounded_literals": 0,
        "literal_has_similar_in_col": 0,
        "literal_found_in_other_col": 0,
        "all_literals_grounded": False,
        "has_string_literals": False,
    }
    warnings: list[str] = []

    q_low = (question or "").lower()
    q_cues = {k: bool(re.search(pat, q_low)) for k, pat in QUESTION_PATTERNS.items()}

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        features["parses"] = True
    except Exception:
        warnings.append("PARSE_ERROR")
        return {"features": features, "warnings": warnings}

    features["has_top_order_by"] = bool(tree.args.get("order"))
    features["has_limit"] = bool(tree.args.get("limit"))
    features["has_set_op"] = isinstance(tree, exp.SetOperation) or any(
        isinstance(n, exp.SetOperation) for n in tree.walk()
    )
    if isinstance(tree, exp.Select):
        features["has_distinct"] = bool(tree.args.get("distinct"))
        sel_exprs = list(tree.expressions)
        if actual_cols is None:
            actual_cols = [e.alias_or_name or e.sql(dialect="sqlite") for e in sel_exprs]
    elif isinstance(tree, exp.SetOperation) and isinstance(tree.this, exp.Select):
        features["has_distinct"] = bool(tree.this.args.get("distinct"))
        if actual_cols is None:
            actual_cols = [e.alias_or_name or e.sql(dialect="sqlite") for e in tree.this.expressions]

    if actual_cols is not None and example_cols:
        features["col_count_match"] = len(actual_cols) == len(example_cols)
        features["col_align_score"] = column_alignment_score(actual_cols, example_cols)
        if not features["col_count_match"]:
            warnings.append(f"COL_COUNT_MISMATCH(got={len(actual_cols)},want={len(example_cols)})")

    if ordered and not features["has_top_order_by"]:
        features["ordered_missing_orderby"] = True
        warnings.append("ORDERED_REQUIRED_BUT_NO_ORDER_BY")

    if features["has_distinct"] and not q_cues["distinct"]:
        features["extra_distinct_no_cue"] = True
    if not features["has_distinct"] and q_cues["distinct"]:
        features["missing_distinct_with_cue"] = True
        warnings.append("MISSING_DISTINCT_FOR_CUE")

    tables_used = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    features["table_count"] = len(tables_used)

    try:
        root = build_scope(tree)
    except Exception:
        root = None

    if root is not None:
        scopes = list(root.traverse())
        features["has_subquery"] = len(scopes) > 1
        for scope in scopes:
            alias_map = {}
            for k, src in scope.sources.items():
                if isinstance(src, exp.Table):
                    alias_map[k.lower()] = src.name.lower()
                    alias_map[src.name.lower()] = src.name.lower()
            sel = scope.expression
            if not isinstance(sel, exp.Select):
                continue

            # Join audit
            for j in sel.args.get("joins") or []:
                features["total_joins"] += 1
                if (j.side or "").upper() == "LEFT":
                    features["has_left_join"] = True
                on = j.args.get("on")
                if on is None or cat is None:
                    continue
                for eq in on.find_all(exp.EQ):
                    if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column):
                        t1, c1 = resolve_column(eq.this, alias_map, cat)
                        t2, c2 = resolve_column(eq.expression, alias_map, cat)
                        q1 = (eq.this.table or "").lower()
                        q2 = (eq.expression.table or "").lower()
                        if t1 and t2 and t1 == t2 and q1 == q2 and c1 == c2:
                            features["suspicious_non_fk_joins"] += 1
                            warnings.append(f"TAUTOLOGICAL_JOIN({eq.sql(dialect='sqlite')})")
                        elif t1 and t2 and t1 != t2:
                            if cat.fk_edges and not cat.is_explicit_fk_pair(t1, c1, t2, c2):
                                features["non_fk_joins"] += 1
                                if not cat.is_implicit_clean_join(t1, c1, t2, c2):
                                    features["suspicious_non_fk_joins"] += 1
                                    warnings.append(
                                        f"NON_FK_JOIN({cat.real_table(t1)}.{cat.real_col(t1,c1)}="
                                        f"{cat.real_table(t2)}.{cat.real_col(t2,c2)})"
                                    )
                            f1, f2 = cat.type_family(t1, c1), cat.type_family(t2, c2)
                            if f1 in ("num", "text") and f2 in ("num", "text") and f1 != f2:
                                features["cross_type_joins"] += 1
                                warnings.append(
                                    f"CROSS_TYPE_JOIN({cat.real_table(t1)}.{cat.real_col(t1,c1)}="
                                    f"{cat.real_table(t2)}.{cat.real_col(t2,c2)})"
                                )

            # Alphabetical ORDER BY on integer/_id column audit (works with or without schema!)
            if q_cues["alphabetical"] and sel.args.get("order"):
                for ordered_item in sel.args["order"].expressions:
                    for col in ordered_item.find_all(exp.Column):
                        cname_l = col.name.lower()
                        if cat is not None:
                            t, c = resolve_column(col, alias_map, cat)
                            if t and c and (cat.type_family(t, c) == "num" or c == "id" or c.endswith("_id")):
                                features["alphabetical_id_order"] = True
                                warnings.append(
                                    f"ALPHABETICAL_ORDERS_BY_ID({cat.real_table(t)}.{cat.real_col(t,c)})"
                                )
                        elif cname_l == "id" or cname_l.endswith("_id"):
                            features["alphabetical_id_order"] = True
                            warnings.append(f"ALPHABETICAL_ORDERS_BY_ID({col.name})")

            # Negation using != / NOT LIKE on a joined query instead of NOT IN / EXCEPT
            if q_cues["negation"] and features["total_joins"] > 0 and not features["has_set_op"] and not features["has_subquery"]:
                where = sel.args.get("where")
                if where is not None:
                    has_neq = any(isinstance(n, exp.NEQ) for n in where.walk())
                    if has_neq:
                        features["negation_uses_neq_on_join"] = True
                        warnings.append("NEGATION_NEQ_ON_JOIN_CONSIDER_EXCEPT_OR_NOT_IN")

    # Optional DB-probe literal grounding audit
    if probe_report and cat is not None:
        preds = [p for p in extract_sql_predicates(sql, cat) if p["is_string"] and p["op"] in ("=", "IN")]
        if preds:
            features["has_string_literals"] = True
            checks = probe_report.get("literal_checks") or {}
            cross = probe_report.get("cross_col_hits") or {}
            samples_by_key = probe_report.get("samples_by_key") or {}
            grounded_count = 0
            checked_count = 0
            for p in preds:
                if not (p["table"] and p["col"]):
                    continue
                key = (p["table"].lower(), p["col"].lower(), p["value"])
                info = checks.get(key)
                if info is None:
                    # Candidate introduced a literal after db_probe ran; infer grounding from column samples
                    col_samples = samples_by_key.get((p["table"].lower(), p["col"].lower()))
                    if col_samples is not None:
                        checked_count += 1
                        if p["value"] in col_samples:
                            grounded_count += 1
                        else:
                            features["ungrounded_literals"] += 1
                            sim_vals = [v for v in col_samples if v.lower() == p["value"].lower() or p["value"].lower() in v.lower()]
                            if sim_vals:
                                features["literal_has_similar_in_col"] += 1
                                warnings.append(
                                    f"LITERAL_MISMATCH({cat.real_table(p['table'])}.{cat.real_col(p['table'],p['col'])}="
                                    f"'{p['value']}' has 0 rows; similar: {sim_vals[:4]})"
                                )
                            else:
                                warnings.append(
                                    f"UNGROUNDED_LITERAL({cat.real_table(p['table'])}.{cat.real_col(p['table'],p['col'])}='{p['value']}' has 0 rows in public DB)"
                                )
                    continue
                checked_count += 1
                if info["exact_count"] == 0:
                    features["ungrounded_literals"] += 1
                    sim_vals = info.get("similar_values") or []
                    if sim_vals:
                        features["literal_has_similar_in_col"] += 1
                        warnings.append(
                            f"LITERAL_MISMATCH({cat.real_table(p['table'])}.{cat.real_col(p['table'],p['col'])}="
                            f"'{p['value']}' has 0 rows; similar: {sim_vals[:4]})"
                        )
                    else:
                        other_hits = cross.get(p["value"].lower()) or []
                        other_hits = [h for h in other_hits if (h[0].lower(), h[1].lower()) != (p["table"].lower(), p["col"].lower())]
                        if other_hits:
                            features["literal_found_in_other_col"] += 1
                            locs = ", ".join(f"{h[0]}.{h[1]}" for h in other_hits[:3])
                            warnings.append(
                                f"LITERAL_WRONG_COLUMN('{p['value']}' not in {cat.real_table(p['table'])}.{cat.real_col(p['table'],p['col'])}; found in {locs})"
                            )
                        else:
                            warnings.append(
                                f"UNGROUNDED_LITERAL({cat.real_table(p['table'])}.{cat.real_col(p['table'],p['col'])}='{p['value']}' has 0 rows in public DB)"
                            )
                else:
                    grounded_count += 1
            if checked_count > 0 and grounded_count == checked_count:
                features["all_literals_grounded"] = True

    return {"features": features, "warnings": warnings}


def validate_executable_locally(sql: str, cat: SchemaCatalog | None) -> tuple[bool, str]:
    """Fast static pre-check before spending a `check` tool call on an LLM candidate.

    Detects:
      - SQLite syntax / non-SELECT errors
      - Nonexistent tables
      - Nonexistent or out-of-scope columns (including uncorrelated derived-table outer alias refs!)
      - Ambiguous unqualified columns across >=2 joined real tables
    Returns (is_valid, reason_if_invalid).
    """
    if not sql or not sql.strip():
        return False, "empty"
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        return False, f"parse_error:{e}"
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return False, "not_select"
    if cat is None or not cat.tables:
        return True, "ok"

    try:
        root = build_scope(tree)
    except Exception:
        return True, "ok"
    if root is None:
        return True, "ok"

    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }

    for tbl in tree.find_all(exp.Table):
        tname = (tbl.name or "").lower()
        if tname and tname not in cte_names and not cat.has_table(tname):
            return False, f"no_such_table:{tbl.name}"

    for scope in root.traverse():
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        # Map local aliases -> real table names (or 'SUBQUERY' for derived tables/CTEs)
        local_real: dict[str, str] = {}
        has_derived = False
        for k, src in scope.sources.items():
            kl = k.lower()
            if isinstance(src, exp.Table):
                tn = src.name.lower()
                if tn in cte_names:
                    has_derived = True
                else:
                    local_real[kl] = tn
            else:
                has_derived = True

        # Outer scope aliases are ONLY accessible inside correlated WHERE/HAVING/SELECT subqueries,
        # NEVER inside a FROM/JOIN derived table `FROM (SELECT ... WHERE outer.x = ...) AS sub`!
        is_derived_subquery = bool(scope.is_derived_table)
        outer_real: dict[str, str] = {}
        if not is_derived_subquery:
            anc = scope.parent
            while anc is not None:
                for k, src in anc.sources.items():
                    if isinstance(src, exp.Table) and src.name.lower() not in cte_names:
                        outer_real.setdefault(k.lower(), src.name.lower())
                anc = anc.parent

        select_aliases = {
            e.alias.lower() for e in sel.expressions if isinstance(e, exp.Alias) and e.alias
        }

        for col in sel.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not sel:
                continue
            cname = col.name.lower()
            qual = col.table.lower() if col.table else ""
            if qual:
                if qual in local_real:
                    if not cat.has_col(local_real[qual], cname):
                        return False, f"no_such_col:{col.table}.{col.name}"
                elif qual in outer_real:
                    if not cat.has_col(outer_real[qual], cname):
                        return False, f"no_such_col:{col.table}.{col.name}"
                elif has_derived:
                    continue
                else:
                    return False, f"unknown_qualifier:{col.table}.{col.name}"
            else:
                # SQLite double-quoted string literal in comparison (e.g. state != "California")
                if getattr(col.this, "quoted", False) and isinstance(
                    col.parent, (exp.EQ, exp.NEQ, exp.In, exp.Like)
                ):
                    continue
                if cname in select_aliases:
                    continue
                if has_derived:
                    continue
                local_owners = [rt for rt in set(local_real.values()) if cat.has_col(rt, cname)]
                if len(local_owners) >= 2:
                    # Note: if two aliases of the same or different tables both own cname
                    real_alias_owners = [al for al, rt in local_real.items() if cat.has_col(rt, cname)]
                    if len(real_alias_owners) >= 2:
                        return False, f"ambiguous_col:{col.name}"
                elif len(local_owners) == 0:
                    outer_owners = [rt for rt in set(outer_real.values()) if cat.has_col(rt, cname)]
                    if not outer_owners:
                        return False, f"no_such_col:{col.name}"

    return True, "ok"

