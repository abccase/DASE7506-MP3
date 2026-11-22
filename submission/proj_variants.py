"""General, schema-aware deterministic SQL repair & variant generators.

Zero task hardcoding. Pure stdlib + sqlglot.
Transforms:
  1. disambiguate_columns: fixes `ambiguous column name` using query scope + schema.
  2. prune_extra_columns: trims extra SELECT cols to match len(example.columns),
     safely inlining any pruned alias referenced by ORDER BY / GROUP BY / HAVING.
  3. complete_missing_columns: adds missing SELECT column from already-joined tables
     when len(cols) < len(example.columns).
  4. reorder_columns: type-, name-, aggregate-, and value-aware SELECT column reordering.
  5. left_to_inner_join: converts LEFT JOIN -> INNER JOIN when result has excess rows.
  6. flip_order_direction: toggles top-level ORDER BY ASC/DESC.
  7. toggle_distinct: adds/removes SELECT DISTINCT when row count indicates duplicate mismatch.
  8. repair_bogus_join_via_fk_bridge: replaces a suspicious non-FK direct join between
     t1 and t2 with the schema's unique 2-hop FK bridge table when one exists.
"""
from __future__ import annotations

import copy
import difflib
import itertools
import re

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

from static_audit import SchemaCatalog, _normalize_col_label, resolve_column


def _cell_norm(v):
    if v is None:
        return ("null", "")
    if isinstance(v, (int, float)):
        s = str(v) if isinstance(v, int) else format(round(float(v), 6), ".6f").rstrip("0").rstrip(".")
        return ("num", "0" if s == "-0" else s)
    return ("str", str(v))


def _cell_type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return "num"
    return "str"


def _col_match_score(
    expr_node: exp.Expression,
    actual_col_name: str,
    expected_col_name: str,
    actual_col_vals: list | None = None,
    expected_col_vals: list | None = None,
) -> float:
    score = 0.0
    agg_a, core_a = _normalize_col_label(actual_col_name)
    agg_e, core_e = _normalize_col_label(expected_col_name)

    # Also inspect AST expression directly (handles aliased aggregates like COUNT(*) AS cnt)
    unwrapped = expr_node.this if isinstance(expr_node, exp.Alias) else expr_node
    ast_agg = ""
    if isinstance(unwrapped, exp.AggFunc):
        ast_agg = type(unwrapped).__name__.lower()
        if ast_agg == "count":
            ast_agg = "count"
    elif any(isinstance(n, exp.AggFunc) for n in unwrapped.walk()):
        for n in unwrapped.walk():
            if isinstance(n, exp.AggFunc):
                ast_agg = type(n).__name__.lower()
                break

    eff_agg_a = agg_a or ast_agg
    if eff_agg_a and agg_e:
        if eff_agg_a == agg_e:
            score += 4.0
        else:
            score -= 1.5
    elif bool(eff_agg_a) != bool(agg_e):
        score -= 2.0

    # Core column name similarity
    ast_col_name = unwrapped.name.lower() if isinstance(unwrapped, exp.Column) else core_a
    if core_e:
        if core_a == core_e or ast_col_name == core_e:
            score += 5.0
        elif core_a and (core_a in core_e or core_e in core_a):
            score += 3.0
        elif ast_col_name and (ast_col_name in core_e or core_e in ast_col_name):
            score += 3.0
        else:
            sim = difflib.SequenceMatcher(None, ast_col_name or core_a, core_e).ratio()
            score += 2.5 * sim

    # Value & type similarity when actual and expected sample rows are available
    if actual_col_vals and expected_col_vals:
        a_norm = {_cell_norm(v) for v in actual_col_vals if v is not None}
        e_norm = {_cell_norm(v) for v in expected_col_vals if v is not None}
        if a_norm and e_norm:
            overlap = len(a_norm & e_norm) / max(1, len(e_norm))
            score += 6.0 * overlap
        a_types = {_cell_type(v) for v in actual_col_vals if v is not None}
        e_types = {_cell_type(v) for v in expected_col_vals if v is not None}
        if a_types and e_types:
            if a_types & e_types:
                score += 2.0
            else:
                score -= 3.0

    return score


def _inline_pruned_aliases(tree: exp.Select, pruned_aliases: dict[str, exp.Expression]) -> None:
    """Replace bare column references matching pruned SELECT aliases in ORDER BY / GROUP BY / HAVING."""
    if not pruned_aliases:
        return
    for clause_key in ("order", "group", "having"):
        clause = tree.args.get(clause_key)
        if clause is None:
            continue
        for col in list(clause.find_all(exp.Column)):
            if not col.table and col.name.lower() in pruned_aliases:
                col.replace(copy.deepcopy(pruned_aliases[col.name.lower()]))


def disambiguate_columns(sql: str, schema: list[dict], error_msg: str | None = None) -> str | None:
    """Qualify ambiguous unqualified column names using the first owning table/alias in scope."""
    if not sql or not schema:
        return None
    cat = SchemaCatalog(schema)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        root = build_scope(tree)
    except Exception:
        return None
    if root is None:
        return None

    target_col = None
    if error_msg:
        m = re.search(r"ambiguous column name:\s*([A-Za-z0-9_]+)", error_msg, flags=re.I)
        if m:
            target_col = m.group(1).lower()

    clone = copy.deepcopy(tree)
    try:
        croot = build_scope(clone)
    except Exception:
        return None
    if croot is None:
        return None

    changed = False
    for scope in croot.traverse():
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        # Ordered list of (alias_or_name, real_table_lower)
        ordered_sources: list[tuple[str, str]] = []
        from_expr = sel.args.get("from_") or sel.args.get("from")
        if from_expr and isinstance(from_expr.this, exp.Table):
            t = from_expr.this
            ordered_sources.append((t.alias_or_name, t.name.lower()))
        for j in sel.args.get("joins") or []:
            if isinstance(j.this, exp.Table):
                t = j.this
                ordered_sources.append((t.alias_or_name, t.name.lower()))
        if len(ordered_sources) < 2:
            continue

        # Collect SELECT aliases so we don't qualify an alias reference in ORDER BY / GROUP BY
        select_aliases = {
            e.alias.lower() for e in sel.expressions if isinstance(e, exp.Alias) and e.alias
        }

        for col in sel.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not sel:
                continue
            if col.table:
                continue
            cname = col.name.lower()
            if target_col and cname != target_col:
                continue
            if not target_col and cname in select_aliases:
                continue
            owners = [(al, rt) for al, rt in ordered_sources if cat.has_col(rt, cname)]
            if len(owners) >= 2:
                # Qualify with the first owning table/alias (primary FROM table wins)
                col.set("table", exp.to_identifier(owners[0][0]))
                changed = True

    if not changed:
        return None
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def prune_extra_columns(
    sql: str,
    example_cols: list[str],
    actual_cols: list[str] | None = None,
    example_rows: list[list] | None = None,
    actual_rows: list[list] | None = None,
) -> str | None:
    """Prune extra SELECT expressions when len(select.expressions) > len(example_cols)."""
    if not sql or not example_cols:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    exprs = list(tree.expressions)
    if len(exprs) <= len(example_cols):
        return None
    if any(isinstance(e, exp.Star) for e in exprs):
        return None

    act_names = actual_cols if (actual_cols and len(actual_cols) == len(exprs)) else [
        e.alias_or_name or e.sql(dialect="sqlite") for e in exprs
    ]

    # Build score matrix [actual_i][expected_j]
    n_act, n_exp = len(exprs), len(example_cols)
    act_col_vals = [
        [r[i] for r in (actual_rows or [])[:8] if i < len(r)] for i in range(n_act)
    ]
    exp_col_vals = [
        [r[j] for r in (example_rows or [])[:8] if j < len(r)] for j in range(n_exp)
    ]

    mat = [
        [
            _col_match_score(exprs[i], act_names[i], example_cols[j], act_col_vals[i], exp_col_vals[j])
            for j in range(n_exp)
        ]
        for i in range(n_act)
    ]

    # Find best injective assignment of n_exp indices from range(n_act)
    if n_act <= 7 and n_exp <= 5:
        best_perm = None
        best_score = -1e9
        for perm in itertools.permutations(range(n_act), n_exp):
            sc = sum(mat[perm[j]][j] for j in range(n_exp))
            # Small tie-breaker favoring original order
            if list(perm) == sorted(perm):
                sc += 0.1
            if sc > best_score:
                best_score = sc
                best_perm = perm
    else:
        used: set[int] = set()
        chosen: list[int] = []
        for j in range(n_exp):
            best_i = max((i for i in range(n_act) if i not in used), key=lambda i: mat[i][j])
            used.add(best_i)
            chosen.append(best_i)
        best_perm = tuple(chosen)

    if best_perm is None:
        return None

    kept_set = set(best_perm)
    clone = copy.deepcopy(tree)
    pruned_aliases: dict[str, exp.Expression] = {}
    for idx, e in enumerate(clone.expressions):
        if idx not in kept_set and isinstance(e, exp.Alias) and e.alias:
            pruned_aliases[e.alias.lower()] = copy.deepcopy(e.this)

    _inline_pruned_aliases(clone, pruned_aliases)
    clone.set("expressions", [copy.deepcopy(clone.expressions[i]) for i in best_perm])
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def reorder_columns(
    sql: str,
    example_cols: list[str],
    actual_cols: list[str] | None = None,
    example_rows: list[list] | None = None,
    actual_rows: list[list] | None = None,
) -> str | None:
    """Type-, name-, agg-, and value-aware column reordering when len(cols) == len(example_cols) >= 2."""
    if not sql or not example_cols or len(example_cols) < 2:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    exprs = list(tree.expressions)
    n = len(exprs)
    if n != len(example_cols) or n > 6:
        return None
    if any(isinstance(e, exp.Star) for e in exprs):
        return None

    act_names = actual_cols if (actual_cols and len(actual_cols) == n) else [
        e.alias_or_name or e.sql(dialect="sqlite") for e in exprs
    ]
    act_col_vals = [[r[i] for r in (actual_rows or [])[:8] if i < len(r)] for i in range(n)]
    exp_col_vals = [[r[j] for r in (example_rows or [])[:8] if j < len(r)] for j in range(n)]

    mat = [
        [
            _col_match_score(exprs[i], act_names[i], example_cols[j], act_col_vals[i], exp_col_vals[j])
            for j in range(n)
        ]
        for i in range(n)
    ]

    identity = tuple(range(n))
    id_score = sum(mat[i][i] for i in range(n))
    best_perm = identity
    best_score = id_score
    for perm in itertools.permutations(range(n)):
        sc = sum(mat[perm[j]][j] for j in range(n))
        if sc > best_score + 0.5:
            best_score = sc
            best_perm = perm

    if best_perm == identity:
        return None
    clone = copy.deepcopy(tree)
    clone.set("expressions", [copy.deepcopy(clone.expressions[i]) for i in best_perm])
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def complete_missing_columns(
    sql: str,
    schema: list[dict],
    example_cols: list[str],
    actual_cols: list[str] | None = None,
) -> str | None:
    """Insert missing SELECT column(s) from already-joined tables when len(cols) < len(example_cols)."""
    if not sql or not schema or not example_cols:
        return None
    cat = SchemaCatalog(schema)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    exprs = list(tree.expressions)
    if len(exprs) >= len(example_cols) or len(example_cols) - len(exprs) > 2:
        return None
    if any(isinstance(e, exp.Star) for e in exprs):
        return None

    # Gather already-joined tables in order: [(alias_or_name, real_table_lower)]
    sources: list[tuple[str, str]] = []
    from_expr = tree.args.get("from_") or tree.args.get("from")
    if from_expr and isinstance(from_expr.this, exp.Table):
        t = from_expr.this
        sources.append((t.alias_or_name, t.name.lower()))
    for j in tree.args.get("joins") or []:
        if isinstance(j.this, exp.Table):
            t = j.this
            sources.append((t.alias_or_name, t.name.lower()))
    if not sources:
        return None

    act_names = actual_cols if (actual_cols and len(actual_cols) == len(exprs)) else [
        e.alias_or_name or e.sql(dialect="sqlite") for e in exprs
    ]
    act_cores = [_normalize_col_label(a) for a in act_names]

    # Map existing expressions to their matching slots in example_cols
    slot_exprs: list[exp.Expression | None] = [None] * len(example_cols)
    used_act: set[int] = set()
    for j, ex_c in enumerate(example_cols):
        ex_agg, ex_core = _normalize_col_label(ex_c)
        for i, (a_agg, a_core) in enumerate(act_cores):
            if i in used_act:
                continue
            if a_agg == ex_agg and a_core == ex_core:
                slot_exprs[j] = copy.deepcopy(exprs[i])
                used_act.add(i)
                break

    if len(used_act) != len(exprs):
        return None  # Existing columns don't cleanly align; don't guess blindly

    has_group = tree.args.get("group") is not None
    # Fill missing slots
    for j, ex_c in enumerate(example_cols):
        if slot_exprs[j] is not None:
            continue
        ex_agg, ex_core = _normalize_col_label(ex_c)
        if ex_agg == "count" and ex_core in ("*", "1", ""):
            slot_exprs[j] = exp.Count(this=exp.Star())
            continue
        if ex_agg:
            return None  # Don't guess complex missing aggregates
        if has_group:
            return None  # Adding non-grouped column to GROUP BY query is risky
        owners = [(al, rt) for al, rt in sources if cat.has_col(rt, ex_core)]
        if not owners:
            return None
        # Prefer unique owner, or primary FROM table
        chosen_al, chosen_rt = owners[0]
        real_c = cat.real_col(chosen_rt, ex_core) or ex_core
        slot_exprs[j] = exp.column(real_c, table=chosen_al)

    if any(s is None for s in slot_exprs):
        return None
    clone = copy.deepcopy(tree)
    clone.set("expressions", slot_exprs)
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def left_to_inner_join(sql: str) -> str | None:
    """Convert LEFT [OUTER] JOIN -> INNER JOIN."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    changed = False
    for j in tree.find_all(exp.Join):
        if (j.side or "").upper() == "LEFT":
            j.set("side", None)
            j.set("kind", "INNER")
            changed = True
    if not changed:
        return None
    out = tree.sql(dialect="sqlite")
    return out if out and out != sql else None


def flip_order_direction(sql: str) -> str | None:
    """Flip top-level ORDER BY ASC <-> DESC."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    order = tree.args.get("order")
    if not order or not order.expressions:
        return None
    clone = copy.deepcopy(tree)
    for node in clone.args["order"].expressions:
        node.set("desc", not bool(node.args.get("desc")))
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def add_missing_group_by_for_mixed_agg(sql: str, expected_rc: int) -> str | None:
    """SQLite quirk fix: if top-level SELECT mixes an aggregate (COUNT/SUM/AVG/MIN/MAX)
    with >=1 bare non-aggregated column(s) AND has NO GROUP BY, SQLite silently collapses
    the result into 1 arbitrary row. When expected_rc != 1, GROUP BY <bare_cols> is
    mathematically required."""
    if expected_rc == 1 or not sql:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select) or tree.args.get("group") is not None:
        return None
    exprs = list(tree.expressions)
    if len(exprs) < 2:
        return None

    has_agg = False
    bare_cols: list[exp.Column] = []
    for e in exprs:
        unwrapped = e.this if isinstance(e, exp.Alias) else e
        if any(isinstance(n, exp.AggFunc) for n in unwrapped.walk()):
            has_agg = True
        elif isinstance(unwrapped, exp.Column):
            bare_cols.append(copy.deepcopy(unwrapped))
        else:
            return None

    if not has_agg or not bare_cols:
        return None
    clone = copy.deepcopy(tree)
    clone.set("group", exp.Group(expressions=bare_cols))
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def repair_alphabetical_id_order(sql: str, schema: list[dict], question: str) -> str | None:
    """If question explicitly asks for alphabetical order, yet ORDER BY sorts by a numeric/*_id
    column `t.col_id`, follow `t.col_id` (or its explicit FK target table) to a unique text
    column and sort by that text column instead."""
    if not sql or not schema or not re.search(r"\b(alphabetical|alphabetically|lexicographical)\b", (question or "").lower()):
        return None
    cat = SchemaCatalog(schema)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        root = build_scope(tree)
    except Exception:
        return None
    if not isinstance(tree, exp.Select) or root is None or not tree.args.get("order"):
        return None

    clone = copy.deepcopy(tree)
    croot = build_scope(clone)
    if croot is None:
        return None

    alias_map = {k.lower(): src.name.lower() for k, src in croot.sources.items() if isinstance(src, exp.Table)}
    taken = set(alias_map.keys()) | {t.alias_or_name.lower() for t in clone.find_all(exp.Table)}
    changed = False

    for ord_item in clone.args["order"].expressions:
        col = ord_item.this if isinstance(ord_item.this, exp.Column) else None
        if col is None:
            continue
        t, c = resolve_column(col, alias_map, cat)
        if not (t and c):
            continue
        if not (cat.type_family(t, c) == "num" or c == "id" or c.endswith("_id")):
            continue

        # Check if (t, c) is an explicit FK to a target table (e.g. Supplier_Addresses.address_id -> Addresses.address_id)
        fk_targets = [(rt, rc) for (lt, lc, rt, rc) in cat.fk_edges if lt == t and lc == c and rt != t]
        target_tbl = fk_targets[0][0] if len(fk_targets) == 1 else t
        fk_ref_col = fk_targets[0][1] if len(fk_targets) == 1 else None

        # Find non-PK text columns on target_tbl
        tinfo = cat.tables.get(target_tbl)
        if not tinfo:
            continue
        text_cols = [
            ci["name"]
            for cl, ci in tinfo["cols"].items()
            if not ci["pk"] and cat.is_text_col(target_tbl, cl) and not cl.endswith("_id") and not cl.endswith("_code")
        ]
        if len(text_cols) != 1:
            continue
        chosen_text_col = text_cols[0]

        # Ensure target_tbl is joined in clone
        existing_qual = next((al for al, rt in alias_map.items() if rt == target_tbl), None)
        if existing_qual is None and fk_ref_col is not None:
            new_al = f"a_{len(taken) + 1}"
            while new_al in taken:
                new_al += "x"
            taken.add(new_al)
            src_qual = col.table or cat.real_table(t)
            joins = list(clone.args.get("joins") or [])
            joins.append(
                exp.Join(
                    this=exp.Table(
                        this=exp.to_identifier(cat.real_table(target_tbl) or target_tbl),
                        alias=exp.TableAlias(this=exp.to_identifier(new_al)),
                    ),
                    kind="INNER",
                    on=exp.EQ(
                        this=exp.column(cat.real_col(t, c) or c, table=src_qual),
                        expression=exp.column(cat.real_col(target_tbl, fk_ref_col) or fk_ref_col, table=new_al),
                    ),
                )
            )
            clone.set("joins", joins)
            alias_map[new_al] = target_tbl
            existing_qual = new_al

        if existing_qual:
            ord_item.set("this", exp.column(chosen_text_col, table=existing_qual))
            changed = True

    if not changed:
        return None
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def toggle_distinct(sql: str, add_distinct: bool) -> str | None:
    """Add or remove top-level SELECT DISTINCT."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    has_d = bool(tree.args.get("distinct"))
    if add_distinct == has_d:
        return None
    clone = copy.deepcopy(tree)
    if add_distinct:
        clone.set("distinct", exp.Distinct())
    else:
        clone.set("distinct", None)
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def repair_bogus_join_via_fk_bridge(sql: str, schema: list[dict]) -> str | None:
    """If a query has a suspicious non-FK direct join between t1 and t2, and the schema has
    a unique 2-hop FK bridge table B connecting t1 <-> B <-> t2, rewrite that join via B."""
    if not sql or not schema:
        return None
    cat = SchemaCatalog(schema)
    if not cat.fk_edges:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        root = build_scope(tree)
    except Exception:
        return None
    if root is None:
        return None

    clone = copy.deepcopy(tree)
    try:
        croot = build_scope(clone)
    except Exception:
        return None
    if croot is None:
        return None

    taken = {t.alias_or_name.lower() for t in clone.find_all(exp.Table)}
    changed = False

    for scope in croot.traverse():
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        alias_map = {}
        alias_to_real_display = {}
        for k, src in scope.sources.items():
            if isinstance(src, exp.Table):
                alias_map[k.lower()] = src.name.lower()
                alias_map[src.name.lower()] = src.name.lower()
                alias_to_real_display[k.lower()] = k

        joins = list(sel.args.get("joins") or [])
        if not joins:
            continue
        new_joins: list[exp.Join] = []
        for j in joins:
            on = j.args.get("on")
            rewritten = False
            if on is not None and isinstance(on, exp.EQ) and isinstance(on.this, exp.Column) and isinstance(on.expression, exp.Column):
                t1, c1 = resolve_column(on.this, alias_map, cat)
                t2, c2 = resolve_column(on.expression, alias_map, cat)
                if t1 and t2 and t1 != t2 and not cat.is_implicit_clean_join(t1, c1, t2, c2):
                    # Find 2-hop FK bridges B such that FK(t1, B) and FK(B, t2)
                    bridges = []
                    for mid in cat.fk_adj.get(t1, set()):
                        if mid == t2 or mid in alias_map.values():
                            continue
                        if t2 in cat.fk_adj.get(mid, set()):
                            # Extract exact FK edge columns
                            e1 = [e for e in cat.fk_edges if e[0] == t1 and e[2] == mid]
                            e2 = [e for e in cat.fk_edges if e[0] == mid and e[2] == t2]
                            if len(e1) == 1 and len(e2) == 1:
                                bridges.append((mid, e1[0], e2[0]))
                    if len(bridges) == 1:
                        mid, e1, e2 = bridges[0]
                        # Figure out which side is the newly joined table j.this
                        j_tbl = j.this.name.lower() if isinstance(j.this, exp.Table) else ""
                        if j_tbl == t2:
                            q_prev = on.this.table or cat.real_table(t1)
                            q_curr = j.this.alias_or_name
                            prev_col, mid_col_a = cat.real_col(t1, e1[1]), cat.real_col(mid, e1[3])
                            mid_col_b, curr_col = cat.real_col(mid, e2[1]), cat.real_col(t2, e2[3])
                        elif j_tbl == t1:
                            q_prev = on.expression.table or cat.real_table(t2)
                            q_curr = j.this.alias_or_name
                            prev_col, mid_col_a = cat.real_col(t2, e2[3]), cat.real_col(mid, e2[1])
                            mid_col_b, curr_col = cat.real_col(mid, e1[3]), cat.real_col(t1, e1[1])
                        else:
                            q_prev = None
                        if q_prev and prev_col and mid_col_a and mid_col_b and curr_col:
                            b_alias = f"b_{len(taken) + 1}"
                            while b_alias in taken:
                                b_alias += "x"
                            taken.add(b_alias)
                            bridge_join = exp.Join(
                                this=exp.Table(
                                    this=exp.to_identifier(cat.real_table(mid) or mid),
                                    alias=exp.TableAlias(this=exp.to_identifier(b_alias)),
                                ),
                                kind="INNER",
                                on=exp.EQ(
                                    this=exp.column(prev_col, table=q_prev),
                                    expression=exp.column(mid_col_a, table=b_alias),
                                ),
                            )
                            j_clone = copy.deepcopy(j)
                            j_clone.set(
                                "on",
                                exp.EQ(
                                    this=exp.column(mid_col_b, table=b_alias),
                                    expression=exp.column(curr_col, table=q_curr),
                                ),
                            )
                            new_joins.append(bridge_join)
                            new_joins.append(j_clone)
                            changed = True
                            rewritten = True
            if not rewritten:
                new_joins.append(j)
        if changed:
            sel.set("joins", new_joins)

    if not changed:
        return None
    out = clone.sql(dialect="sqlite")
    return out if out and out != sql else None


def generate_variants(
    sql: str,
    schema: list[dict] | None,
    example: dict,
    feedback: dict | None,
    question: str,
    ordered: bool = False,
) -> list[tuple[str, str]]:
    """Generate prioritized deterministic SQL repairs for a failing (or suspicious) candidate."""
    if not sql:
        return []
    ex_cols = list(example.get("columns") or [])
    ex_rows = list(example.get("rows") or [])
    ex_rc = int(example.get("row_count") or 0)
    fb = feedback or {}
    ok = bool(fb.get("ok"))
    act_cols = list(fb.get("columns") or []) if ok else None
    act_rows = list(fb.get("rows") or []) if ok else None
    act_rc = int(fb.get("row_count") or 0) if ok else 0
    err_msg = ""
    if not ok and isinstance(fb.get("error"), dict):
        err_msg = str(fb["error"].get("message") or "")
    elif not ok and fb.get("error"):
        err_msg = str(fb["error"])

    out: list[tuple[str, str]] = []
    seen = {sql.strip()}

    def _add(tag: str, cand: str | None) -> None:
        if cand and cand.strip() and cand.strip() not in seen:
            seen.add(cand.strip())
            out.append((tag, cand.strip()))

    # 1. Ambiguous column error -> disambiguate + optionally also prune/reorder in one step!
    if not ok and schema and "ambiguous column" in err_msg.lower():
        dis = disambiguate_columns(sql, schema, err_msg)
        if dis:
            # Check if column count also exceeds expected (e.g. SELECT artist_name, COUNT(*) vs ['artist_name'])
            pruned = prune_extra_columns(dis, ex_cols)
            if pruned:
                _add("disambiguate+prune", pruned)
            else:
                _add("disambiguate", dis)

    # 2. Bogus non-FK join -> FK bridge repair (works even when ok=True & ex_rc=0!)
    if schema:
        bridged = repair_bogus_join_via_fk_bridge(sql, schema)
        _add("fk_bridge", bridged)

    # Projection/row fixes only make sense on queries that actually executed (ok=True);
    # running prune/reorder on a query that threw an execution error just repeats the error.
    if not ok:
        return out

    # 3. Projection pruning when too many columns
    if ex_cols:
        pruned = prune_extra_columns(sql, ex_cols, act_cols, ex_rows, act_rows)
        _add("prune_cols", pruned)

    # 4. Projection completion when too few columns
    if schema and ex_cols:
        comp = complete_missing_columns(sql, schema, ex_cols, act_cols)
        _add("complete_cols", comp)

    # 5. Smart column reordering when same column count >= 2
    if ex_cols and len(ex_cols) >= 2:
        reord = reorder_columns(sql, ex_cols, act_cols, ex_rows, act_rows)
        _add("reorder_cols", reord)

    # 6. LEFT JOIN -> INNER JOIN when actual returned more rows than expected
    if ok and act_rc > ex_rc:
        inner = left_to_inner_join(sql)
        _add("left_to_inner", inner)

    # 7. DISTINCT toggle when actual row count mismatches expected
    if ok and act_rc > ex_rc and ex_rc > 0:
        _add("add_distinct", toggle_distinct(sql, True))
    elif ok and 0 < act_rc < ex_rc and not re.search(r"\b(distinct|different|unique)\b", (question or "").lower()):
        _add("drop_distinct", toggle_distinct(sql, False))

    # 8. Flip ORDER BY ASC/DESC when ordered or LIMIT query got wrong result
    if ok and not fb.get("matches_example"):
        _add("flip_order", flip_order_direction(sql))

    return out
