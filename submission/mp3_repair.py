"""Schema 感知的 SQL 修复引擎（MP3 v1）。

只做一件确定性的事：把草稿里"数据库里不存在"的标识符，锚定到真实的表/列上，
必要时沿着外键图补 JOIN。不做语义改写（不改 DISTINCT、不改聚合、不改过滤条件）。

用法：
    from mp3_repair import repair
    fixed = repair(sql, schema)      # schema = manifest["databases"][db_id]["schema"]
    if fixed is None:                # 解析失败 / 无需修改 / 缺少可推断的修复
        ...

对外只有一个函数 repair()；其余以下划线开头的都是内部实现。
"""
from __future__ import annotations

import difflib
import re
from collections import deque

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

MAX_JOINS = 3
FUZZY_TABLE = 0.72
FUZZY_COLUMN = 0.78
TOKEN_COLUMN = 0.58
STOPWORDS = {"of", "the", "a", "an", "in", "on", "by", "for", "and", "or"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


class Catalog:
    """把一个 db 的 schema 索引成便于查找的结构。"""

    def __init__(self, schema: list[dict]):
        self.tables: dict[str, dict] = {}
        for table in schema:
            columns = {c["name"].lower(): c["name"] for c in table.get("columns", [])}
            fks = []
            for fk in table.get("foreign_keys", []) or []:
                fks.append((fk["from"].lower(), fk["table"].lower(), fk["to"].lower()))
            self.tables[table["name"].lower()] = {
                "name": table["name"],
                "columns": columns,
                "fks": fks,
            }
        self.column_index: dict[str, list[str]] = {}
        for name, info in self.tables.items():
            for column in info["columns"]:
                self.column_index.setdefault(column, []).append(name)

    def real(self, name: str) -> str | None:
        info = self.tables.get(name.lower())
        return info["name"] if info else None

    def has_column(self, table: str, column: str) -> bool:
        info = self.tables.get(table.lower())
        return bool(info) and column.lower() in info["columns"]

    def column(self, table: str, column: str) -> str | None:
        info = self.tables.get(table.lower())
        return info["columns"].get(column.lower()) if info else None

    def owners(self, column: str) -> list[str]:
        return list(self.column_index.get(column.lower(), []))

    def best_table(self, name: str) -> str | None:
        if not self.tables:
            return None
        matches = difflib.get_close_matches(name.lower(), list(self.tables), n=2,
                                            cutoff=FUZZY_TABLE)
        return matches[0] if len(matches) == 1 else None

    def best_column(self, column: str) -> str | None:
        matches = difflib.get_close_matches(column.lower(), list(self.column_index), n=2,
                                           cutoff=FUZZY_COLUMN)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _score(left: str, right: str) -> float:
        ratio = difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()
        a, b = _tokens(left), _tokens(right)
        overlap = len(a & b) / max(1, len(a | b))
        return 0.6 * ratio + 0.4 * overlap

    def match_column(self, name: str, table: str | None = None) -> tuple[str, str] | None:
        """按词形/词元相似度找最可能的真实列；不唯一就放弃。"""
        if table and table.lower() in self.tables:
            pool = {table.lower(): self.tables[table.lower()]}
        else:
            pool = self.tables
        scored = []
        for table_name, info in pool.items():
            for column in info["columns"]:
                score = self._score(name, column)
                if score >= TOKEN_COLUMN:
                    scored.append((score, table_name, info["columns"][column]))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
            return None
        return scored[0][1], scored[0][2]


def _sources(scope) -> dict[str, str]:
    """scope 里 alias/表名 -> 真实表名（只保留真实表，跳过子查询）。"""
    result = {}
    for key, source in scope.sources.items():
        if isinstance(source, exp.Table):
            result[key.lower()] = source.name
    return result


def _ancestor_sources(scope) -> dict[str, str]:
    merged, node = {}, scope.parent
    while node is not None and len(merged) < 16:
        merged.update(_sources(node))
        node = node.parent
    return merged


def _path(cat: Catalog, starts: list[str], target: str):
    """在无向外键图上找 starts -> target 的最短路径，返回边列表。"""
    target = target.lower()
    queue = deque()
    seen: dict[str, tuple[str, tuple] | None] = {}
    for start in starts:
        if start.lower() in cat.tables:
            seen[start.lower()] = None
            queue.append(start.lower())
    if not seen:
        return None
    while queue:
        node = queue.popleft()
        if node == target:
            edges, cursor = [], node
            while seen[cursor] is not None:
                parent, edge = seen[cursor]
                edges.append(edge)
                cursor = parent
            return list(reversed(edges))
        info = cat.tables.get(node)
        if not info:
            continue
        for column, ref_table, ref_column in info["fks"]:
            if ref_table in cat.tables and ref_table not in seen:
                seen[ref_table] = (node, (node, column, ref_table, ref_column))
                queue.append(ref_table)
        for other_name, other in cat.tables.items():
            if other_name in seen:
                continue
            for column, ref_table, ref_column in other["fks"]:
                if ref_table == node:
                    seen[other_name] = (node, (other_name, column, node, ref_column))
                    queue.append(other_name)
                    break
    return None


def _shared_join(select: exp.Select, cat: Catalog, scope, target: str,
                 state: dict) -> str | None:
    """没有外键元数据时的兜底：用两表同名列做等值连接。

    只在"公共列里存在唯一强候选（*_id 或 id）"时动手，避免瞎连。
    """
    local = _sources(scope)
    target_info = cat.tables.get(target.lower())
    if not target_info:
        return None
    candidates = []
    for local_key, real in local.items():
        info = cat.tables.get(real.lower())
        if not info:
            continue
        for column in set(info["columns"]) & set(target_info["columns"]):
            strength = 2 if column.endswith("_id") else 1 if column == "id" else 0
            candidates.append((strength, local_key, real, column))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[3]))
    if candidates[0][0] == 0:
        return None
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0] and candidates[0][3] != candidates[1][3]:
        return None
    strength, local_key, real, column = candidates[0]
    alias = next(state["aliases"])
    joined_map = state["joined"].setdefault(id(select), {})
    joined_map[target.lower()] = alias
    joins = list(select.args.get("joins") or [])
    joins.append(exp.Join(
        this=exp.Table(this=exp.to_identifier(cat.real(target) or target),
                       alias=exp.TableAlias(this=exp.to_identifier(alias))),
        kind="INNER",
        on=exp.EQ(this=exp.column(cat.column(real, column) or column, local_key),
                  expression=exp.column(cat.column(target, column) or column, alias))))
    select.set("joins", joins)
    return alias


def _add_join(select: exp.Select, cat: Catalog, scope, target: str, state: dict) -> str | None:
    """把 target 表接进 select，返回它的限定名（别名或表名）。"""
    key = target.lower()
    joined_map = state["joined"].setdefault(id(select), {})
    if key in joined_map:
        return joined_map[key]
    local = _sources(scope)
    for local_key, real in local.items():
        if real.lower() == key:
            return local_key
    if not local:
        return None
    path = _path(cat, list(local.values()), target)
    if not path:
        return _shared_join(select, cat, scope, target, state)

    qualifier = {real.lower(): local_key for local_key, real in local.items()}
    joins = list(select.args.get("joins") or [])
    added: dict[str, str] = dict(joined_map)

    def qualifier_of(name: str) -> str | None:
        return qualifier.get(name.lower()) or added.get(name.lower())

    for left, left_column, right, right_column in path:
        left_q, right_q = qualifier_of(left), qualifier_of(right)
        if left_q and right_q:
            continue
        if left_q is None and right_q is None:
            return None
        if left_q is None:
            left_q = next(state["aliases"])
            added[left.lower()] = left_q
            table_expr = exp.Table(this=exp.to_identifier(cat.real(left) or left),
                                   alias=exp.TableAlias(this=exp.to_identifier(left_q)))
            condition = exp.EQ(this=exp.column(cat.column(left, left_column) or left_column,
                                               left_q),
                               expression=exp.column(cat.column(right, right_column)
                                                     or right_column, right_q))
        else:
            right_q = next(state["aliases"])
            added[right.lower()] = right_q
            table_expr = exp.Table(this=exp.to_identifier(cat.real(right) or right),
                                   alias=exp.TableAlias(this=exp.to_identifier(right_q)))
            condition = exp.EQ(this=exp.column(cat.column(left, left_column) or left_column,
                                               left_q),
                               expression=exp.column(cat.column(right, right_column)
                                                     or right_column, right_q))
        joins.append(exp.Join(this=table_expr, kind="INNER", on=condition))

    select.set("joins", joins)
    joined_map.update(added)
    return qualifier_of(target)


def _nearest(cat: Catalog, local: dict[str, str], owners: list[str]) -> str:
    """多个表都可能有这一列时，选外键图上离当前 FROM 最近的。"""
    best, best_len = owners[0], 99
    for owner in owners:
        path = _path(cat, list(local.values()), owner)
        length = len(path) if path else 98
        if length < best_len:
            best, best_len = owner, length
    return best


def _fix_column(column: exp.Column, cat: Catalog, scope, state: dict) -> None:
    local = _sources(scope)
    outer = _ancestor_sources(scope)
    select = column.find_ancestor(exp.Select)
    name, qualifier = column.name, column.table
    scoped = qualifier.lower() in local if qualifier else False

    if qualifier:
        key = qualifier.lower()
        if scoped:
            real = local[key]
            if cat.has_column(real, name):
                return
            inside = cat.match_column(name, real)
            if inside:
                column.set("this", exp.to_identifier(inside[1]))
                state["changed"] = True
                return
            alternatives = [k for k, r in local.items() if cat.has_column(r, name)]
            if len(alternatives) == 1:
                alt = alternatives[0]
                # Never create a tautological self-equality `alt.col = alt.col` inside an EQ!
                if isinstance(column.parent, exp.EQ):
                    sibling = column.parent.expression if column.parent.this is column else column.parent.this
                    if isinstance(sibling, exp.Column) and (sibling.table or "").lower() == alt.lower():
                        pass
                    else:
                        column.set("table", exp.to_identifier(alt))
                        state["changed"] = True
                        return
                else:
                    column.set("table", exp.to_identifier(alt))
                    state["changed"] = True
                    return
        elif key in outer:
            return
        else:
            renamed = [k for k, r in local.items() if r.lower() == key]
            if renamed:
                column.set("table", exp.to_identifier(renamed[0]))
                state["changed"] = True
                return
            if key not in cat.tables:
                guess = cat.best_table(key)
                if guess:
                    column.set("table", exp.to_identifier(guess))
                    state["changed"] = True
                    return
    else:
        if any(cat.has_column(real, name) for real in local.values()):
            return
        if any(cat.has_column(real, name) for real in outer.values()):
            return

    owners = cat.owners(name)
    real_name = name
    if not owners:
        hit = cat.match_column(name)
        if not hit:
            return
        owners, real_name = [hit[0]], hit[1]
    if scoped and not owners:
        return

    target = owners[0] if len(owners) == 1 else _nearest(cat, local, owners)
    alias = _add_join(select, cat, scope, target, state)
    if not alias:
        return
    column.set("table", exp.to_identifier(alias))
    resolved = cat.column(target, real_name)
    if resolved:
        column.set("this", exp.to_identifier(resolved))
    state["changed"] = True


def repair(sql: str, schema: list[dict]) -> str | None:
    """返回修复后的 SQL；无法解析、无需修改或无从推断时返回 None。"""
    if not sql or not schema:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return None

    cat = Catalog(schema)
    try:
        root = build_scope(tree)
    except Exception:
        return None
    if root is None:
        return None

    scopes = list(root.traverse())
    scope_by_select = {id(scope.expression): scope for scope in scopes}
    taken = {key for scope in scopes for key in _sources(scope)}
    taken |= {table.alias_or_name.lower() for table in tree.find_all(exp.Table)}

    def alias_gen():
        index = 0
        while True:
            index += 1
            if f"j{index}" not in taken:
                taken.add(f"j{index}")
                yield f"j{index}"

    state = {"changed": False, "joined": {}, "aliases": alias_gen()}

    for table in tree.find_all(exp.Table):
        if table.name.lower() in cat.tables:
            continue
        guess = cat.best_table(table.name)
        if guess:
            table.set("this", exp.to_identifier(cat.real(guess) or guess))
            state["changed"] = True

    # Pre-resolve undeclared table aliases per Select (e.g. WHERE T2.address_id = T3.address_id AND T3.state = 'CA')
    for scope in scopes:
        sel = scope.expression
        if not isinstance(sel, exp.Select):
            continue
        local = _sources(scope)
        outer = _ancestor_sources(scope)
        undeclared_cols: dict[str, set[str]] = {}
        undeclared_nodes: dict[str, list[exp.Column]] = {}
        for col in sel.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not sel or not col.table:
                continue
            q = col.table.lower()
            if q in local or q in outer or q in cat.tables or any(r.lower() == q for r in local.values()):
                continue
            undeclared_cols.setdefault(q, set()).add(col.name.lower())
            undeclared_nodes.setdefault(q, []).append(col)
        for q, req_cols in undeclared_cols.items():
            # Find schema tables that own ALL columns referenced under undeclared alias q
            candidate_tables = [
                tname for tname, tinfo in cat.tables.items()
                if req_cols.issubset(set(tinfo["columns"].keys()))
                and tname not in {r.lower() for r in local.values()}
            ]
            if len(candidate_tables) == 1:
                target = candidate_tables[0]
                alias = _add_join(sel, cat, scope, target, state)
                if alias:
                    for col_node in undeclared_nodes[q]:
                        col_node.set("table", exp.to_identifier(alias))
                        resolved = cat.column(target, col_node.name)
                        if resolved:
                            col_node.set("this", exp.to_identifier(resolved))
                    state["changed"] = True

    for column in list(tree.find_all(exp.Column)):
        # Skip SQLite double-quoted string literals like WHERE state = "California"
        if not column.table and getattr(column.this, "quoted", False):
            if not cat.owners(column.name):
                continue
        select = column.find_ancestor(exp.Select)
        if select is None:
            continue
        scope = scope_by_select.get(id(select))
        if scope is None:
            continue
        try:
            _fix_column(column, cat, scope, state)
        except Exception:
            continue

    if not state["changed"]:
        return None
    try:
        out = tree.sql(dialect="sqlite")
    except Exception:
        return None
    return out if out and out != sql else None
