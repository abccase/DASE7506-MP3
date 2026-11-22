"""Lightweight pure-Python False-Positive Risk Scorer & Multi-Candidate Ranker.

Weights are fitted strictly on `train` (480 tasks, 118 DBs) via DB-grouped CV
and stored in `reranker_weights.json` (< 4 KiB, zero extra pip dependencies).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from static_audit import SchemaCatalog, audit_sql


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "reranker_weights.json"

FEATURE_NAMES = [
    "ex_rows_0",
    "ex_rows_1",
    "is_draft",
    "is_hint",
    "is_repaired_or_llm",
    "suspicious_non_fk_joins",
    "cross_type_joins",
    "ungrounded_literals",
    "literal_has_similar_in_col",
    "literal_found_in_other_col",
    "all_literals_grounded",
    "ordered_missing_orderby",
    "alphabetical_id_order",
    "extra_distinct_no_cue",
    "missing_distinct_with_cue",
    "negation_uses_neq_on_join",
    "col_align_score",
    "hint_vote_frac",
    "matches_any_hint_ast",
    "table_count_norm",
    "has_empty_ref_table",
    "has_left_join",
]


def canonical_sql_key(sql: str) -> str:
    """Normalize SQL AST for structural consensus voting across hints."""
    if not sql:
        return ""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        # Strip table aliases to canonicalize cosmetic alias differences
        return re.sub(r"\s+", " ", tree.sql(dialect="sqlite")).strip().lower()
    except Exception:
        return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()


def extract_reranker_features(
    *,
    sql: str,
    source: str,
    cat: SchemaCatalog | None,
    question: str,
    ordered: bool,
    example: dict,
    feedback: dict | None,
    probe_report: dict | None,
    hint_sqls: list[str] | None = None,
) -> tuple[dict[str, float], dict]:
    """Extract normalized numeric feature dict and full audit result for a candidate."""
    ex_cols = list(example.get("columns") or [])
    ex_rc = int(example.get("row_count") or 0)
    act_cols = list((feedback or {}).get("columns") or []) if (feedback or {}).get("ok") else None

    audit = audit_sql(
        sql=sql,
        cat=cat,
        question=question,
        ordered=ordered,
        example_cols=ex_cols,
        actual_cols=act_cols,
        probe_report=probe_report,
    )
    af = audit["features"]

    # Hint consensus features
    cand_key = canonical_sql_key(sql)
    h_keys = [canonical_sql_key(h) for h in (hint_sqls or []) if isinstance(h, str) and h.strip()]
    if h_keys:
        votes = sum(1 for hk in h_keys if hk == cand_key)
        vote_frac = votes / len(h_keys)
        matches_hint = 1.0 if votes > 0 else 0.0
    else:
        vote_frac = 0.0
        matches_hint = 0.0

    # Check if any table referenced in SQL has 0 rows in public DB (from db_probe)
    has_empty_tbl = 0.0
    if probe_report and probe_report.get("ok") and probe_report.get("table_counts"):
        tcounts_lc = {k.lower(): v for k, v in probe_report["table_counts"].items()}
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            for t in tree.find_all(exp.Table):
                if t.name and tcounts_lc.get(t.name.lower(), 1) == 0:
                    has_empty_tbl = 1.0
                    break
        except Exception:
            pass

    src_low = (source or "").lower()
    feats = {
        "ex_rows_0": 1.0 if ex_rc == 0 else 0.0,
        "ex_rows_1": 1.0 if ex_rc == 1 else 0.0,
        "is_draft": 1.0 if src_low == "draft" else 0.0,
        "is_hint": 1.0 if src_low.startswith("hint") or src_low.startswith("h") else 0.0,
        "is_repaired_or_llm": 1.0 if (src_low.startswith("llm") or src_low.startswith("repair") or src_low.startswith("var") or src_low.startswith("lit")) else 0.0,
        "suspicious_non_fk_joins": float(min(af["suspicious_non_fk_joins"], 3)),
        "cross_type_joins": float(min(af["cross_type_joins"], 2)),
        "ungrounded_literals": 1.0 if af["ungrounded_literals"] > 0 else 0.0,
        "literal_has_similar_in_col": 1.0 if af["literal_has_similar_in_col"] > 0 else 0.0,
        "literal_found_in_other_col": 1.0 if af["literal_found_in_other_col"] > 0 else 0.0,
        "all_literals_grounded": 1.0 if af["all_literals_grounded"] else 0.0,
        "ordered_missing_orderby": 1.0 if af["ordered_missing_orderby"] else 0.0,
        "alphabetical_id_order": 1.0 if af["alphabetical_id_order"] else 0.0,
        "extra_distinct_no_cue": 1.0 if af["extra_distinct_no_cue"] else 0.0,
        "missing_distinct_with_cue": 1.0 if af["missing_distinct_with_cue"] else 0.0,
        "negation_uses_neq_on_join": 1.0 if af["negation_uses_neq_on_join"] else 0.0,
        "col_align_score": float(af["col_align_score"]),
        "hint_vote_frac": float(vote_frac),
        "matches_any_hint_ast": float(matches_hint),
        "table_count_norm": float(min(af["table_count"], 6)) / 6.0,
        "has_empty_ref_table": float(has_empty_tbl),
        "has_left_join": 1.0 if af["has_left_join"] else 0.0,
    }
    return feats, audit


class FPReranker:
    """Logistic model predicting P(all3_correct | public_pass, features)."""

    def __init__(self, weights_path: Path | str | None = None):
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.bias = float(data.get("bias", 1.5))
            self.weights = {k: float(v) for k, v in (data.get("weights") or {}).items()}
        else:
            # Sensible domain defaults before fitting
            self.bias = 1.8
            self.weights = {
                "ex_rows_0": -1.2,
                "ex_rows_1": -0.4,
                "is_draft": -0.8,
                "is_hint": 0.4,
                "is_repaired_or_llm": 0.5,
                "suspicious_non_fk_joins": -2.5,
                "cross_type_joins": -1.2,
                "ungrounded_literals": -1.0,
                "literal_has_similar_in_col": -3.0,
                "literal_found_in_other_col": -2.5,
                "all_literals_grounded": 1.2,
                "ordered_missing_orderby": -1.0,
                "alphabetical_id_order": -2.0,
                "extra_distinct_no_cue": -0.3,
                "missing_distinct_with_cue": -0.4,
                "negation_uses_neq_on_join": -1.5,
                "col_align_score": 0.8,
                "hint_vote_frac": 0.8,
                "matches_any_hint_ast": 0.3,
                "table_count_norm": -0.2,
                "has_empty_ref_table": -0.5,
                "has_left_join": -0.3,
            }

    def predict_true_prob(self, feats: dict[str, float]) -> float:
        z = self.bias
        for k, w in self.weights.items():
            z += w * float(feats.get(k, 0.0))
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fp_risk(self, feats: dict[str, float]) -> float:
        return 1.0 - self.predict_true_prob(feats)

    def has_hard_red_flag(self, feats: dict[str, float]) -> bool:
        """Deterministic structural/probe red flags that ALWAYS trigger the FP Firewall.

        Note: a plain ungrounded literal on a 1-table unanimous-hint 0-row query (where the
        filter value simply isn't in the small public DB and has no similar/other-col match)
        is scored via `fp_risk` rather than treated as an unconditional hard defect.
        """
        return bool(
            feats.get("suspicious_non_fk_joins", 0) > 0
            or feats.get("literal_has_similar_in_col", 0) > 0
            or feats.get("literal_found_in_other_col", 0) > 0
            or feats.get("alphabetical_id_order", 0) > 0
            or feats.get("ordered_missing_orderby", 0) > 0
            or feats.get("negation_uses_neq_on_join", 0) > 0
        )

    def score_passing_candidate(self, cand: dict[str, Any]) -> tuple:
        """Sorting key for public-passing candidates (higher is better)."""
        feats = cand.get("features") or {}
        p_true = self.predict_true_prob(feats)
        # Hard penalty for deterministic red flags (non-FK join, similar literal mismatch, alpha-id order)
        hard_pen = (
            -3.0 * feats.get("suspicious_non_fk_joins", 0.0)
            - 3.0 * feats.get("literal_has_similar_in_col", 0.0)
            - 2.5 * feats.get("literal_found_in_other_col", 0.0)
            - 2.0 * feats.get("alphabetical_id_order", 0.0)
        )
        # Bonus for explicit verified repair / llm_fp_audit candidate when it fixed a red flag
        bonus = float(cand.get("bonus", 0.0))
        return (
            round(p_true + hard_pen + bonus, 6),
            1.0 if feats.get("all_literals_grounded", 0.0) > 0 else 0.0,
            feats.get("col_align_score", 0.0),
            -int(cand.get("order_idx", 0)),
        )

    def pick_best_passing(self, passing_pool: list[dict[str, Any]]) -> dict[str, Any]:
        return max(passing_pool, key=self.score_passing_candidate)

    @staticmethod
    def pick_best_fallback(all_pool: list[dict[str, Any]], example: dict) -> dict[str, Any]:
        """Fallback selector when NO candidate matched the public example."""
        ex_rc = int(example.get("row_count") or 0)

        def _key(c: dict[str, Any]) -> tuple:
            fb = c.get("feedback") or {}
            feats = c.get("features") or {}
            ok = 1 if fb.get("ok") else 0
            col_ok = 1 if (ok and feats.get("col_align_score", -1.0) >= 0.0) else 0
            fk_ok = 1 if feats.get("suspicious_non_fk_joins", 0.0) == 0 else 0
            rc_diff = -abs(int(fb.get("row_count") or 0) - ex_rc) if ok else -99999
            src = (c.get("source") or "").lower()
            src_prio = 3 if src.startswith("llm") else (2 if src.startswith("hint") or src.startswith("var") or src.startswith("lit") else 1)
            return (ok, col_ok, fk_ok, feats.get("col_align_score", -1.0), rc_diff, src_prio, -int(c.get("order_idx", 0)))

        return max(all_pool, key=_key)
