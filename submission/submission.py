"""Upgraded Perceive-Plan-Act SQL Repair Agent Harness (v9-r2-harness).

Round 2 Capabilities:
  1. Static schema/FK join validator, SQL semantics auditor & local pre-check
     executable validator (`static_audit.py`).
  2. General deterministic projection/hygiene/FK-bridge variants (`proj_variants.py`),
     including zero-row passing column-order alignment (`+3/-0` on train-480) and
     wrong-column-count hint pre-pruning.
  3. SetOperation-aware (`UNION`/`EXCEPT`/`INTERSECT`) & undeclared-alias schema
     anchoring (`mp3_repair.py`).
  4. Smart 1-call `execute` DB value prober & deterministic literal repairer (`db_probe.py`).
  5. Train-fitted False-Positive Firewall & multi-candidate ranker (`reranker.py`).
  6. Test-Time Scaling (`llm_runner.py`):
     - Pass 1 greedy (`n=1`) + local schema pre-validation & autofix before `check`.
     - Pass 2 True 1x-prefill KV-cache-shared diverse Best-of-N (`n<=4`, `<=6` total
       completions/task) on Public-Fail tasks (SQL error, wrong result, or duplicate),
       ranked by self-consistency AST votes + schema/column alignment.
     - Broader calibrated `fp_audit` coverage on `ex_rows == 0` & risky `ex_rows == 1`,
       with `Bo3` diversity escalation when greedy rubber-stamps a `hard_flag` defect
       and `Bo2` self-consistency guard on soft rewrites to prevent regressions.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import sqlglot
from sqlglot import exp

import llm_runner
from db_probe import (
    build_probe_query,
    parse_probe_result,
    repair_literals_from_probe,
)
from proj_variants import (
    add_missing_group_by_for_mixed_agg,
    disambiguate_columns,
    generate_variants,
    prune_extra_columns,
    reorder_columns,
    repair_alphabetical_id_order,
    repair_bogus_join_via_fk_bridge,
)
from prompt_format import build_messages
from reranker import (
    FPReranker,
    canonical_sql_key,
    extract_reranker_features,
)
from static_audit import (
    SchemaCatalog,
    audit_sql,
    column_alignment_score,
    extract_sql_predicates,
    validate_executable_locally,
)

try:
    from mp3_repair import repair as schema_anchor_repair
except Exception:
    schema_anchor_repair = None


DEFAULT_CONFIG: dict[str, Any] = {
    "max_hint_checks": 4,
    "enable_variants": True,
    "max_variant_checks": 2,
    "enable_r0_reorder": True,
    "pre_prune_hints": True,
    "enable_db_probe": True,
    "enable_fp_firewall": True,
    "fp_risk_probe_threshold": 0.16,
    "fp_risk_llm_threshold": 0.20,
    "fp_audit_broad": True,
    "fp_risk_r0_broad_threshold": 0.14,
    "fp_risk_r1_broad_threshold": 0.22,
    "fp_hard_flag_escalate_bon": 3,
    "fp_soft_sc_guard": True,
    "enable_schema_anchor": True,
    "use_llm": True,
    "llm_require_adapter": True,
    "llm_prompt_style": "v2_rich_chat",  # "v2_rich_chat" | "legacy_raw"
    "llm_enable_thinking": False,
    "llm_max_new_tokens": 192,
    "llm_best_of_n": 1,
    "llm_second_pass": True,
    "llm_retry_bon": 4,
    "llm_retry_temp": 0.65,
    "llm_local_precheck": True,
    "llm_on_fp": True,
    "llm_fp_trust_rewrite": True,
    "reranker_weights": None,
}


def _parses(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql, read="sqlite")
        return True
    except Exception:
        return False


class HarnessAgent:
    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.reranker = FPReranker(self.cfg.get("reranker_weights"))

        # Task context
        self.task_id: str = ""
        self.question: str = ""
        self.draft: str = ""
        self.ordered: bool = False
        self.example: dict[str, Any] = {}
        self.ex_cols: list[str] = []
        self.ex_rc: int = 0

        # Collected tool artifacts
        self.schema_list: list[dict] | None = None
        self.cat: SchemaCatalog | None = None
        self.raw_hints: list[str] = []
        self.unique_hints: list[dict[str, Any]] = []  # [{"sql", "votes", "idx"}]
        self.probe_plan: dict[str, Any] | None = None
        self.probe_report: dict[str, Any] | None = None
        self.probe_attempted: bool = False

        # Action tracking & candidate pool
        self.used_actions: set[str] = set()
        self.checked_sqls: set[str] = set()
        self.candidates: list[dict[str, Any]] = []  # evaluated candidates
        self.pending_checks: list[tuple[str, str, float]] = []  # (source, sql, bonus)
        self.last_action_meta: dict[str, Any] = {}
        self.last_action_meta_source: str = ""
        self.last_action_meta_bonus: float = 0.0

        # Stage flags
        self.draft_checked: bool = False
        self.hints_fetched: bool = False
        self.hints_queued: bool = False
        self.variants_queued: bool = False
        self.lit_repair_queued: bool = False
        self.r0_reorder_done: bool = False
        self.fp_audit_done: bool = False
        self.llm_repair_done: bool = False
        self.llm_retry_done: bool = False
        self.llm_post_var_done: bool = False
        self.anchor_done: bool = False
        self.last_llm_cand: dict[str, Any] | None = None
        self.unverified_llm_attempt: dict[str, Any] | None = None

    # ---------- Public interface ----------

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        try:
            self._ingest_last_observation(observation)
            return self._decide(observation)
        except Exception:
            return self._submit_best()

    # ---------- Observation ingestion ----------

    def _ingest_last_observation(self, observation: dict[str, Any]) -> None:
        if not self.task_id:
            task = observation.get("task") or {}
            self.task_id = str(task.get("id") or "")
            self.question = str(task.get("question") or "")
            self.draft = str(observation.get("issue_sql") or "")
            self.ordered = bool(observation.get("ordered", False))
            self.example = dict(observation.get("example") or {})
            self.ex_cols = list(self.example.get("columns") or [])
            self.ex_rc = int(self.example.get("row_count") or 0)
            llm_runner.reset_task_budget(self.task_id)

        history = observation.get("history") or []
        if not history or not self.last_action_meta:
            return

        last_ev = history[-1]
        fb = last_ev.get("observation")
        kind = self.last_action_meta.get("kind")
        self.last_action_meta = {}

        if kind == "check":
            sql = last_ev.get("sql") or ""
            source = self.last_action_meta_source
            bonus = self.last_action_meta_bonus
            self._record_candidate(source, sql, fb if isinstance(fb, dict) else {}, bonus)
            if source == "draft":
                self.draft_checked = True
            if source.startswith("llm"):
                self.last_llm_cand = self.candidates[-1]
        elif kind == "repair_hint":
            self.hints_fetched = True
            self._parse_hints(fb)
        elif kind == "schema":
            if isinstance(fb, list):
                self.schema_list = fb
                self.cat = SchemaCatalog(fb)
                self._refresh_all_audits()
        elif kind == "probe":
            self.probe_attempted = True
            if isinstance(fb, dict) and self.probe_plan:
                self.probe_report = parse_probe_result(fb, self.probe_plan)
                self._refresh_all_audits()

    def _parse_hints(self, suggestions: Any) -> None:
        if not isinstance(suggestions, list):
            return
        raw: list[str] = []
        for item in suggestions:
            if isinstance(item, str) and item.strip():
                raw.append(item.strip())
        self.raw_hints = raw

        vote_counts = Counter(canonical_sql_key(s) for s in raw if _parses(s))
        seen_sql: set[str] = set()
        seen_canon: set[str] = set()
        uniq: list[dict[str, Any]] = []
        for idx, s in enumerate(raw):
            if s in seen_sql or not _parses(s):
                continue
            ck = canonical_sql_key(s)
            if ck in seen_canon:
                continue
            seen_sql.add(s)
            seen_canon.add(ck)
            uniq.append({"sql": s, "votes": vote_counts[ck], "idx": idx})
        self.unique_hints = uniq
        self._refresh_all_audits()

    def _record_candidate(self, source: str, sql: str, feedback: dict[str, Any], bonus: float = 0.0) -> None:
        feats, audit = extract_reranker_features(
            sql=sql,
            source=source,
            cat=self.cat,
            question=self.question,
            ordered=self.ordered,
            example=self.example,
            feedback=feedback,
            probe_report=self.probe_report,
            hint_sqls=self.raw_hints,
        )
        entry = {
            "source": source,
            "sql": sql,
            "feedback": feedback,
            "matches": bool(feedback.get("matches_example")),
            "ok": bool(feedback.get("ok")),
            "features": feats,
            "warnings": audit["warnings"],
            "bonus": bonus,
            "order_idx": len(self.candidates),
        }
        self.candidates.append(entry)

    def _refresh_all_audits(self) -> None:
        for c in self.candidates:
            feats, audit = extract_reranker_features(
                sql=c["sql"],
                source=c["source"],
                cat=self.cat,
                question=self.question,
                ordered=self.ordered,
                example=self.example,
                feedback=c["feedback"],
                probe_report=self.probe_report,
                hint_sqls=self.raw_hints,
            )
            c["features"] = feats
            c["warnings"] = audit["warnings"]

    # ---------- Decision logic ----------

    def _decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        remaining = int(observation.get("remaining") or 0)
        if remaining <= 0:
            return self._submit_best()

        # Step 1: ALWAYS check exact original draft first (hard protocol requirement).
        if not self.draft_checked:
            return self._action_check("draft", self.draft)

        # Step 2: ALWAYS fetch repair_hint second (if budget allows).
        if not self.hints_fetched and remaining >= 2 and self.cfg["max_hint_checks"] > 0:
            return self._action_tool({"tool": "repair_hint"}, kind="repair_hint")

        # Step 3: For high-FP-risk tasks (ex_rows == 0) fetch schema EARLY so we can
        # audit FK joins and pre-prune wrong-col-count hints before burning check calls!
        if (
            self.cfg["enable_fp_firewall"]
            and self.ex_rc == 0
            and self.schema_list is None
            and remaining >= 2
        ):
            return self._action_tool({"tool": "schema"}, kind="schema")

        # Step 4: Queue unique hints if not yet queued.
        if not self.hints_queued:
            self.hints_queued = True
            self._queue_hints(remaining)

        # Step 5: Drain pending checks (hints / deterministic repairs / LLM candidates)
        while self.pending_checks and remaining >= 1:
            if self._can_fast_stop():
                return self._submit_best()
            is_spec_var = self.pending_checks[0][0].startswith("var_")
            if not any(c["matches"] for c in self.candidates) and is_spec_var:
                if (
                    self.cfg["enable_db_probe"]
                    and not self.probe_attempted
                    and self.schema_list is not None
                    and remaining == 2
                ):
                    break
                if self.probe_report and self.probe_report.get("ok") and not self.lit_repair_queued:
                    break
            src, sql, bonus = self.pending_checks.pop(0)
            if sql in self.checked_sqls:
                continue
            return self._action_check(src, sql, bonus=bonus)

        # Step 6: Evaluate current state — do we already have a public-passing candidate?
        passing = [c for c in self.candidates if c["matches"]]

        if passing:
            # --- FALSE-POSITIVE FIREWALL PATH ---
            if not self.cfg["enable_fp_firewall"]:
                return self._submit_best()

            best = self.reranker.pick_best_passing(passing)
            need_schema_for_fp = (
                self.schema_list is None
                and remaining >= 2
                and (
                    self.ex_rc <= 1
                    or self.reranker.has_hard_red_flag(best["features"])
                    or self.reranker.fp_risk(best["features"]) >= self.cfg["fp_risk_probe_threshold"]
                )
            )
            if need_schema_for_fp:
                return self._action_tool({"tool": "schema"}, kind="schema")

            best = self.reranker.pick_best_passing(passing)
            best_feats = best["features"]
            best_risk = self.reranker.fp_risk(best_feats)
            hard_flag = self.reranker.has_hard_red_flag(best_feats)

            # Upgrade 3a: Zero-row passing column-order alignment against example.columns (+3/-0 on train!)
            if (
                self.cfg.get("enable_r0_reorder", True)
                and self.ex_rc == 0
                and len(self.ex_cols) >= 2
                and not self.r0_reorder_done
                and remaining >= 1
            ):
                self.r0_reorder_done = True
                reord = reorder_columns(best["sql"], self.ex_cols)
                if reord and reord not in self.checked_sqls:
                    # Verify normalized alignment strictly improves
                    try:
                        t_reord = sqlglot.parse_one(reord, read="sqlite")
                        if isinstance(t_reord, exp.Select):
                            new_cols = [e.alias_or_name or e.sql(dialect="sqlite") for e in t_reord.expressions]
                            if column_alignment_score(new_cols, self.ex_cols) > best_feats.get("col_align_score", 0.0) + 0.15:
                                return self._action_check("var_r0_reorder", reord, bonus=0.35)
                    except Exception:
                        pass

            # Check if passing candidate has alphabetical ORDER BY on an *_id column fixable via FK text col!
            if (
                best_feats.get("alphabetical_id_order", 0) > 0
                and self.schema_list
                and remaining >= 1
            ):
                alpha_fixed = repair_alphabetical_id_order(best["sql"], self.schema_list, self.question)
                if alpha_fixed and alpha_fixed not in self.checked_sqls:
                    return self._action_check("repair_alpha_order", alpha_fixed, bonus=0.45)

            # Check if passing candidate has a bogus non-FK join fixable via unique FK bridge!
            if (
                best_feats.get("suspicious_non_fk_joins", 0) > 0
                and self.schema_list
                and remaining >= 1
            ):
                bridged = repair_bogus_join_via_fk_bridge(best["sql"], self.schema_list)
                if bridged and bridged not in self.checked_sqls:
                    return self._action_check("repair_fk_bridge", bridged, bonus=0.35)

            # Run 1-call DB probe
            if (
                self.cfg["enable_db_probe"]
                and not self.probe_attempted
                and self.schema_list
                and remaining >= 2
                and self._should_probe_passing(best, best_risk, hard_flag)
            ):
                act = self._maybe_action_probe()
                if act is not None:
                    return act

            # After probe: deterministic literal repair on passing FP
            if (
                self.probe_report
                and self.probe_report.get("ok")
                and not self.lit_repair_queued
                and remaining >= 1
            ):
                self.lit_repair_queued = True
                for cand in [best] + [c for c in self.candidates if c is not best]:
                    repaired = repair_literals_from_probe(
                        cand["sql"], self.schema_list or [], self.probe_report, self.question
                    )
                    if repaired and repaired not in self.checked_sqls:
                        return self._action_check("lit_repair", repaired, bonus=0.45)

            # Check if a non-passing hint had LEFT JOIN excess rows while draft passed via bogus non-FK join
            if (
                best_feats.get("suspicious_non_fk_joins", 0) > 0
                and not self.variants_queued
                and remaining >= 1
            ):
                self.variants_queued = True
                non_pass = [c for c in self.candidates if not c["matches"] and c["ok"]]
                for c in non_pass[:2]:
                    for tag, vsql in generate_variants(
                        c["sql"], self.schema_list, self.example, c["feedback"], self.question, self.ordered
                    ):
                        if vsql not in self.checked_sqls:
                            return self._action_check(f"var_{tag}", vsql, bonus=0.35)

            # Upgrade 2: Calibrated LLM FP-Audit (+ Hard-Flag Bo3 Escalation & Soft-Audit SC Guard)
            if (
                self._llm_ready()
                and self.cfg["llm_on_fp"]
                and not self.fp_audit_done
                and not best["source"].startswith("llm")
                and remaining >= 1
                and self.schema_list
                and self._should_llm_fp_audit(best, best_risk, hard_flag)
            ):
                self.fp_audit_done = True
                fp_cands = self._run_fp_audit_llm(best=best, hard_flag=hard_flag)
                for lsql, fp_bonus in fp_cands:
                    if lsql not in self.checked_sqls:
                        self.pending_checks.append(("llm_fp", lsql, fp_bonus))
                if self.pending_checks:
                    src, sql, bonus = self.pending_checks.pop(0)
                    return self._action_check(src, sql, bonus=bonus)

            return self._submit_best()

        # --- NO CANDIDATE PASSED PUBLIC YET (REPAIR PATH) ---

        # Ensure schema is loaded
        if self.schema_list is None and remaining >= 2:
            return self._action_tool({"tool": "schema"}, kind="schema")

        # Step A: Deterministic projection/disambiguation/LEFT->INNER/reorder variants
        if self.cfg["enable_variants"] and not self.variants_queued and remaining >= 1:
            self.variants_queued = True
            self._queue_deterministic_variants()
            if self.pending_checks:
                if not (
                    self.cfg["enable_db_probe"]
                    and not self.probe_attempted
                    and self.schema_list is not None
                    and remaining == 2
                ):
                    src, sql, bonus = self.pending_checks.pop(0)
                    return self._action_check(src, sql, bonus=bonus)

        # Step B: 1-call DB probe (feeds deterministic literal repair AND LLM prompt!)
        if (
            self.cfg["enable_db_probe"]
            and not self.probe_attempted
            and self.schema_list
            and remaining >= 2
        ):
            act = self._maybe_action_probe()
            if act is not None:
                return act

        # Step C: Deterministic literal repair from DB probe
        if (
            self.probe_report
            and self.probe_report.get("ok")
            and not self.lit_repair_queued
            and remaining >= 1
        ):
            self.lit_repair_queued = True
            for cand in self._prioritized_sources():
                repaired = repair_literals_from_probe(
                    cand["sql"], self.schema_list or [], self.probe_report, self.question
                )
                if repaired and repaired not in self.checked_sqls:
                    self.pending_checks.insert(0, ("lit_repair", repaired, 0.2))
                    break
            if self.pending_checks and self.pending_checks[0][0] == "lit_repair":
                src, sql, bonus = self.pending_checks.pop(0)
                return self._action_check(src, sql, bonus=bonus)

        # Drain any remaining variant if LLM is disabled
        if not self._llm_ready() and self.pending_checks and remaining >= 1:
            src, sql, bonus = self.pending_checks.pop(0)
            if sql not in self.checked_sqls:
                return self._action_check(src, sql, bonus=bonus)

        # Step D: Pass 1 Live LLM repair (greedy + local pre-validation & SetOp/alias autofix!)
        if self._llm_ready() and not self.llm_repair_done and remaining >= 1 and self.schema_list:
            self.llm_repair_done = True
            best_src = self._prioritized_sources()[0] if self.candidates else None
            raw_sqls = self._invoke_llm_batch(
                mode="repair",
                source_cand=best_src,
                n=int(self.cfg.get("llm_best_of_n", 1)),
                temperature=0.0 if int(self.cfg.get("llm_best_of_n", 1)) == 1 else float(self.cfg.get("llm_retry_temp", 0.65)),
            )
            ranked = self._rank_and_polish_llm_batch(raw_sqls, allow_invalid_fallback=False)
            if ranked:
                max_chk = min(len(ranked), max(1, remaining - (1 if self.cfg["llm_second_pass"] and remaining >= 2 else 0)))
                for idx, (lsql, _) in enumerate(ranked[:max_chk]):
                    if lsql not in self.checked_sqls:
                        self.pending_checks.append((f"llm_{idx}", lsql, 0.15))
                if self.pending_checks:
                    src, sql, bonus = self.pending_checks.pop(0)
                    return self._action_check(src, sql, bonus=bonus)
            elif raw_sqls:
                # Greedy produced a duplicate or locally invalid SQL (e.g. outer alias in FROM subquery)!
                # Record the local error so Pass 2 retry sees why it failed without wasting a `check` call!
                bad_sql = raw_sqls[0]
                ok_loc, reason = validate_executable_locally(bad_sql, self.cat)
                self.unverified_llm_attempt = {
                    "sql": bad_sql,
                    "feedback": {
                        "ok": False,
                        "error": {"type": "StaticSchemaError", "message": reason},
                    } if not ok_loc else next((c["feedback"] for c in self.candidates if c["sql"] == bad_sql), {}),
                    "warnings": [reason] if not ok_loc else [],
                }

        # Step E: Upgrade 1 — Pass 2 KV-Cache-Shared Diverse Best-of-N (`mode="retry"`)!
        # Fires when Pass 1 failed check (SQL error OR wrong result!) OR was skipped locally!
        if (
            self._llm_ready()
            and self.cfg["llm_second_pass"]
            and not self.llm_retry_done
            and remaining >= 1
            and self.schema_list
        ):
            self.llm_retry_done = True
            # First check if a fast deterministic variant fixes Pass 1's checked output
            if self.last_llm_cand is not None and not self.last_llm_cand["matches"]:
                llm_vars = generate_variants(
                    self.last_llm_cand["sql"],
                    self.schema_list,
                    self.example,
                    self.last_llm_cand["feedback"],
                    self.question,
                    self.ordered,
                )
                for tag, vsql in llm_vars[:1]:
                    if vsql not in self.checked_sqls and tag in ("disambiguate+prune", "disambiguate", "prune_cols", "reorder_cols"):
                        self.pending_checks.append((f"llm_var_{tag}", vsql, 0.15))

            retry_n = min(int(self.cfg.get("llm_retry_bon", 4)), llm_runner.remaining_completions(self.task_id))
            if retry_n >= 1:
                # Use mode="repair" (without anchoring on failed Pass 1) + multi-temperature ladder [0.80..1.40]
                # so the model explores genuinely distinct SQL structures instead of copying Pass 1!
                best_src = self._prioritized_sources()[0] if self.candidates else None
                retry_temp = max(0.80, float(self.cfg.get("llm_retry_temp", 0.80))) if retry_n > 1 else 0.80
                raw_retry = self._invoke_llm_batch(
                    mode="repair",
                    source_cand=best_src,
                    n=retry_n,
                    temperature=retry_temp,
                )
                ranked_retry = self._rank_and_polish_llm_batch(raw_retry, allow_invalid_fallback=True)
                for idx, (lsql, _) in enumerate(ranked_retry[: min(2, remaining)]):
                    if lsql not in self.checked_sqls and all(s != lsql for _, s, _ in self.pending_checks):
                        self.pending_checks.append((f"llm_retry_{idx}", lsql, 0.18))

            if self.pending_checks:
                src, sql, bonus = self.pending_checks.pop(0)
                return self._action_check(src, sql, bonus=bonus)

        # Step E2: Post-LLM deterministic variant on checked retry candidate if budget still remains
        if (
            not self.llm_post_var_done
            and self.last_llm_cand is not None
            and self.last_llm_cand["ok"]
            and not self.last_llm_cand["matches"]
            and remaining >= 1
        ):
            self.llm_post_var_done = True
            for tag, vsql in generate_variants(
                self.last_llm_cand["sql"],
                self.schema_list,
                self.example,
                self.last_llm_cand["feedback"],
                self.question,
                self.ordered,
            ):
                if vsql not in self.checked_sqls:
                    return self._action_check(f"llm_post_{tag}", vsql, bonus=0.15)

        # Step F: Last-resort schema-anchoring repair
        if (
            self.cfg["enable_schema_anchor"]
            and schema_anchor_repair is not None
            and not self.anchor_done
            and self.schema_list
            and remaining >= 1
        ):
            self.anchor_done = True
            for cand in self._prioritized_sources()[:2]:
                if not cand["ok"]:
                    try:
                        fixed = schema_anchor_repair(cand["sql"], self.schema_list)
                    except Exception:
                        fixed = None
                    if fixed and fixed not in self.checked_sqls:
                        return self._action_check("anchor_repair", fixed, bonus=0.05)

        return self._submit_best()

    # ---------- Local Pre-Validation, Polishing & Ranking ----------

    def _polish_candidate_locally(self, sql: str) -> tuple[str, bool]:
        """Apply safe schema/literal/projection fixes BEFORE spending a `check` tool call."""
        if not sql or not self.cfg.get("llm_local_precheck", True):
            return sql, _parses(sql)

        cur = sql.strip()
        if self.cat is not None and self.schema_list:
            ok_loc, reason = validate_executable_locally(cur, self.cat)
            if not ok_loc:
                if "ambiguous_col:" in reason:
                    col_name = reason.split(":", 1)[1]
                    dis = disambiguate_columns(cur, self.schema_list, f"ambiguous column name: {col_name}")
                    if dis:
                        cur = dis
                        ok_loc, reason = validate_executable_locally(cur, self.cat)
                if not ok_loc and schema_anchor_repair is not None:
                    try:
                        anchored = schema_anchor_repair(cur, self.schema_list)
                    except Exception:
                        anchored = None
                    if anchored:
                        ok_anch, _ = validate_executable_locally(anchored, self.cat)
                        if ok_anch:
                            cur = anchored

        if self.probe_report and self.probe_report.get("ok") and self.schema_list:
            lit_fixed = repair_literals_from_probe(cur, self.schema_list, self.probe_report, self.question)
            if lit_fixed:
                cur = lit_fixed

        if self.ex_cols:
            pruned = prune_extra_columns(cur, self.ex_cols)
            if pruned:
                cur = pruned

        gb_fixed = add_missing_group_by_for_mixed_agg(cur, self.ex_rc)
        if gb_fixed:
            cur = gb_fixed

        valid_ok = validate_executable_locally(cur, self.cat)[0] if self.cat is not None else _parses(cur)
        return cur, valid_ok

    def _rank_and_polish_llm_batch(
        self, raw_sqls: list[str], allow_invalid_fallback: bool = False
    ) -> list[tuple[str, int]]:
        """Polish a batch of LLM samples, deduplicate against checked_sqls, and rank by SC votes + audit."""
        if not raw_sqls:
            return []
        polished_items: list[tuple[str, bool, str]] = []
        for rsql in raw_sqls:
            psql, vok = self._polish_candidate_locally(rsql)
            if not _parses(psql):
                continue
            ck = canonical_sql_key(psql)
            polished_items.append((psql, vok, ck))

        if not polished_items:
            return []

        vote_counts = Counter(ck for _, _, ck in polished_items)
        seen_ck: set[str] = set()
        scored: list[tuple[tuple, str, int]] = []
        for idx, (psql, vok, ck) in enumerate(polished_items):
            if ck in seen_ck or psql in self.checked_sqls:
                continue
            seen_ck.add(ck)
            if not vok and not allow_invalid_fallback:
                continue
            aud = audit_sql(psql, self.cat, self.question, self.ordered, self.ex_cols, probe_report=self.probe_report)
            af = aud["features"]
            votes = vote_counts[ck]
            key = (
                1 if vok else 0,
                1 if af["col_count_match"] else 0,
                1 if af["suspicious_non_fk_joins"] == 0 else 0,
                votes,
                round(af["col_align_score"], 3),
                -idx,
            )
            scored.append((key, psql, votes))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(psql, votes) for _, psql, votes in scored]

    # ---------- Upgrade 2: FP-Audit Policy & Escalation ----------

    def _should_probe_passing(self, best: dict[str, Any], risk: float, hard_flag: bool) -> bool:
        if hard_flag:
            return True
        if self.cfg.get("fp_audit_broad", False) and self.ex_rc == 0:
            return True
        if self.cat is not None:
            for c in self.candidates[:4]:
                preds = extract_sql_predicates(c["sql"], self.cat)
                if any(p["is_string"] for p in preds) and self.ex_rc <= 1:
                    return True
        if self.ex_rc == 0 and risk >= self.cfg["fp_risk_probe_threshold"]:
            return True
        if self.cfg.get("fp_audit_broad", False) and self.ex_rc == 1 and risk >= 0.18:
            return True
        return False

    def _should_llm_fp_audit(self, best: dict[str, Any], best_risk: float, hard_flag: bool) -> bool:
        if hard_flag:
            return True
        if not self.cfg.get("fp_audit_broad", False):
            return bool(self.ex_rc == 0 and best_risk >= self.cfg["fp_risk_llm_threshold"])

        feats = best["features"]
        if self.ex_rc == 0:
            if best_risk >= float(self.cfg.get("fp_risk_r0_broad_threshold", 0.14)):
                return True
            if feats.get("table_count_norm", 0.0) >= 2.0 / 6.0 - 1e-6:
                return True
            if feats.get("extra_distinct_no_cue", 0.0) > 0:
                return True
            return False

        if self.ex_rc == 1:
            if feats.get("extra_distinct_no_cue", 0.0) > 0 and feats.get("table_count_norm", 0.0) >= 2.0 / 6.0 - 1e-6:
                return True
        return False

    def _boost_existing_passing(self, target_ck: str, bonus: float) -> bool:
        """If an LLM FP-audit votes for another ALREADY-CHECKED passing candidate (e.g. draft
        without DISTINCT over hint with DISTINCT), boost it directly without spending a tool call!"""
        for c in self.candidates:
            if c["matches"] and canonical_sql_key(c["sql"]) == target_ck:
                c["bonus"] = max(float(c.get("bonus", 0.0)), bonus)
                return True
        return False

    def _run_fp_audit_llm(self, *, best: dict[str, Any], hard_flag: bool) -> list[tuple[str, float]]:
        """Run greedy `fp_audit`, with Bo3 escalation on hard-flag rubber-stamping
        and Bo2 self-consistency guard on soft rewrites."""
        best_ck = canonical_sql_key(best["sql"])
        raw_g = self._invoke_llm_batch(mode="fp_audit", source_cand=best, n=1, temperature=0.0)
        if not raw_g:
            return []

        g_sql, g_valid = self._polish_candidate_locally(raw_g[0])
        g_ck = canonical_sql_key(g_sql)
        same_as_best = (g_ck == best_ck) or (not g_valid)

        # Case 1: Hard red flag exists -> if greedy produced a valid fix != best, trust it;
        # if greedy rubber-stamped `best`, escalate to multi-temperature Bo3!
        if hard_flag:
            if not same_as_best:
                if self._boost_existing_passing(g_ck, 0.65):
                    return []
                if g_sql not in self.checked_sqls:
                    return [(g_sql, 0.65)]
            esc_n = min(
                int(self.cfg.get("fp_hard_flag_escalate_bon", 3)),
                llm_runner.remaining_completions(self.task_id),
            )
            if esc_n >= 1 and self.cfg.get("fp_audit_broad", False):
                raw_esc = self._invoke_llm_batch(
                    mode="fp_audit",
                    source_cand=best,
                    n=esc_n,
                    temperature=max(0.85, float(self.cfg.get("llm_retry_temp", 0.85))),
                )
                ranked_esc = self._rank_and_polish_llm_batch(raw_esc, allow_invalid_fallback=False)
                clean_esc = []
                for cand_sql, votes in ranked_esc:
                    cck = canonical_sql_key(cand_sql)
                    if cck == best_ck:
                        continue
                    feats_c, _ = extract_reranker_features(
                        sql=cand_sql,
                        source="llm_fp",
                        cat=self.cat,
                        question=self.question,
                        ordered=self.ordered,
                        example=self.example,
                        feedback={"ok": True, "columns": self.ex_cols, "matches_example": True},
                        probe_report=self.probe_report,
                        hint_sqls=self.raw_hints,
                    )
                    if not self.reranker.has_hard_red_flag(feats_c):
                        if self._boost_existing_passing(cck, 0.65):
                            return []
                        clean_esc.append((cand_sql, 0.65))
                if clean_esc:
                    return clean_esc[:1]
            return []

        # Case 2: Soft audit (no hard red flag)
        if same_as_best:
            return []  # Greedy confirmed `best` -> 0 extra tool calls!

        fp_bonus = 0.65 if self.cfg.get("llm_fp_trust_rewrite", True) else 0.12
        need_sc = bool(
            self.cfg.get("fp_audit_broad", False)
            and self.cfg.get("fp_soft_sc_guard", True)
            and (self.ex_rc >= 1 or self.reranker.fp_risk(best["features"]) < self.cfg["fp_risk_llm_threshold"])
        )
        if not need_sc:
            if self._boost_existing_passing(g_ck, fp_bonus):
                return []
            return [(g_sql, fp_bonus)] if g_sql not in self.checked_sqls else []

        sc_n = min(2, llm_runner.remaining_completions(self.task_id))
        if sc_n <= 0:
            return []
        raw_sc = self._invoke_llm_batch(mode="fp_audit", source_cand=best, n=sc_n, temperature=0.50)
        all_cks = [g_ck]
        for s in raw_sc:
            ps, pv = self._polish_candidate_locally(s)
            if pv:
                all_cks.append(canonical_sql_key(ps))
        if all_cks.count(g_ck) >= 2:
            if self._boost_existing_passing(g_ck, fp_bonus):
                return []
            return [(g_sql, fp_bonus)] if g_sql not in self.checked_sqls else []
        return []

    # ---------- Queueing & Helpers ----------

    def _queue_hints(self, remaining: int) -> None:
        if not self.unique_hints:
            return
        max_h = min(self.cfg["max_hint_checks"], len(self.unique_hints))
        if self.ex_rc == 0:
            max_h = min(max_h, max(1, remaining - 3))
        else:
            max_h = min(max_h, max(1, remaining - 2))

        scored_hints = []
        for h in self.unique_hints:
            h_sql = h["sql"]
            # Upgrade 3b: Pre-prune hints that obviously have more SELECT columns than example_cols!
            if self.cfg.get("pre_prune_hints", True) and self.ex_cols:
                pruned = prune_extra_columns(h_sql, self.ex_cols)
                if pruned and _parses(pruned):
                    h_sql = pruned
            if h_sql == self.draft:
                continue
            audit = audit_sql(h_sql, self.cat, self.question, self.ordered, self.ex_cols)
            af = audit["features"]
            key = (
                1 if af["suspicious_non_fk_joins"] == 0 else 0,
                1 if af["col_count_match"] else 0,
                h["votes"],
                -h["idx"],
            )
            scored_hints.append((key, h["idx"], h_sql))
        scored_hints.sort(key=lambda x: x[0], reverse=True)

        seen_q = set()
        for _, h_idx, h_sql in scored_hints:
            if h_sql in seen_q:
                continue
            seen_q.add(h_sql)
            self.pending_checks.append((f"hint_{h_idx}", h_sql, 0.0))
            if len(self.pending_checks) >= max_h:
                break

    def _can_fast_stop(self) -> bool:
        passing = [c for c in self.candidates if c["matches"]]
        if not passing:
            return False
        if not self.cfg["enable_fp_firewall"]:
            return any(c["source"] != "draft" for c in passing)

        if self.ex_rc == 0:
            return False
        if self.ex_rc == 1:
            non_draft_pass = [c for c in passing if c["source"] != "draft"]
            if not non_draft_pass:
                return False
            if any(src.startswith(("repair", "lit", "llm")) for src, _, _ in self.pending_checks):
                return False
            c0 = non_draft_pass[0]
            if c0["warnings"]:
                return False
            if self.cfg.get("fp_audit_broad", False) and c0["features"].get("extra_distinct_no_cue", 0.0) > 0:
                return False
            if self.cat is not None:
                lits = [p for p in extract_sql_predicates(c0["sql"], self.cat) if p["is_string"]]
                if lits:
                    return False
            if self.pending_checks and c0["features"].get("hint_vote_frac", 0.0) < 0.75:
                return False
            return True

        non_draft_pass = [c for c in passing if c["source"] != "draft"]
        if not non_draft_pass:
            return False
        best = self.reranker.pick_best_passing(non_draft_pass)
        return not self.reranker.has_hard_red_flag(best["features"])

    def _maybe_action_probe(self) -> dict[str, Any] | None:
        self.probe_attempted = True
        if not self.schema_list:
            return None
        cand_sqls = [c["sql"] for c in self._prioritized_sources()[:3]]
        for h in self.unique_hints[:2]:
            if h["sql"] not in cand_sqls:
                cand_sqls.append(h["sql"])
        q, plan = build_probe_query(self.schema_list, self.question, cand_sqls, self.ex_cols)
        if not q:
            return None
        action = {"tool": "execute", "sql": q}
        ident = json.dumps(action, sort_keys=True)
        if ident in self.used_actions:
            return None
        self.probe_plan = plan
        return self._action_tool(action, kind="probe")

    def _prioritized_sources(self) -> list[dict[str, Any]]:
        def _k(c: dict[str, Any]):
            is_hint = 1 if c["source"].startswith("hint") else 0
            return (1 if c["ok"] else 0, is_hint, c["features"].get("col_align_score", -1.0), -c["order_idx"])
        return sorted(self.candidates, key=_k, reverse=True)

    def _queue_deterministic_variants(self) -> None:
        added = 0
        limit = int(self.cfg["max_variant_checks"])
        for cand in self._prioritized_sources()[:3]:
            variants = generate_variants(
                sql=cand["sql"],
                schema=self.schema_list,
                example=self.example,
                feedback=cand["feedback"],
                question=self.question,
                ordered=self.ordered,
            )
            for tag, vsql in variants:
                if vsql in self.checked_sqls or any(s == vsql for _, s, _ in self.pending_checks):
                    continue
                self.pending_checks.append((f"var_{tag}", vsql, 0.1))
                added += 1
                if added >= limit:
                    return

    def _llm_ready(self) -> bool:
        if not self.cfg["use_llm"]:
            return False
        if llm_runner.remaining_completions(self.task_id) <= 0:
            return False
        return llm_runner.available(require_adapter=bool(self.cfg["llm_require_adapter"]))

    def _invoke_llm_batch(
        self,
        *,
        mode: str,
        source_cand: dict[str, Any] | None,
        n: int = 1,
        temperature: float = 0.0,
    ) -> list[str]:
        style = self.cfg.get("llm_prompt_style", "v2_rich_chat")
        max_new = int(self.cfg.get("llm_max_new_tokens", 192))

        if style == "legacy_raw":
            src_sql = source_cand["sql"] if source_cand else self.draft
            src_fb = source_cand["feedback"] if source_cand else {}
            prompt = llm_runner.build_legacy_raw_prompt(
                self.question, self.schema_list or [], src_sql, src_fb, self.example
            )
            return llm_runner.generate_legacy_raw(
                task_id=self.task_id,
                prompt=prompt,
                max_new_tokens=min(max_new, 128),
                n=n,
                temperature=max(0.5, temperature) if n > 1 else 0.6,
            )

        draft_cand = next((c for c in self.candidates if c["source"] == "draft"), None)
        by_sql = {c["sql"]: c for c in self.candidates}
        hint_entries = []
        for h in self.unique_hints[:4]:
            ev = by_sql.get(h["sql"])
            if ev is not None:
                hint_entries.append({
                    "sql": h["sql"],
                    "feedback": ev["feedback"],
                    "warnings": ev["warnings"],
                    "votes": h["votes"],
                })
            else:
                aud = audit_sql(
                    h["sql"], self.cat, self.question, self.ordered, self.ex_cols, probe_report=self.probe_report
                )
                hint_entries.append({
                    "sql": h["sql"],
                    "feedback": None,
                    "warnings": aud["warnings"],
                    "votes": h["votes"],
                })

        prev_attempt = None
        if mode == "retry" and source_cand is not None:
            prev_attempt = {
                "sql": source_cand["sql"],
                "feedback": source_cand.get("feedback"),
                "warnings": source_cand.get("warnings"),
            }

        messages = build_messages(
            question=self.question,
            schema=self.schema_list or [],
            ordered=self.ordered,
            example=self.example,
            issue_sql=self.draft,
            draft_feedback=draft_cand["feedback"] if draft_cand else None,
            draft_warnings=draft_cand["warnings"] if draft_cand else None,
            hints=hint_entries,
            probe_report=self.probe_report,
            mode=mode,
            previous_attempt=prev_attempt,
        )
        return llm_runner.generate_chat_raw_batch(
            task_id=self.task_id,
            messages=messages,
            mode=mode,
            max_new_tokens=max_new,
            n=n,
            temperature=temperature,
            enable_thinking=bool(self.cfg.get("llm_enable_thinking", False)),
        )

    # ---------- Tool action helpers ----------

    def _action_check(self, source: str, sql: str, bonus: float = 0.0) -> dict[str, Any]:
        action = {"tool": "check", "sql": sql}
        ident = json.dumps(action, sort_keys=True)
        if ident in self.used_actions:
            return self._submit_best()
        self.used_actions.add(ident)
        self.checked_sqls.add(sql)
        self.last_action_meta = {"kind": "check"}
        self.last_action_meta_source = source
        self.last_action_meta_bonus = bonus
        return action

    def _action_tool(self, action: dict[str, Any], kind: str) -> dict[str, Any]:
        ident = json.dumps(action, sort_keys=True)
        if ident in self.used_actions:
            return self._submit_best()
        self.used_actions.add(ident)
        self.last_action_meta = {"kind": kind}
        return action

    def _submit_best(self) -> dict[str, Any]:
        passing = [c for c in self.candidates if c["matches"]]
        if passing:
            best = self.reranker.pick_best_passing(passing)
            return {"tool": "submit", "sql": best["sql"]}
        if self.candidates:
            best = self.reranker.pick_best_fallback(self.candidates, self.example)
            return {"tool": "submit", "sql": best["sql"]}
        return {"tool": "submit", "sql": self.draft or "SELECT 1"}


def create_agent(checkpoint=None):
    config: dict[str, Any] = {}
    cfg_path = Path(checkpoint) if checkpoint is not None else (HARNESS_DIR / "config.json")
    if cfg_path.is_file():
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
    return HarnessAgent(config)
