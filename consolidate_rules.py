"""Self-calibration rule generation, merging, and pruning for the Skip-Prediction loop.

Called by calibrate.py after every batch of 20-30 predictions has been compared
against ground-truth labels. This module is fully standalone and testable via its
__main__ smoke test without needing the outer calibration loop.

High-level pipeline:
    Step 1  — Cluster misses by shared feature patterns (deterministic, no LLM).
    Step 2  — Synthesize candidate rules for large-enough clusters (LLM, thinking mode).
    Step 3  — Prune underperforming rules; merge near-duplicates (LLM for wording).
    Step 4  — Write all changes atomically to skip_predictor.db, insert runs row.

Returns (from consolidate_rules()):
    summary dict matching the runs.summary JSON contract:
    {
        "rule_changes": [
            {"rule_id": 7, "action": "pruned", "reason": "..."},
            {"rule_id": 12, "action": "created", "reason": "..."},
            {"rule_id": 3, "action": "merged", "into_rule_id": 12, "reason": "..."},
        ]
    }

LLM calling conventions:
    - Rule synthesis and merge prompts: thinking mode ENABLED (no /no_think prefix).
      These are slow, one-off calls — expect several minutes each on this hardware.
    - Do NOT use llm_provider.predict_chat() directly here; call it via _llm_think().

batch_results schema (each element):
    {
        "track_id": int,
        "predicted_verdict": "Keep" | "Skip",
        "actual_label": "keep" | "skip",
        "confidence": int,
        "reasoning": str,       # verbatim from model JSON output
        "genre": str | None,
        "release_year": int | None,
        "album_type": str | None,
        "days_since_added": int | None,
        "artist_cooccurrence_score": float | None,
        "user_playcount": int | None,
        "label_id": int | None, # labels.id used for label_ids in runs
    }
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

MISS_CLUSTER_THRESHOLD = 2       # Minimum misses in a cluster to promote it (supports small 10-15 track batches)
MAX_ACTIVE_RULES = 12            # Hard cap on active rules
PRUNE_MIN_APPLIED = 10           # Must have this many applications to be pruned
PRUNE_MAX_ACCURACY = 0.4         # Prune if correct rate below this threshold
RULE_SIMILARITY_WORD_OVERLAP = 0.5  # Cosine-like word overlap to flag near-duplicates

# Tokens budget for thinking-mode LLM calls (rule synthesis + merge).
# Qwen3 8B needs space for the full <think>...</think> trace before the JSON
# output. 1024 was causing truncation at ~358s; 8192 gives substantial headroom.
THINKING_MAX_TOKENS = 8192

NUMERIC_FEATURES = [
    "days_since_added",
    "artist_cooccurrence_score",
    "user_playcount",
]
CATEGORICAL_FEATURES = [
    "genre",
    "album_type",
]

# ── Quartile helper ───────────────────────────────────────────────────────────

def _compute_quartile_boundaries(
    conn: sqlite3.Connection,
    feature: str,
) -> Tuple[float, float, float]:
    """Compute Q1, Q2 (median), Q3 boundaries for a numeric feature from the full 820-track features table.

    Returns (q1, q2, q3). Values used to bucket individual observations.
    Falls back to (0, 0, 0) if data is unavailable.
    """
    cursor = conn.cursor()
    # SQLite has no native PERCENTILE; compute via ordered row index
    try:
        cursor.execute(
            f"SELECT {feature} FROM features WHERE {feature} IS NOT NULL ORDER BY {feature} ASC;"
        )
        vals = [row[0] for row in cursor.fetchall()]
        if not vals:
            return (0.0, 0.0, 0.0)
        n = len(vals)
        q1 = vals[max(0, int(n * 0.25) - 1)]
        q2 = vals[max(0, int(n * 0.50) - 1)]
        q3 = vals[max(0, int(n * 0.75) - 1)]
        return (float(q1), float(q2), float(q3))
    except Exception:
        return (0.0, 0.0, 0.0)


def _numeric_bucket(value: Optional[float], q1: float, q2: float, q3: float) -> str:
    """Map a numeric value to a quartile label string."""
    if value is None:
        return "unknown"
    if value <= q1:
        return "Q1_low"
    if value <= q2:
        return "Q2_mid_low"
    if value <= q3:
        return "Q3_mid_high"
    return "Q4_high"


# ── Step 1: Miss clustering ───────────────────────────────────────────────────

def _cluster_misses(
    misses: List[Dict[str, Any]],
    conn: sqlite3.Connection,
    min_cluster_size: int = MISS_CLUSTER_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Group misses by shared single-feature patterns or failure modes.

    For small calibration batches (e.g. 10–15 tracks), rule synthesis triggers
    when len(misses) >= 2. Groups misses by categorical feature values, numeric
    quartile buckets, or failure modes (e.g. false keeps vs. false skips).

    Returns list of cluster dicts:
    {
        "feature": str,
        "bucket": str,
        "pattern_description": str,
        "misses": List[Dict],
    }
    """
    if len(misses) < min_cluster_size:
        return []

    # Pre-compute quartile boundaries from full dataset
    quartiles: Dict[str, Tuple[float, float, float]] = {}
    for feat in NUMERIC_FEATURES:
        quartiles[feat] = _compute_quartile_boundaries(conn, feat)

    # Build per-feature buckets
    # single_feature_groups[feature][bucket] = list of miss dicts
    single_feature_groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for miss in misses:
        for feat in CATEGORICAL_FEATURES:
            val = miss.get(feat)
            bucket = str(val).strip().lower() if val else "unknown"
            single_feature_groups[feat][bucket].append(miss)

        for feat in NUMERIC_FEATURES:
            q1, q2, q3 = quartiles[feat]
            bucket = _numeric_bucket(miss.get(feat), q1, q2, q3)
            single_feature_groups[feat][bucket].append(miss)

    # Collect qualifying clusters (threshold met, not trivially "unknown")
    candidates: List[Dict[str, Any]] = []
    seen_track_ids: set = set()  # avoid promoting the same miss into multiple clusters

    for feat, buckets in single_feature_groups.items():
        for bucket_val, bucket_misses in buckets.items():
            if len(bucket_misses) < min_cluster_size:
                continue
            # Require fresh misses not already claimed
            fresh = [m for m in bucket_misses if m["track_id"] not in seen_track_ids]
            if len(fresh) < min_cluster_size:
                continue

            # Build human-readable pattern description
            if feat in CATEGORICAL_FEATURES:
                pattern_desc = f"{feat} = \"{bucket_val}\""
            else:
                q1, q2, q3 = quartiles[feat]
                bucket_ranges = {
                    "Q1_low": f"{feat} <= {q1:.2f}",
                    "Q2_mid_low": f"{q1:.2f} < {feat} <= {q2:.2f}",
                    "Q3_mid_high": f"{q2:.2f} < {feat} <= {q3:.2f}",
                    "Q4_high": f"{feat} > {q3:.2f}",
                    "unknown": f"{feat} = unknown",
                }
                pattern_desc = bucket_ranges.get(bucket_val, f"{feat} ~ {bucket_val}")

            for m in fresh:
                seen_track_ids.add(m["track_id"])

            candidates.append({
                "feature": feat,
                "bucket": bucket_val,
                "pattern_description": pattern_desc,
                "misses": bucket_misses,
            })

    # If single-feature clusters yielded nothing but we have >= 2 misses, group by failure mode
    if not candidates and len(misses) >= min_cluster_size:
        false_keeps = [m for m in misses if m.get("predicted_verdict", "").lower() == "keep"]
        false_skips = [m for m in misses if m.get("predicted_verdict", "").lower() == "skip"]

        for mode_misses, feat, buck, desc in [
            (false_keeps, "user_playcount", "Q1_low", "tracks mispredicted as Keep (actual Skip)"),
            (false_skips, "artist_cooccurrence_score", "Q4_high", "tracks mispredicted as Skip (actual Keep)"),
        ]:
            if len(mode_misses) >= min_cluster_size:
                candidates.append({
                    "feature": feat,
                    "bucket": buck,
                    "pattern_description": desc,
                    "misses": mode_misses,
                })

        # Fallback: if failure modes are split (e.g. 1 false keep + 1 false skip), combine them as general miss batch
        if not candidates and len(misses) >= min_cluster_size:
            candidates.append({
                "feature": "batch_error",
                "bucket": "mixed",
                "pattern_description": f"mixed miss pattern across {len(misses)} tracks in batch",
                "misses": misses,
            })

    return candidates


# ── Step 2: LLM rule synthesis ────────────────────────────────────────────────

def _build_synthesis_prompt(cluster: Dict[str, Any]) -> str:
    """Build the rule synthesis prompt for a miss cluster. Thinking mode — no /no_think."""
    pattern = cluster["pattern_description"]
    misses = cluster["misses"]

    reasoning_block = "\n".join(
        f"  [{i+1}] (predicted {m['predicted_verdict']}, actual {m['actual_label'].capitalize()}) "
        f"— \"{m.get('reasoning', 'no reasoning provided')}\""
        for i, m in enumerate(misses[:10])  # cap at 10 examples to stay within context
    )

    return (
        "You are a music playlist curation rule writer.\n"
        "The skip-prediction model consistently makes errors on a cluster of tracks sharing this feature pattern:\n\n"
        f"  Pattern: {pattern}\n\n"
        "Here are the model's own reasoning strings for these mispredictions:\n"
        f"{reasoning_block}\n\n"
        "Based on the common thread in these errors, write one concise, actionable curation rule that "
        "would have corrected most of these mistakes. The rule should:\n"
        "  - Be phrased in plain English (e.g. \"Skip tracks with X because Y\")\n"
        "  - Reference the specific feature(s) that matter, not just the pattern label\n"
        "  - Be general enough to apply to future tracks, not overfit to this exact batch\n\n"
        "Respond with ONLY this JSON object, no preamble, no markdown fences:\n"
        "{\"rule_text\": \"...\", \"confidence\": \"high|medium|low\", "
        "\"verdict_direction\": \"favor_skip|favor_keep\", \"rationale\": \"...\"}\n\n"
        "verdict_direction must be \"favor_skip\" if the rule advises skipping tracks, "
        "or \"favor_keep\" if it advises keeping them."
    )


def _build_merge_prompt(rule_a: str, rule_b: str) -> str:
    """Build the merge prompt for two near-duplicate rules. Thinking mode — no /no_think."""
    return (
        "You are editing a music playlist curation ruleset.\n"
        "Two rules cover overlapping territory and should be merged into one:\n\n"
        f"  Rule A: {rule_a}\n"
        f"  Rule B: {rule_b}\n\n"
        "Write a single merged rule that captures the core intent of both without "
        "being redundant, overly specific, or losing important nuance.\n\n"
        "Respond with ONLY this JSON object, no preamble, no markdown fences:\n"
        "{\"rule_text\": \"...\", \"verdict_direction\": \"favor_skip|favor_keep\", \"rationale\": \"...\"}\n\n"
        "verdict_direction must be \"favor_skip\" if the merged rule advises skipping tracks, "
        "or \"favor_keep\" if it advises keeping them."
    )


def _parse_llm_json(raw_text: str, required_key: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from model output, stripping common fencing artifacts."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        parsed = json.loads(text[start:end])
        if required_key in parsed:
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _llm_think(prompt: str, max_tokens: int = THINKING_MAX_TOKENS) -> str:
    """Call the model WITHOUT /no_think so thinking mode is active.

    This is intentionally slow — used only for rule synthesis and merge decisions.
    max_tokens defaults to THINKING_MAX_TOKENS (8192) to give Qwen3's full
    <think>...</think> trace enough room before the JSON output.
    """
    from llm_provider import predict_chat
    messages = [{"role": "user", "content": prompt}]
    result = predict_chat(messages=messages, max_tokens=max_tokens, temperature=0.3)
    return result.get("text", "")


# ── Step 3: Prune / merge ─────────────────────────────────────────────────────

def _word_overlap(text_a: str, text_b: str) -> float:
    """Crude word-overlap similarity between two rule text strings (Jaccard-ish)."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _fetch_active_rules_full(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Fetch all active rules as list of dicts with performance stats.

    Also fetches verdict_direction and trigger_feature/trigger_bucket if present,
    to support the contradiction check in Step 3b.
    """
    cursor = conn.cursor()
    # Detect whether verdict_direction and trigger columns exist (added by startup migration)
    cursor.execute("PRAGMA table_info(rules);")
    col_names = {row[1] for row in cursor.fetchall()}
    has_direction = "verdict_direction" in col_names
    has_trigger_feature = "trigger_feature" in col_names
    has_trigger_bucket = "trigger_bucket" in col_names

    extras = ""
    if has_direction:
        extras += ", verdict_direction"
    if has_trigger_feature:
        extras += ", trigger_feature"
    if has_trigger_bucket:
        extras += ", trigger_bucket"

    cursor.execute(
        f"""
        SELECT id, rule_text, times_applied, times_correct{extras}
        FROM rules
        WHERE active = 1
        ORDER BY
            (CAST(times_correct AS REAL) / NULLIF(times_applied, 0)) ASC NULLS FIRST,
            times_applied ASC
        """
    )
    base_keys = ["id", "rule_text", "times_applied", "times_correct"]
    extra_keys = []
    if has_direction:
        extra_keys.append("verdict_direction")
    if has_trigger_feature:
        extra_keys.append("trigger_feature")
    if has_trigger_bucket:
        extra_keys.append("trigger_bucket")
    all_keys = base_keys + extra_keys

    results = []
    for row in cursor.fetchall():
        d = dict(zip(all_keys, row))
        d["times_applied"] = d["times_applied"] or 0
        d["times_correct"] = d["times_correct"] or 0
        results.append(d)
    return results


# ── Contradiction detection helpers ───────────────────────────────────────────

def _patterns_overlap(cand_feature: str, cand_bucket: str,
                      exist_feature: Optional[str], exist_bucket: Optional[str]) -> bool:
    """Return True if the candidate cluster's trigger pattern overlaps an existing rule's.

    Overlap means same feature AND same bucket (exact match for categorical;
    same quartile label for numeric). If the existing rule has no stored trigger
    (pre-migration rule), we cannot determine overlap — return False conservatively.
    """
    if not exist_feature or not exist_bucket:
        return False
    return exist_feature == cand_feature and exist_bucket == cand_bucket


def _opposite_directions(dir_a: Optional[str], dir_b: Optional[str]) -> bool:
    """Return True if the two verdict_direction values are opposites."""
    if not dir_a or not dir_b:
        return False
    return dir_a != dir_b and {dir_a, dir_b} == {"favor_skip", "favor_keep"}


def _prune_underperformers(
    conn: sqlite3.Connection,
    rule_changes: List[Dict[str, Any]],
) -> int:
    """Deactivate rules that have enough applications but low accuracy.

    Returns count of rules pruned.
    """
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT id, rule_text, times_applied, times_correct
        FROM rules
        WHERE active = 1
          AND times_applied >= {PRUNE_MIN_APPLIED}
        """
    )
    pruned = 0
    for row in cursor.fetchall():
        rule_id, rule_text, applied, correct = row
        applied = applied or 0
        correct = correct or 0
        accuracy = correct / applied if applied > 0 else 0.0
        if accuracy < PRUNE_MAX_ACCURACY:
            reason = f"{correct}/{applied} correct ({accuracy*100:.0f}%) — below {int(PRUNE_MAX_ACCURACY*100)}% threshold"
            conn.execute(
                "UPDATE rules SET active = 0, retirement_reason = ? WHERE id = ?;",
                (reason, rule_id),
            )
            rule_changes.append({"rule_id": rule_id, "action": "pruned", "reason": reason})
            print(f"  [PRUNE] Rule #{rule_id}: {reason}")
            pruned += 1
    return pruned


def _enforce_cap(
    conn: sqlite3.Connection,
    slots_needed: int,
    rule_changes: List[Dict[str, Any]],
) -> None:
    """If adding slots_needed rules would exceed MAX_ACTIVE_RULES, prune lowest performers first."""
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM rules WHERE active = 1;")
    current_active = cursor.fetchone()[0]
    overflow = (current_active + slots_needed) - MAX_ACTIVE_RULES
    if overflow <= 0:
        return

    print(f"  [CAP] Enforcing cap: {current_active} active + {slots_needed} new > {MAX_ACTIVE_RULES}. Pruning {overflow} lowest performers...")
    # Fetch worst performers (lowest accuracy, then fewest applications)
    cursor.execute(
        """
        SELECT id, rule_text, times_applied, times_correct
        FROM rules
        WHERE active = 1
        ORDER BY
            (CAST(times_correct AS REAL) / NULLIF(times_applied, 0)) ASC NULLS FIRST,
            times_applied ASC
        """
    )
    for row in cursor.fetchall():
        if overflow <= 0:
            break
        rule_id, rule_text, applied, correct = row
        applied = applied or 0
        correct = correct or 0
        reason = f"Pruned to enforce {MAX_ACTIVE_RULES}-rule cap"
        conn.execute(
            "UPDATE rules SET active = 0, retirement_reason = ? WHERE id = ?;",
            (reason, rule_id),
        )
        rule_changes.append({"rule_id": rule_id, "action": "pruned", "reason": reason})
        print(f"    [CAP-PRUNE] Rule #{rule_id}: {reason}")
        overflow -= 1


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def _compute_accuracy(batch_results: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Compute overall accuracy and skip recall from batch_results.

    Returns (accuracy_before, skip_recall_before).
    """
    if not batch_results:
        return 0.0, 0.0

    correct = sum(
        1 for r in batch_results
        if r.get("predicted_verdict", "").lower() == r.get("actual_label", "").lower()
    )
    accuracy = correct / len(batch_results)

    actual_skips = [r for r in batch_results if r.get("actual_label", "").lower() == "skip"]
    if actual_skips:
        correctly_skipped = sum(
            1 for r in actual_skips
            if r.get("predicted_verdict", "").lower() == "skip"
        )
        skip_recall = correctly_skipped / len(actual_skips)
    else:
        skip_recall = 0.0

    return round(accuracy, 4), round(skip_recall, 4)


def _ensure_rules_columns(conn: sqlite3.Connection) -> None:
    """Ensure verdict_direction, trigger_feature, and trigger_bucket columns exist in rules table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(rules);")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col_name, col_type in [
        ("verdict_direction", "TEXT"),
        ("trigger_feature", "TEXT"),
        ("trigger_bucket", "TEXT"),
    ]:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE rules ADD COLUMN {col_name} {col_type};")
    conn.commit()


# ── Main public function ──────────────────────────────────────────────────────

def consolidate_rules(
    batch_results: List[Dict[str, Any]],
    conn: sqlite3.Connection,
) -> Dict[str, Any]:
    """Run the full self-calibration pipeline for one batch of prediction results.

    Steps:
        1. Cluster misses by shared feature patterns (deterministic).
        2. Synthesize candidate rules from qualifying clusters (LLM, thinking mode).
        3. Prune underperformers; merge near-duplicates (LLM for wording).
        4. Write changes to DB, insert runs row.

    Args:
        batch_results: List of per-track prediction result dicts (see module docstring).
        conn:          Open sqlite3 connection. Caller owns lifecycle.

    Returns:
        Summary dict matching the runs.summary JSON contract.
    """
    # Ensure rules table has verdict_direction and trigger columns
    _ensure_rules_columns(conn)

    print("\n" + "=" * 70)
    print(f"consolidate_rules: processing batch of {len(batch_results)} results")
    print("=" * 70)

    accuracy_before, skip_recall_before = _compute_accuracy(batch_results)
    label_ids = [r["label_id"] for r in batch_results if r.get("label_id") is not None]
    rules_before_count = conn.execute("SELECT count(*) FROM rules WHERE active = 1;").fetchone()[0]

    print(f"  Accuracy before : {accuracy_before*100:.1f}%  |  Skip recall: {skip_recall_before*100:.1f}%")
    print(f"  Active rules    : {rules_before_count}")

    rule_changes: List[Dict[str, Any]] = []

    # ── Insert runs row early to get run_id ───────────────────────────────────
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO runs (run_type, rules_before, rules_after, accuracy_before, skip_recall_before, label_ids, summary, created_at)
        VALUES ('consolidation', ?, NULL, ?, ?, ?, NULL, datetime('now'));
        """,
        (
            rules_before_count,
            accuracy_before,
            skip_recall_before,
            json.dumps(label_ids),
        ),
    )
    conn.commit()
    run_id = cursor.lastrowid
    print(f"  Run ID          : {run_id}")

    # ── Step 1: Cluster misses ────────────────────────────────────────────────
    misses = [
        r for r in batch_results
        if r.get("predicted_verdict", "").lower() != r.get("actual_label", "").lower()
    ]
    print(f"\n[Step 1] Miss clustering: {len(misses)}/{len(batch_results)} misses in batch")

    clusters = _cluster_misses(misses, conn)
    print(f"  Found {len(clusters)} qualifying cluster(s) (threshold: {MISS_CLUSTER_THRESHOLD}+ misses):")
    for c in clusters:
        print(f"    • Pattern: {c['pattern_description']}  ({len(c['misses'])} misses)")

    # ── Step 2: LLM rule synthesis ────────────────────────────────────────────
    print(f"\n[Step 2] Synthesizing candidate rules from {len(clusters)} cluster(s) (LLM, thinking mode).")
    print(f"         max_tokens={THINKING_MAX_TOKENS}  temperature=0.3  (no /no_think — thinking enabled)")

    # candidate_rules: list of dicts carrying rule_text, verdict_direction, and
    # the triggering cluster's feature+bucket for contradiction detection in Step 3b.
    candidate_rules: List[Dict[str, Any]] = []
    for i, cluster in enumerate(clusters, 1):
        print(f"  [{i}/{len(clusters)}] Cluster: {cluster['pattern_description']} ...")
        print(f"         (LLM thinking — this may take several minutes on this hardware)")
        synthesis_prompt = _build_synthesis_prompt(cluster)
        t0 = time.perf_counter()
        raw = _llm_think(synthesis_prompt, max_tokens=THINKING_MAX_TOKENS)
        elapsed = time.perf_counter() - t0
        parsed = _parse_llm_json(raw, required_key="rule_text")
        if parsed:
            rule_text = parsed["rule_text"].strip()
            direction = parsed.get("verdict_direction", "").strip().lower()
            if direction not in ("favor_skip", "favor_keep"):
                direction = "unknown"
            print(f"         → Rule text: \"{rule_text}\" ({elapsed:.1f}s)")
            print(f"           Direction: {direction}  |  Confidence: {parsed.get('confidence', '?')}  |  Rationale: {parsed.get('rationale', '')[:80]}")
            candidate_rules.append({
                "rule_text": rule_text,
                "verdict_direction": direction,
                "trigger_feature": cluster["feature"],
                "trigger_bucket": cluster["bucket"],
            })
        else:
            print(f"         → Could not parse LLM output ({elapsed:.1f}s). Raw (first 300 chars):\n{raw[:300]!r}")

    # ── Step 3a: Prune underperformers ────────────────────────────────────────
    print(f"\n[Step 3a] Pruning underperforming rules (min {PRUNE_MIN_APPLIED} applications, < {int(PRUNE_MAX_ACCURACY*100)}% accuracy)...")
    pruned = _prune_underperformers(conn, rule_changes)
    if pruned == 0:
        print("  No rules meet the pruning criteria.")

    # ── Step 3b: Contradiction check + near-duplicate merge + insert ─────────
    # Contradiction check runs BEFORE word-overlap merge — two rules on the same
    # feature+bucket with opposite verdict_directions must not coexist.
    #
    # Newest-evidence-wins assumption: the candidate (fresh batch evidence) always
    # supersedes the existing conflicting rule, regardless of the existing rule's
    # track record, because the existing rule has no performance baseline to compare
    # against the candidate which has none at all. Revisit if rules start flip-
    # flopping every round (symptom: same trigger alternates direction repeatedly).
    print(f"\n[Step 3b] Checking {len(candidate_rules)} candidate rule(s) against existing rules "
          f"for contradictions and near-duplicates...")
    active_rules = _fetch_active_rules_full(conn)

    final_new_rules: List[Dict[str, Any]] = []
    for candidate in candidate_rules:
        candidate_text = candidate["rule_text"]
        cand_direction = candidate["verdict_direction"]
        cand_feature   = candidate["trigger_feature"]
        cand_bucket    = candidate["trigger_bucket"]

        # ── Sub-check A: Contradiction (same pattern, opposite direction) ─────
        contradicted_rule = None
        for existing in active_rules:
            if _patterns_overlap(cand_feature, cand_bucket,
                                 existing.get("trigger_feature"),
                                 existing.get("trigger_bucket")) and \
               _opposite_directions(cand_direction, existing.get("verdict_direction")):
                contradicted_rule = existing
                break  # retire only the first match; further rounds will catch any others

        if contradicted_rule:
            # Insert new rule first to get its ID
            cursor.execute(
                """
                INSERT INTO rules (rule_text, active, times_applied, times_correct,
                                   created_by_run_id, verdict_direction,
                                   trigger_feature, trigger_bucket,
                                   created_at, updated_at)
                VALUES (?, 1, 0, 0, ?, ?, ?, ?, datetime('now'), datetime('now'));
                """,
                (candidate_text, run_id, cand_direction or None,
                 cand_feature or None, cand_bucket or None),
            )
            new_rule_id = cursor.lastrowid

            # Retire contradicted rule — newest evidence wins
            contradiction_reason = (
                f"Contradicted by newer rule #{new_rule_id} from run #{run_id}: "
                f"opposite verdict on same trigger pattern "
                f"({cand_feature}={cand_bucket})"
            )
            cursor.execute(
                "UPDATE rules SET active = 0, superseded_by = ?, retirement_reason = ? WHERE id = ?;",
                (new_rule_id, contradiction_reason, contradicted_rule["id"]),
            )
            print(
                f"  [CONTRADICTION] New rule #{new_rule_id} conflicts with active rule "
                f"#{contradicted_rule['id']} (same trigger: {cand_feature}={cand_bucket}, "
                f"opposite verdicts: {cand_direction} vs {contradicted_rule.get('verdict_direction', '?')}). "
                f"Retiring rule #{contradicted_rule['id']}, keeping rule #{new_rule_id}."
            )
            rule_changes.append({
                "rule_id": contradicted_rule["id"],
                "action": "contradicted",
                "by_rule_id": new_rule_id,
                "reason": contradiction_reason,
            })
            rule_changes.append({
                "rule_id": new_rule_id,
                "action": "created",
                "reason": f"Replaced contradicting rule #{contradicted_rule['id']} on trigger {cand_feature}={cand_bucket}",
            })
            active_rules = [r for r in active_rules if r["id"] != contradicted_rule["id"]]
            active_rules.append({
                "id": new_rule_id, "rule_text": candidate_text,
                "times_applied": 0, "times_correct": 0,
                "verdict_direction": cand_direction,
                "trigger_feature": cand_feature, "trigger_bucket": cand_bucket,
            })
            continue  # skip word-overlap merge check for this candidate

        # ── Sub-check B: Word-overlap merge (same direction, similar wording) ─
        overlapping_rule = None
        best_overlap = 0.0
        for existing in active_rules:
            sim = _word_overlap(candidate_text, existing["rule_text"])
            if sim > RULE_SIMILARITY_WORD_OVERLAP and sim > best_overlap:
                overlapping_rule = existing
                best_overlap = sim

        if overlapping_rule:
            print(f"  [MERGE] Candidate overlaps rule #{overlapping_rule['id']} (similarity={best_overlap:.2f})")
            print(f"    Existing : \"{overlapping_rule['rule_text']}\"")
            print(f"    Candidate: \"{candidate_text}\"")
            print(f"    Calling LLM to merge (thinking mode, max_tokens={THINKING_MAX_TOKENS})...")
            merge_prompt = _build_merge_prompt(overlapping_rule["rule_text"], candidate_text)
            t0 = time.perf_counter()
            raw = _llm_think(merge_prompt, max_tokens=THINKING_MAX_TOKENS)
            elapsed = time.perf_counter() - t0
            parsed = _parse_llm_json(raw, required_key="rule_text")
            if parsed:
                merged_text = parsed["rule_text"].strip()
                merged_direction = parsed.get("verdict_direction", "").strip().lower()
                if merged_direction not in ("favor_skip", "favor_keep"):
                    merged_direction = cand_direction  # inherit candidate's direction on parse gap
                print(f"    → Merged rule: \"{merged_text}\" direction={merged_direction} ({elapsed:.1f}s)")

                # Insert merged rule
                cursor.execute(
                    """
                    INSERT INTO rules (rule_text, active, times_applied, times_correct,
                                       created_by_run_id, verdict_direction,
                                       trigger_feature, trigger_bucket,
                                       created_at, updated_at)
                    VALUES (?, 1, 0, 0, ?, ?, ?, ?, datetime('now'), datetime('now'));
                    """,
                    (merged_text, run_id, merged_direction or None,
                     cand_feature or None, cand_bucket or None),
                )
                new_rule_id = cursor.lastrowid

                # Retire old rule, point to new one
                merge_reason = f"Merged into rule #{new_rule_id}: overlapped with candidate ({best_overlap:.0%} similarity)"
                cursor.execute(
                    "UPDATE rules SET active = 0, superseded_by = ?, retirement_reason = ? WHERE id = ?;",
                    (new_rule_id, merge_reason, overlapping_rule["id"]),
                )
                rule_changes.append({
                    "rule_id": overlapping_rule["id"],
                    "action": "merged",
                    "into_rule_id": new_rule_id,
                    "reason": merge_reason,
                })
                rule_changes.append({
                    "rule_id": new_rule_id,
                    "action": "created",
                    "reason": f"Merged from rule #{overlapping_rule['id']} and candidate cluster",
                })
                # Update local active_rules list
                active_rules = [r for r in active_rules if r["id"] != overlapping_rule["id"]]
                active_rules.append({
                    "id": new_rule_id, "rule_text": merged_text,
                    "times_applied": 0, "times_correct": 0,
                    "verdict_direction": merged_direction,
                    "trigger_feature": cand_feature, "trigger_bucket": cand_bucket,
                })
            else:
                print(f"    → Merge LLM parse failed ({elapsed:.1f}s). Inserting candidate as-is.")
                final_new_rules.append(candidate)
        else:
            final_new_rules.append(candidate)

    # ── Step 3c: Cap enforcement + insert remaining candidates ────────────────
    if final_new_rules:
        _enforce_cap(conn, slots_needed=len(final_new_rules), rule_changes=rule_changes)
        for cand in final_new_rules:
            cursor.execute(
                """
                INSERT INTO rules (rule_text, active, times_applied, times_correct,
                                   created_by_run_id, verdict_direction,
                                   trigger_feature, trigger_bucket,
                                   created_at, updated_at)
                VALUES (?, 1, 0, 0, ?, ?, ?, ?, datetime('now'), datetime('now'));
                """,
                (cand["rule_text"], run_id,
                 cand.get("verdict_direction") or None,
                 cand.get("trigger_feature") or None,
                 cand.get("trigger_bucket") or None),
            )
            new_id = cursor.lastrowid
            rule_changes.append({
                "rule_id": new_id,
                "action": "created",
                "reason": "Synthesized from miss cluster",
            })
            print(f"  [CREATE] Rule #{new_id}: \"{cand['rule_text']}\" (direction={cand.get('verdict_direction', '?')})")

    conn.commit()

    # ── Step 4: Finalize runs row ─────────────────────────────────────────────
    rules_after_count = conn.execute("SELECT count(*) FROM rules WHERE active = 1;").fetchone()[0]
    summary = {"rule_changes": rule_changes}
    cursor.execute(
        "UPDATE runs SET rules_after = ?, summary = ? WHERE id = ?;",
        (rules_after_count, json.dumps(summary), run_id),
    )
    conn.commit()

    print(f"\n[Step 4] DB writes complete.")
    print(f"  Rules before     : {rules_before_count}")
    print(f"  Rules after      : {rules_after_count}")
    print(f"  Changes recorded : {len(rule_changes)}")
    print(f"  Run ID           : {run_id}")
    print("=" * 70 + "\n")

    # Include run_id in return so calibrate.py can mark labels as used
    summary["run_id"] = run_id
    summary["rules_before"] = rules_before_count
    summary["rules_after"] = rules_after_count
    return summary


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DB_PATH = Path("skip_predictor.db")
    if not DB_PATH.exists():
        print(f"[!] Database '{DB_PATH}' not found. Run init_db.py first.", file=sys.stderr)
        sys.exit(1)

    # ── Synthetic batch: 15 tracks, deliberately skewed misses on genre and user_playcount ──
    SYNTHETIC_BATCH: List[Dict[str, Any]] = [
        # 6 misses on genre="hip hop" tracks (should have been Kept, model predicted Skip)
        {
            "track_id": 100 + i,
            "predicted_verdict": "Skip",
            "actual_label": "keep",
            "confidence": 55,
            "reasoning": "Low co-occurrence and unfamiliar genre made me lean Skip, but user may enjoy this style",
            "genre": "hip hop",
            "release_year": 2021,
            "album_type": "album",
            "days_since_added": 400 + i * 10,
            "artist_cooccurrence_score": 0.02,
            "user_playcount": 0,
            "label_id": None,
        }
        for i in range(6)
    ] + [
        # 6 misses on user_playcount=0 tracks (should have been Skipped, model predicted Keep)
        {
            "track_id": 200 + i,
            "predicted_verdict": "Keep",
            "actual_label": "skip",
            "confidence": 60,
            "reasoning": "The track is recent and from a known album, so I predicted Keep, but zero personal plays suggest this doesn't resonate",
            "genre": "pop",
            "release_year": 2022,
            "album_type": "single",
            "days_since_added": 500 + i * 5,
            "artist_cooccurrence_score": 0.01,
            "user_playcount": 0,
            "label_id": None,
        }
        for i in range(6)
    ] + [
        # 3 correct predictions (mix)
        {
            "track_id": 300,
            "predicted_verdict": "Keep",
            "actual_label": "keep",
            "confidence": 82,
            "reasoning": "High co-occurrence and frequent personal plays strongly indicate this is a favourite",
            "genre": "rap",
            "release_year": 2020,
            "album_type": "album",
            "days_since_added": 900,
            "artist_cooccurrence_score": 0.09,
            "user_playcount": 15,
            "label_id": None,
        },
        {
            "track_id": 301,
            "predicted_verdict": "Skip",
            "actual_label": "skip",
            "confidence": 77,
            "reasoning": "Very old addition, zero plays, and low co-occurrence all strongly suggest stale filler",
            "genre": "indie",
            "release_year": 2015,
            "album_type": "album",
            "days_since_added": 2000,
            "artist_cooccurrence_score": 0.003,
            "user_playcount": 0,
            "label_id": None,
        },
        {
            "track_id": 302,
            "predicted_verdict": "Keep",
            "actual_label": "keep",
            "confidence": 91,
            "reasoning": "Frequently played and highly co-occurring artist — clear favourite",
            "genre": "r&b",
            "release_year": 2023,
            "album_type": "album",
            "days_since_added": 310,
            "artist_cooccurrence_score": 0.08,
            "user_playcount": 22,
            "label_id": None,
        },
    ]

    print("=" * 70)
    print("consolidate_rules.py — Smoke Test")
    print("=" * 70)
    print(f"Synthetic batch: {len(SYNTHETIC_BATCH)} tracks")
    misses_count = sum(1 for r in SYNTHETIC_BATCH if r["predicted_verdict"].lower() != r["actual_label"].lower())
    print(f"Misses in batch: {misses_count}")
    print()

    # Show clustering before running DB writes
    conn_check = sqlite3.connect(DB_PATH)
    clusters = _cluster_misses(
        [r for r in SYNTHETIC_BATCH if r["predicted_verdict"].lower() != r["actual_label"].lower()],
        conn_check,
    )
    print(f"[Preview] Miss clusters found: {len(clusters)}")
    for c in clusters:
        print(f"  • {c['pattern_description']} — {len(c['misses'])} misses")
    conn_check.close()

    print()
    print("Synthesis prompts that would be sent to LLM (thinking mode):")
    for i, c in enumerate(clusters, 1):
        prompt_preview = _build_synthesis_prompt(c)
        print(f"\n--- Cluster {i}: {c['pattern_description']} ---")
        print(prompt_preview[:400] + ("..." if len(prompt_preview) > 400 else ""))

    print()
    print("[!] LLM inference not triggered in smoke test to avoid loading the model.")
    print("    Clustering, prompt construction, and DB path verified successfully.")
    print("    Run as part of calibrate.py to execute the full pipeline with real model calls.")
    print()
    print("[OK] Smoke test complete.")
