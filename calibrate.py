"""Outer self-calibration loop — ties together prompts, LLM inference, and rule consolidation.

Startup migration:
    On every startup, this script checks whether the 'labels' table has a
    'used_in_run_id' column and adds it if missing (idempotent, same
    PRAGMA table_info pattern as migrate_phase3_schema.py). This migration
    happens before any other DB operation.

Workflow (per round):
    1. Pull a fresh batch of unused labeled tracks (used_in_run_id IS NULL).
    2. Build the prediction prompt for each track via prompts.build_prediction_prompt().
    3. Call llm_provider.predict_chat() with the prompt.
    4. Parse the model's JSON response; skip malformed responses gracefully.
    5. Build batch_results in the shape consolidate_rules expects.
    6. Call consolidate_rules(batch_results, conn) → summary dict.
    7. Mark labels as used (used_in_run_id = run_id from summary).
    8. Print per-round summary.

Usage:
    python3 calibrate.py
    python3 calibrate.py --batch-size 25 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from consolidate_rules import consolidate_rules, _compute_accuracy
from llm_provider import predict_chat
from migrate_phase3_schema import migrate as migrate_phase3
from prompts import build_prediction_prompt

DEFAULT_DB_PATH = config.DB_PATH
DEFAULT_BATCH_SIZE = 25


# ── Schema migration ──────────────────────────────────────────────────────────

def _ensure_used_in_run_id_column(conn: sqlite3.Connection) -> None:
    """Add used_in_run_id INTEGER (nullable) to labels if it does not already exist.

    Idempotent — safe to run on every startup. Uses the same PRAGMA table_info
    check pattern as migrate_phase3_schema.py.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(labels);")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "used_in_run_id" not in existing_cols:
        print("[calibrate] Migrating labels table: adding 'used_in_run_id' column...")
        cursor.execute("ALTER TABLE labels ADD COLUMN used_in_run_id INTEGER;")
        conn.commit()
        print("[calibrate] Migration complete.")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_unused_labels(
    conn: sqlite3.Connection, batch_size: int
) -> List[Dict[str, Any]]:
    """Pull labeled tracks with their features that have not yet been used in a run.

    Joins labels → tracks → features and returns all fields needed for
    batch_results. Only rows where used_in_run_id IS NULL are returned.
    """
    cursor = conn.cursor()

    # Detect whether user_playcount column exists in features
    cursor.execute("PRAGMA table_info(features);")
    feature_cols = {row["name"] for row in cursor.fetchall()}
    has_user_playcount = "user_playcount" in feature_cols
    user_pc_sel = "f.user_playcount" if has_user_playcount else "NULL AS user_playcount"

    cursor.execute(
        f"""
        SELECT
            l.id          AS label_id,
            l.track_id,
            l.label       AS actual_label,
            t.title,
            t.artist,
            f.genre,
            f.release_year,
            f.album_type,
            f.days_since_added,
            f.artist_cooccurrence_score,
            {user_pc_sel}
        FROM labels l
        JOIN tracks t ON t.id = l.track_id
        JOIN features f ON f.track_id = l.track_id
        WHERE l.used_in_run_id IS NULL
        ORDER BY l.id ASC
        LIMIT ?;
        """,
        (batch_size,),
    )
    return [dict(row) for row in cursor.fetchall()]


def _count_unused_labels(conn: sqlite3.Connection) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM labels WHERE used_in_run_id IS NULL;")
    return cursor.fetchone()[0]


def _mark_labels_used(
    conn: sqlite3.Connection, label_ids: List[int], run_id: int
) -> None:
    """Set used_in_run_id on all labels in this batch."""
    if not label_ids:
        return
    placeholders = ",".join("?" * len(label_ids))
    conn.execute(
        f"UPDATE labels SET used_in_run_id = ? WHERE id IN ({placeholders});",
        [run_id] + label_ids,
    )
    conn.commit()


# ── JSON response parsing ─────────────────────────────────────────────────────

def _parse_model_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse verdict/confidence/reasoning from model JSON output.

    Returns None if the response cannot be parsed, rather than crashing.
    Strips markdown code fences and finds the first { ... } block.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    verdict = parsed.get("verdict", "")
    if verdict not in ("Keep", "Skip"):
        # Handle lowercase variants
        verdict = verdict.capitalize()
        if verdict not in ("Keep", "Skip"):
            return None

    return {
        "verdict": verdict,
        "confidence": int(parsed.get("confidence", 50)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ── Prediction step ───────────────────────────────────────────────────────────

def _predict_batch(
    labeled_rows: List[Dict[str, Any]],
    conn: sqlite3.Connection,
    debug_raw: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Run LLM predictions for each labeled track.

    Args:
        labeled_rows: Track+label rows from the DB.
        conn:         Open DB connection for prompt building.
        debug_raw:    If True, prints the full raw model response before parsing.
                      Use to verify confidence values and response format.

    Returns:
        (batch_results, skipped_label_ids)
        batch_results : list of dicts in the shape consolidate_rules expects.
        skipped_label_ids : label IDs whose responses could not be parsed
                            (still marked used to avoid re-trying bad data).
    """
    batch_results: List[Dict[str, Any]] = []
    skipped_label_ids: List[int] = []
    total = len(labeled_rows)

    for idx, row in enumerate(labeled_rows, 1):
        label_id = row["label_id"]
        track_id = row["track_id"]
        title = row.get("title", "Unknown")
        artist = row.get("artist", "Unknown")

        print(
            f"  [Predicting {idx:>2}/{total}] \"{title}\" — {artist} ...",
            end="",
            flush=True,
        )

        # Build prompt — prompts.py handles /no_think and rule injection
        prompt_str = build_prediction_prompt(row, conn)
        messages = [{"role": "user", "content": prompt_str}]

        try:
            t0 = time.perf_counter()
            pred = predict_chat(
                messages=messages,
                max_tokens=config.DEFAULT_MAX_TOKENS,
                temperature=config.TEMPERATURE,
            )
            elapsed = time.perf_counter() - t0
            raw_text = pred.get("text", "")
        except Exception as e:
            print(f" ERROR calling model: {e}")
            skipped_label_ids.append(label_id)
            continue

        if debug_raw:
            print(f"\n    [RAW RESPONSE label_id={label_id}]\n{raw_text}\n    [END RAW]")

        parsed = _parse_model_response(raw_text)
        if parsed is None:
            print(f" PARSE FAIL (label_id={label_id}) — raw: {raw_text[:80]!r}")
            skipped_label_ids.append(label_id)
            continue

        verdict = parsed["verdict"]
        actual = row["actual_label"]  # "keep" or "skip" from labels table
        correct = verdict.lower() == actual.lower()

        print(
            f" → {verdict:<4}  (actual: {actual.capitalize()}) "
            f"{'✓' if correct else '✗'}  conf={parsed['confidence']}  {elapsed:.1f}s"
        )

        batch_results.append({
            "track_id": track_id,
            "label_id": label_id,
            "predicted_verdict": verdict,
            "actual_label": actual,
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "genre": row.get("genre"),
            "release_year": row.get("release_year"),
            "album_type": row.get("album_type"),
            "days_since_added": row.get("days_since_added"),
            "artist_cooccurrence_score": row.get("artist_cooccurrence_score"),
            "user_playcount": row.get("user_playcount"),
        })

    return batch_results, skipped_label_ids


# ── Round summary ─────────────────────────────────────────────────────────────

def _print_round_summary(
    round_num: int,
    batch_results: List[Dict[str, Any]],
    skipped_count: int,
    summary: Dict[str, Any],
    elapsed_sec: float,
) -> None:
    accuracy, skip_recall = _compute_accuracy(batch_results)
    rule_changes = summary.get("rule_changes", [])
    created = sum(1 for c in rule_changes if c["action"] == "created")
    pruned  = sum(1 for c in rule_changes if c["action"] == "pruned")
    merged  = sum(1 for c in rule_changes if c["action"] == "merged")
    active_after = summary.get("rules_after", "?")
    run_id = summary.get("run_id", "?")

    mins = int(elapsed_sec // 60)
    secs = int(elapsed_sec % 60)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Round {round_num} Summary".ljust(59) + "║")
    print("╠══════════════════════════════════════════════════════════╣")

    def row(label: str, val: str) -> None:
        line = f"  {label:<28}{val}"
        print(f"║{line:<58}║")

    row("Tracks predicted:", str(len(batch_results)))
    row("Skipped (parse errors):", str(skipped_count))
    row("Accuracy:", f"{accuracy * 100:.1f}%")
    row("Skip recall:", f"{skip_recall * 100:.1f}%")
    row("Rules created:", str(created))
    row("Rules merged:", str(merged))
    row("Rules pruned:", str(pruned))
    row("Active rules after:", str(active_after))
    row("Run ID:", str(run_id))
    row("Round time:", f"{mins}m {secs:02d}s")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_calibration(
    db_path: Path,
    batch_size: int,
    num_rounds: int,
    debug_raw: bool = False,
) -> None:
    """Execute calibration rounds."""
    if not db_path.exists():
        print(f"[!] Database '{db_path}' not found. Run init_db.py first.", file=sys.stderr)
        sys.exit(1)

    # ── Ensure all Phase 3 columns (rules, runs) exist ───────────────────────
    migrate_phase3(db_path)

    conn = _get_connection(db_path)

    # ── One-time startup migration for labels ──────────────────────────────────
    _ensure_used_in_run_id_column(conn)

    rounds_completed = 0
    try:
        for round_num in range(1, num_rounds + 1):
            print()
            print(f"{'=' * 70}")
            print(f"CALIBRATION ROUND {round_num}/{num_rounds}  —  batch_size={batch_size}")
            print(f"{'=' * 70}")

            round_start = time.perf_counter()

            # ── 1. Check label availability BEFORE doing anything ─────────────
            unused_total = _count_unused_labels(conn)
            if unused_total < batch_size:
                print(
                    f"\n[!] Only {unused_total} unused label(s) remain, need {batch_size}.\n"
                    f"    Run label_cli.py to label more tracks, or lower --batch-size.\n"
                    f"    No DB writes were made. Stopping after {rounds_completed} completed round(s).",
                    file=sys.stderr,
                )
                sys.exit(1)

            # ── 2. Fetch batch ────────────────────────────────────────────────
            labeled_rows = _fetch_unused_labels(conn, batch_size)
            if len(labeled_rows) < batch_size:
                # Shouldn't happen if count check passed, but guard anyway
                print(
                    f"[!] Fetched only {len(labeled_rows)} rows (expected {batch_size}). "
                    f"DB may have changed mid-run. Aborting.",
                    file=sys.stderr,
                )
                sys.exit(1)

            label_ids_in_batch = [r["label_id"] for r in labeled_rows]
            print(f"[+] Fetched {len(labeled_rows)} labeled tracks. Starting predictions...\n")

            # ── 3. Predict ────────────────────────────────────────────────────
            batch_results, skipped_label_ids = _predict_batch(labeled_rows, conn, debug_raw=debug_raw)

            if not batch_results:
                print(
                    "[!] All predictions failed to parse — no valid batch_results. "
                    "Check model output format. Aborting round.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # ── 4. Consolidate rules ──────────────────────────────────────────
            # consolidate_rules handles its own DB writes inside a transaction.
            summary = consolidate_rules(batch_results, conn)
            run_id = summary.get("run_id")
            if run_id is None:
                print("[!] consolidate_rules did not return a run_id. Cannot mark labels. Aborting.", file=sys.stderr)
                sys.exit(1)

            # ── 5. Mark ALL labels in this batch as used ─────────────────────
            # Includes both successful predictions and parse-failed ones so they
            # are never silently retried with bad data.
            all_used_ids = list(set(label_ids_in_batch))
            _mark_labels_used(conn, all_used_ids, run_id)
            print(f"[+] Marked {len(all_used_ids)} label(s) as used (run_id={run_id}).")

            round_elapsed = time.perf_counter() - round_start
            rounds_completed += 1

            _print_round_summary(
                round_num=round_num,
                batch_results=batch_results,
                skipped_count=len(skipped_label_ids),
                summary=summary,
                elapsed_sec=round_elapsed,
            )

    finally:
        conn.close()

    print(f"[calibrate] Done. {rounds_completed}/{num_rounds} round(s) completed successfully.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-calibration loop: runs LLM predictions on labeled tracks, "
            "then calls consolidate_rules to update the active rule set."
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of labeled tracks to process per round (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of calibration rounds to run sequentially (default: 1)",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        default=False,
        help="Print full raw model response for every track before JSON parsing (for debugging confidence/format issues).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        print("[!] --batch-size must be at least 1.", file=sys.stderr)
        sys.exit(1)
    if args.rounds < 1:
        print("[!] --rounds must be at least 1.", file=sys.stderr)
        sys.exit(1)

    run_calibration(
        db_path=Path(args.db),
        batch_size=args.batch_size,
        num_rounds=args.rounds,
        debug_raw=args.debug_raw,
    )


if __name__ == "__main__":
    main()
