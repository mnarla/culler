"""Interactive terminal tool for manually labeling tracks as Keep or Skip.

Uses the same deterministic 50-track sample as benchmark.py (seed=42) so
labels align exactly with benchmark results. Resumable — skips already-labeled
tracks and picks up from the first unlabeled one.

Usage:
    python label_cli.py
    python label_cli.py --db skip_predictor.db --sample-size 50 --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_DB_PATH = "skip_predictor.db"
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42

# ── ANSI colour helpers ───────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BG_DARK = "\033[48;5;235m"


def _c(code: str, text: str) -> str:
    """Wrap text in ANSI code + reset."""
    return f"{code}{text}{RESET}"


def supports_color() -> bool:
    """Return True if the terminal likely supports ANSI color codes."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = supports_color()


def fmt(code: str, text: str) -> str:
    return _c(code, text) if USE_COLOR else text


# ── DB helpers ────────────────────────────────────────────────────────────────


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Connect to SQLite database with Row factory and FK enforcement."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def load_sample_tracks(
    db_path: str | Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Pull a deterministic sample of tracks joined with features.

    Identical logic to benchmark.py so track order matches exactly.
    user_playcount is read if the column exists; defaults to None otherwise.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Detect whether user_playcount column exists
    cursor.execute("PRAGMA table_info(features);")
    feature_cols = {row["name"] for row in cursor.fetchall()}
    has_user_playcount = "user_playcount" in feature_cols

    user_pc_sel = "f.user_playcount" if has_user_playcount else "NULL AS user_playcount"

    query = f"""
        SELECT
            t.id AS track_id,
            t.title,
            t.artist,
            t.album,
            t.playlist_name,
            f.genre,
            f.release_year,
            f.album_type,
            f.days_since_added,
            f.artist_cooccurrence_score,
            {user_pc_sel}
        FROM tracks t
        JOIN features f ON t.id = f.track_id
        ORDER BY t.id ASC;
    """
    cursor.execute(query)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    if not rows:
        print(fmt(RED, "[!] No tracks with features found in the database."), file=sys.stderr)
        sys.exit(1)

    if len(rows) < sample_size:
        sample_size = len(rows)

    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def get_labeled_track_ids(conn: sqlite3.Connection, track_ids: List[int]) -> Dict[int, str]:
    """Return {track_id: label} for any of the given track IDs already in labels table."""
    if not track_ids:
        return {}
    placeholders = ",".join("?" * len(track_ids))
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT track_id, label
        FROM labels
        WHERE track_id IN ({placeholders})
        ORDER BY created_at DESC
        """,
        track_ids,
    )
    # Deduplicate: keep the most recent label per track_id
    seen: Dict[int, str] = {}
    for row in cursor.fetchall():
        tid = row["track_id"]
        if tid not in seen:
            seen[tid] = row["label"]
    return seen


def upsert_label(
    conn: sqlite3.Connection,
    track_id: int,
    label: str,
    time_of_day: str,
    day_of_week: str,
    created_at: str,
) -> None:
    """Insert a new label row (or replace if already exists for this track_id)."""
    cursor = conn.cursor()
    # Delete any existing label for this track (re-run / undo scenario)
    cursor.execute("DELETE FROM labels WHERE track_id = ?;", (track_id,))
    cursor.execute(
        """
        INSERT INTO labels (track_id, label, time_of_day, day_of_week, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (track_id, label, time_of_day, day_of_week, created_at),
    )
    conn.commit()


def delete_label(conn: sqlite3.Connection, track_id: int) -> None:
    """Remove label entry for a track (used by Undo)."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM labels WHERE track_id = ?;", (track_id,))
    conn.commit()


# ── Display helpers ───────────────────────────────────────────────────────────


def clear_line() -> None:
    print("\r\033[K", end="", flush=True)


def print_header(current: int, total: int) -> None:
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = current / total * 100 if total > 0 else 0

    print()
    print(fmt(BOLD + CYAN, "╔══════════════════════════════════════════════════════════╗"))
    print(fmt(BOLD + CYAN, f"║") + fmt(BOLD + WHITE, f"  CULLER — Manual Labeler".ljust(58)) + fmt(BOLD + CYAN, "║"))
    progress_line = f"  Track {current}/{total}  [{bar}]  {pct:.0f}%"
    print(fmt(BOLD + CYAN, "║") + fmt(YELLOW, f"{progress_line:<58}") + fmt(BOLD + CYAN, "║"))
    print(fmt(BOLD + CYAN, "╚══════════════════════════════════════════════════════════╝"))


def print_track_card(track: Dict[str, Any], idx: int, total: int) -> None:
    print_header(idx, total)

    cooccur = track.get("artist_cooccurrence_score")
    cooccur_str = f"{cooccur:.4f}" if isinstance(cooccur, (int, float)) else "—"
    user_pc = track.get("user_playcount")
    user_pc_str = str(user_pc) if user_pc is not None else "—"

    print()
    print(fmt(BOLD + WHITE, f"  🎵  {track['title']}"))
    print(fmt(DIM, f"       by {track['artist']}"))
    if track.get("album"):
        print(fmt(DIM, f"       on {track['album']}"))
    print()
    print(fmt(CYAN, "  ┌─ Features ──────────────────────────────────────────┐"))
    rows_display = [
        ("Genre",                  track.get("genre") or "—"),
        ("Release Year",           str(track.get("release_year") or "—")),
        ("Album Type",             track.get("album_type") or "—"),
        ("Days Since Added",       str(track.get("days_since_added") if track.get("days_since_added") is not None else "—")),
        ("Artist Co-occurrence",   cooccur_str),
        ("Your Play Count",        user_pc_str),
    ]
    for label_col, val in rows_display:
        padded_label = f"{label_col}:".ljust(24)
        print(fmt(CYAN, "  │  ") + fmt(DIM, padded_label) + fmt(WHITE, val))
    print(fmt(CYAN, "  └─────────────────────────────────────────────────────┘"))
    print()
    print(
        fmt(GREEN, "  [K]") + fmt(WHITE, " Keep    ") +
        fmt(RED, "[S]") + fmt(WHITE, " Skip    ") +
        fmt(YELLOW, "[U]") + fmt(WHITE, " Undo    ") +
        fmt(DIM, "[Q] Quit")
    )
    print()


def get_time_context() -> Tuple[str, str]:
    """Return (time_of_day bucket, day_of_week) for the current moment."""
    now = datetime.datetime.now()
    hour = now.hour
    if 5 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 17:
        tod = "afternoon"
    elif 17 <= hour < 21:
        tod = "evening"
    else:
        tod = "night"
    dow = now.strftime("%A").lower()  # e.g. "monday"
    return tod, dow


def prompt_input(prompt_text: str) -> str:
    """Read a single line of input, returning stripped lowercase string."""
    try:
        return input(prompt_text).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


# ── Review mode ───────────────────────────────────────────────────────────────


def review_mode(
    conn: sqlite3.Connection,
    tracks: List[Dict[str, Any]],
    labeled: Dict[int, str],
) -> None:
    """Let the user view and optionally re-label already-labeled tracks."""
    print()
    print(fmt(BOLD + YELLOW, "  All 50 tracks are already labeled. Entering review mode."))
    print(fmt(DIM, "  Press Enter to step through each track, or type a new label (K/S) to change it. [Q] to quit.\n"))

    keep_count = sum(1 for v in labeled.values() if v == "keep")
    skip_count = sum(1 for v in labeled.values() if v == "skip")
    print(fmt(CYAN, f"  Current labels: {keep_count} Keep / {skip_count} Skip\n"))

    for idx, track in enumerate(tracks, 1):
        tid = track["track_id"]
        current_label = labeled.get(tid, "?")
        label_colored = fmt(GREEN, "Keep") if current_label == "keep" else fmt(RED, "Skip")

        print_track_card(track, idx, len(tracks))
        print(fmt(DIM, f"  Current label: {label_colored}"))
        print()

        ans = prompt_input("  Change label? (K/S to relabel, Enter to keep, Q to quit): ")
        if ans in ("q",):
            break
        elif ans in ("k", "keep"):
            tod, dow = get_time_context()
            ts = datetime.datetime.now().isoformat()
            upsert_label(conn, tid, "keep", tod, dow, ts)
            labeled[tid] = "keep"
            print(fmt(GREEN, "  ✓ Updated to Keep.\n"))
        elif ans in ("s", "skip"):
            tod, dow = get_time_context()
            ts = datetime.datetime.now().isoformat()
            upsert_label(conn, tid, "skip", tod, dow, ts)
            labeled[tid] = "skip"
            print(fmt(RED, "  ✓ Updated to Skip.\n"))
        else:
            print(fmt(DIM, "  — No change.\n"))


# ── Main labeling loop ────────────────────────────────────────────────────────


def run_labeler(
    db_path: str | Path = DEFAULT_DB_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    if not Path(db_path).exists():
        print(fmt(RED, f"[!] Database '{db_path}' not found. Run init_db.py first."), file=sys.stderr)
        sys.exit(1)

    conn = get_connection(db_path)
    tracks = load_sample_tracks(db_path=db_path, sample_size=sample_size, seed=seed)
    track_ids = [t["track_id"] for t in tracks]
    labeled = get_labeled_track_ids(conn, track_ids)

    # ── All already labeled → review mode ────────────────────────────────────
    if len(labeled) >= len(tracks):
        review_mode(conn, tracks, labeled)
        conn.close()
        return

    # ── Find first unlabeled index ────────────────────────────────────────────
    start_idx = 0
    for i, track in enumerate(tracks):
        if track["track_id"] not in labeled:
            start_idx = i
            break

    if start_idx > 0:
        print()
        print(fmt(CYAN, f"  ↩  Resuming session — {start_idx} track(s) already labeled. Starting from track {start_idx + 1}."))

    session_start = time.perf_counter()
    session_labels: Dict[int, str] = {}  # labels added this session

    i = start_idx
    while i < len(tracks):
        track = tracks[i]
        tid = track["track_id"]
        display_num = i + 1

        print_track_card(track, display_num, len(tracks))

        ans = prompt_input("  Your decision: ")

        if ans in ("q", "quit"):
            print()
            print(fmt(YELLOW, "  Progress saved. Run the script again to continue."))
            break

        elif ans in ("u", "undo"):
            if i == start_idx and not session_labels:
                print(fmt(DIM, "  Nothing to undo (this is the first track of this session)."))
                continue
            # Step back and remove previous label
            i -= 1
            prev_tid = tracks[i]["track_id"]
            delete_label(conn, prev_tid)
            if prev_tid in session_labels:
                del session_labels[prev_tid]
            if prev_tid in labeled:
                del labeled[prev_tid]
            print(fmt(YELLOW, f"  ↩  Undone. Re-labeling track {i + 1}.\n"))
            continue

        elif ans in ("k", "keep"):
            verdict = "keep"
        elif ans in ("s", "skip"):
            verdict = "skip"
        else:
            print(fmt(DIM, "  Unrecognized input. Please enter K, S, U, or Q.\n"))
            continue

        tod, dow = get_time_context()
        ts = datetime.datetime.now().isoformat()
        upsert_label(conn, tid, verdict, tod, dow, ts)
        labeled[tid] = verdict
        session_labels[tid] = verdict

        verdict_display = fmt(GREEN, "✓ Keep") if verdict == "keep" else fmt(RED, "✗ Skip")
        print(fmt(DIM, f"  {verdict_display}  saved.\n"))
        i += 1

    # ── Session summary ───────────────────────────────────────────────────────
    session_elapsed = time.perf_counter() - session_start
    all_labeled_now = get_labeled_track_ids(conn, track_ids)
    total_keep = sum(1 for v in all_labeled_now.values() if v == "keep")
    total_skip = sum(1 for v in all_labeled_now.values() if v == "skip")
    total_done = len(all_labeled_now)

    mins = int(session_elapsed // 60)
    secs = int(session_elapsed % 60)

    print()
    print(fmt(BOLD + CYAN, "╔══════════════════════════════════════════════════════════╗"))
    print(fmt(BOLD + CYAN, "║") + fmt(BOLD + WHITE, "  Session Summary".ljust(58)) + fmt(BOLD + CYAN, "║"))
    print(fmt(BOLD + CYAN, "╠══════════════════════════════════════════════════════════╣"))

    def summary_row(label: str, val: str) -> None:
        line = f"  {label:<28}{val}"
        print(fmt(BOLD + CYAN, "║") + fmt(WHITE, f"{line:<58}") + fmt(BOLD + CYAN, "║"))

    summary_row("Labeled this session:", str(len(session_labels)))
    summary_row("Total labeled so far:", f"{total_done}/{len(tracks)}")
    summary_row("Keep:", str(total_keep))
    summary_row("Skip:", str(total_skip))
    summary_row("Session duration:", f"{mins}m {secs:02d}s")

    if total_done >= len(tracks):
        print(fmt(BOLD + CYAN, "╠══════════════════════════════════════════════════════════╣"))
        complete_msg = "  🎉  All tracks labeled!"
        print(fmt(BOLD + CYAN, "║") + fmt(GREEN + BOLD, f"{complete_msg:<58}") + fmt(BOLD + CYAN, "║"))

    print(fmt(BOLD + CYAN, "╚══════════════════════════════════════════════════════════╝"))
    print()
    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive labeling tool — manually label tracks as Keep or Skip."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of tracks to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for track sampling (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    run_labeler(
        db_path=Path(args.db),
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
