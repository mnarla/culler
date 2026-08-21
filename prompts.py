"""Prompt builder for the Skip-Prediction calibration loop.

Constructs the full inference prompt for a single track, incorporating:
  - Formatted track features (identical layout to benchmark.py)
  - Active rules from the database, ranked by correctness rate
  - /no_think prefix (required for Qwen3 models)

JSON Output Contract:
    The model is instructed to respond with ONLY this JSON structure.
    No surrounding text, no markdown code fences. calibrate.py depends on
    parsing exactly this shape via json.loads():

    {
        "verdict": "Keep" | "Skip",
        "confidence": <integer 0-100>,
        "reasoning": "<1-2 sentence explanation referencing specific features>"
    }

    Fields:
        verdict    : "Keep" or "Skip" — mandatory.
        confidence : Integer 0–100 expressing model certainty — mandatory.
        reasoning  : Sentence(s) naming the specific features that drove the
                     verdict (e.g. "zero user playcount combined with a high
                     co-occurrence score suggests stale filler"). This field
                     is mined by consolidate_rules.py for error pattern
                     discovery, so generic filler is actively harmful.

    calibrate.py is responsible for parsing and validating this JSON.
    This module builds only the input prompt.

Usage:
    from prompts import build_prediction_prompt
    prompt = build_prediction_prompt(track_dict, conn)

Smoke test (prints a sample prompt against the live DB):
    python prompts.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Rule query ────────────────────────────────────────────────────────────────

_ACTIVE_RULES_QUERY = """
    SELECT id, rule_text
    FROM rules
    WHERE active = 1
    ORDER BY
        (CAST(times_correct AS REAL) / NULLIF(times_applied, 0)) DESC NULLS LAST,
        times_applied DESC
"""


def _fetch_active_rules(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    """Return list of (rule_id, rule_text) for all active rules, ranked by performance.

    Rules are sorted by correctness rate (times_correct / times_applied) descending,
    with ties broken by most-tested first, and unscored rules (NULL rate) last.
    Returns an empty list when no active rules exist.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(_ACTIVE_RULES_QUERY)
    except sqlite3.OperationalError:
        # Fallback if table has not been migrated yet
        cursor.execute("SELECT id, rule_text FROM rules WHERE active = 1 ORDER BY times_applied DESC;")
    return [(row[0], row[1]) for row in cursor.fetchall()]


# ── Feature formatting ────────────────────────────────────────────────────────
# Exact same logic as benchmark.py — keep in sync if benchmark.py changes.

def _format_features(track: Dict[str, Any]) -> str:
    """Format the 6 Phase 2 features into the standard feature block string."""
    cooccur = track.get("artist_cooccurrence_score")
    cooccur_str = f"{cooccur:.4f}" if isinstance(cooccur, (int, float)) else "Unknown"

    user_pc = track.get("user_playcount")
    user_pc_str = str(user_pc) if user_pc is not None else "Unknown"

    days = track.get("days_since_added")
    days_str = str(days) if days is not None else "Unknown"

    return (
        f"  - Genre: {track.get('genre') or 'Unknown'}\n"
        f"  - Release Year: {track.get('release_year') or 'Unknown'}\n"
        f"  - Album Type: {track.get('album_type') or 'Unknown'}\n"
        f"  - Days Since Added: {days_str}\n"
        f"  - Artist Co-occurrence Score: {cooccur_str}\n"
        f"  - Your Play Count: {user_pc_str}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def build_prediction_prompt(track: Dict[str, Any], conn: sqlite3.Connection) -> str:
    """Build the full prediction prompt for a track, with active rules injected.

    Args:
        track: Dict containing track metadata and features. Expected keys:
               title, artist, genre, release_year, album_type, days_since_added,
               artist_cooccurrence_score, user_playcount.
               Missing keys default to "Unknown" or 0 gracefully.
        conn:  Open sqlite3.Connection to skip_predictor.db. Used read-only
               to fetch active rules; caller owns the connection lifetime.

    Returns:
        Full prompt string ready to pass to llm_provider.predict_chat() as
        the user message content. Always prefixed with /no_think.

    Note:
        If no active rules exist, the "Guidelines" section is omitted entirely.
        The JSON output contract is described in this module's docstring.
    """
    active_rules = _fetch_active_rules(conn)
    feature_block = _format_features(track)

    # ── Core prompt ───────────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append(
        "You are an expert music playlist curator with deep knowledge of a listener's taste."
    )
    lines.append(
        "Evaluate the track below and decide whether it should be Kept or Skipped from the playlist."
    )
    lines.append("")

    # Track identity
    title = track.get("title") or "Unknown Track"
    artist = track.get("artist") or "Unknown Artist"
    lines.append(f'Track: "{title}" by {artist}')
    lines.append("")

    # Feature block
    lines.append("Features:")
    lines.append(feature_block)
    lines.append("")

    # ── Active rules block (omitted when empty) ───────────────────────────────
    if active_rules:
        lines.append("Guidelines from past feedback:")
        for i, (rule_id, rule_text) in enumerate(active_rules, 1):
            lines.append(f"  {i}. {rule_text.strip()}")
        lines.append("")

    # ── Output instruction ────────────────────────────────────────────────────
    lines.append(
        "Respond with ONLY a JSON object — no explanation, no preamble, no markdown fences. "
        "Use exactly this structure:"
    )
    lines.append("")
    lines.append('{"verdict": "Keep" | "Skip", "confidence": <integer 0-100>, "reasoning": "<1-2 sentences>"}')
    lines.append("")
    lines.append(
        'In "reasoning", name the specific feature(s) that drove your verdict '
        "(e.g. \"zero user playcount and high co-occurrence score suggest stale filler\"). "
        "Generic or vague reasoning is not acceptable."
    )

    prompt_body = "\n".join(lines)

    # /no_think is required for Qwen3 models and harmless for others
    return f"/no_think\n{prompt_body}"


# ── Smoke test ────────────────────────────────────────────────────────────────

def _fetch_sample_track(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Pull one real track with features from the database for smoke-testing."""
    cursor = conn.cursor()

    # Detect whether user_playcount column exists
    cursor.execute("PRAGMA table_info(features);")
    feature_cols = {row[1] for row in cursor.fetchall()}
    has_user_playcount = "user_playcount" in feature_cols

    user_pc_sel = "f.user_playcount" if has_user_playcount else "NULL AS user_playcount"

    cursor.execute(
        f"""
        SELECT
            t.id AS track_id,
            t.title,
            t.artist,
            f.genre,
            f.release_year,
            f.album_type,
            f.days_since_added,
            f.artist_cooccurrence_score,
            {user_pc_sel}
        FROM tracks t
        JOIN features f ON t.id = f.track_id
        LIMIT 1;
        """
    )
    row = cursor.fetchone()
    if not row:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


if __name__ == "__main__":
    db_path = Path("skip_predictor.db")
    if not db_path.exists():
        print(f"[!] Database '{db_path}' not found. Run init_db.py first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    track = _fetch_sample_track(conn)
    if not track:
        print("[!] No tracks with features found in the database.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    active_rules = _fetch_active_rules(conn)
    prompt = build_prediction_prompt(track, conn)
    conn.close()

    print("=" * 70)
    print("prompts.py — Smoke Test")
    print("=" * 70)
    print(f"Track   : \"{track['title']}\" by {track['artist']}")
    print(f"Rules   : {len(active_rules)} active rule(s) injected")
    print("=" * 70)
    print()
    print(prompt)
    print()
    print("=" * 70)
    print("[OK] Prompt built successfully.")
