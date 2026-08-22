"""Prompt builder for the Skip-Prediction calibration loop.

Constructs the full inference prompt for a single track, incorporating:
  - Formatted track features (identical layout to benchmark.py)
  - Active rules from the database, ranked by correctness rate
  - Structured 3-step reasoning constraints (no /no_think prefix)

Note on /no_think:
    The /no_think prefix was intentionally removed after direct comparison
    testing showed it causes ungrounded, hallucinated reasoning: constant
    confidence scores regardless of track, and factually incorrect feature
    claims (e.g. asserting high playcount when the value was 0). Allowing
    the model's full thinking trace produces grounded reasoning that
    correctly references the actual feature values in the prompt.

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
                     verdict (e.g. Keep: "high co-occurrence (0.065) and positive
                     plays confirm anchor artist despite age"; Skip: "zero plays,
                     low co-occurrence (0.008), and added 1900 days ago indicate
                     forgotten filler"). This field is mined by
                     consolidate_rules.py for error pattern discovery.

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
        the user message content. No /no_think prefix — thinking mode is
        intentionally left active to produce grounded, feature-accurate reasoning.

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

    # ── Decision Framework & Reasoning Constraints ────────────────────────────
    lines.append("Decision Framework:")
    lines.append("  - Taste Preservation Prior: Assume saved tracks represent genuine taste unless clear decay is demonstrated.")
    lines.append("  - Strong KEEP Signals: High Artist Co-occurrence Score (>= 0.05 / 5%), positive personal play count (> 0), or core anchor artists. High artist co-occurrence strongly protects older tracks from automatic skip verdicts.")
    lines.append("  - Strong SKIP Signals: Clear evidence of decay — 0 play count AND high Days Since Added (> 1000) AND low Artist Co-occurrence Score (< 0.02 / 2%), representing forgotten one-offs.")
    lines.append("  - Signal Hierarchy: Artist loyalty/co-occurrence and positive plays outweigh age alone. A track added years ago should NOT be skipped if it belongs to a recurring core artist.")
    lines.append("")
    lines.append("Reasoning Process (follow these 3 steps in order):")
    lines.append("  1. Identify KEEP signals: Assess artist loyalty, co-occurrence score, and personal play count.")
    lines.append("  2. Identify SKIP signals: Check for isolated one-off artists, zero plays, and age decay.")
    lines.append("  3. Objective Synthesis: Weigh both sides against the framework to determine whether Keep or Skip wins and why.")
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
        '(e.g. Keep: "high artist co-occurrence (0.065) and positive playcount confirm core favorite despite age"; '
        'Skip: "zero plays, low co-occurrence (0.008), and added 1900 days ago indicate forgotten one-off"). '
        "Generic or vague reasoning is not acceptable."
    )

    prompt_body = "\n".join(lines)

    # /no_think is intentionally absent: prepending it was found to produce
    # ungrounded reasoning — constant confidence scores, incorrect feature claims.
    # Allowing the full thinking trace yields feature-accurate verdicts.
    return prompt_body


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
