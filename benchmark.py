"""Model benchmarking harness for Skip-Prediction Playlist Agent (Culler).

Pulls a reproducible sample of 50 tracks from SQLite, runs inference using
the active model configured in config.py via llm_provider.predict_chat(),
evaluates inference throughput, and exports structured CSV results.

Usage:
    python benchmark.py
    python benchmark.py --sample-size 50 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from llm_provider import predict_chat


def get_db_connection(db_path: Path | str) -> sqlite3.Connection:
    """Connect to SQLite database with Row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_sample_tracks(
    db_path: Path | str,
    sample_size: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Pull a deterministic sample of tracks joined with their features.
    
    Sorting by track ID before deterministic sampling ensures exact reproducibility
    across separate script invocations.
    """
    conn = get_db_connection(db_path)
    query = """
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
            f.artist_cooccurrence_score
        FROM tracks t
        JOIN features f ON t.id = f.track_id
        ORDER BY t.id ASC;
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(f"No tracks with features found in database: '{db_path}'")

    if len(rows) < sample_size:
        print(f"[!] Warning: Requested sample size {sample_size} > available tracks ({len(rows)}). Using all tracks.")
        sample_size = len(rows)

    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def parse_verdict(response_text: str) -> str:
    """Parse 'Keep' or 'Skip' from model response text.
    
    Returns 'Keep', 'Skip', or 'UNCLEAR' if response cannot be determined.
    """
    if not response_text:
        return "UNCLEAR"

    text = response_text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", text)
    tokens = set(cleaned.split())

    has_keep = "keep" in tokens
    has_skip = "skip" in tokens

    if has_keep and not has_skip:
        return "Keep"
    if has_skip and not has_keep:
        return "Skip"

    # Fallback to substring check if punctuation/formatting obscured word boundaries
    if "keep" in text and "skip" not in text:
        return "Keep"
    if "skip" in text and "keep" not in text:
        return "Skip"

    return "UNCLEAR"


def build_prediction_prompt(track: Dict[str, Any], is_qwen3: bool = False) -> str:
    """Construct structured 5-feature classification prompt."""
    cooccur = track.get("artist_cooccurrence_score")
    cooccur_str = f"{cooccur:.4f}" if isinstance(cooccur, (int, float)) else "Unknown"

    prompt_body = (
        "You are an intelligent music playlist curator.\n"
        "Evaluate the following track based on its playlist features and decide whether to Keep or Skip it.\n\n"
        f"Track: \"{track.get('title')}\" by {track.get('artist')}\n"
        f"Features:\n"
        f"  - Genre: {track.get('genre') or 'Unknown'}\n"
        f"  - Release Year: {track.get('release_year') or 'Unknown'}\n"
        f"  - Album Type: {track.get('album_type') or 'Unknown'}\n"
        f"  - Days Since Added: {track.get('days_since_added') if track.get('days_since_added') is not None else 'Unknown'}\n"
        f"  - Artist Co-occurrence Score: {cooccur_str}\n\n"
        "Should this track be kept or skipped in the playlist? Answer Keep or Skip in one word."
    )

    if is_qwen3:
        return f"/no_think\n{prompt_body}"
    return prompt_body


def run_benchmark(
    db_path: Path | str = config.DB_PATH,
    sample_size: int = 50,
    seed: int = 42,
    output_dir: Path = Path("benchmark_results"),
) -> Path:
    """Execute benchmarking harness and write results CSV."""
    active_model_path: Path = config.ACTIVE_MODEL_PATH
    is_qwen3: bool = "qwen3" in str(active_model_path).lower()
    model_name: str = active_model_path.stem

    print("=" * 70)
    print("Skip-Prediction Model Benchmark Harness")
    print("=" * 70)
    print(f"  Active Model : {active_model_path}")
    print(f"  Qwen3 Mode   : {'Enabled (/no_think prefix)' if is_qwen3 else 'Disabled'}")
    print(f"  Database     : {db_path}")
    print(f"  Sample Size  : {sample_size} tracks (seed={seed})")
    print("=" * 70)

    # 1. Fetch deterministic sample
    tracks = load_sample_tracks(db_path=db_path, sample_size=sample_size, seed=seed)
    print(f"[+] Loaded {len(tracks)} tracks for evaluation.\n")

    results: List[Dict[str, Any]] = []
    total_start_time = time.perf_counter()

    # 2. Iterate through sampled tracks
    for idx, track in enumerate(tracks, 1):
        prompt = build_prediction_prompt(track, is_qwen3=is_qwen3)
        messages = [{"role": "user", "content": prompt}]

        track_label = f"\"{track['title']}\" - {track['artist']}"
        print(f"[{idx:>2}/{len(tracks)}] Evaluating: {track_label[:45]:<45}", end=" ", flush=True)

        try:
            pred = predict_chat(
                messages=messages,
                max_tokens=config.DEFAULT_MAX_TOKENS,
                temperature=config.TEMPERATURE,
            )
            raw_text = pred.get("text", "")
            verdict = parse_verdict(raw_text)
            elapsed = pred.get("elapsed_sec", 0.0)
            comp_tokens = pred.get("completion_tokens", 0)
            prompt_tokens = pred.get("prompt_tokens", 0)
            total_tokens = pred.get("total_tokens", 0)

            print(f"-> {verdict:<7} ({elapsed:5.2f}s, {comp_tokens:>3} tokens)")

            results.append({
                "track_id": track["track_id"],
                "title": track["title"],
                "artist": track["artist"],
                "genre": track["genre"],
                "release_year": track["release_year"],
                "album_type": track["album_type"],
                "days_since_added": track["days_since_added"],
                "artist_cooccurrence_score": track["artist_cooccurrence_score"],
                "verdict": verdict,
                "raw_response": raw_text,
                "elapsed_sec": elapsed,
                "completion_tokens": comp_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            })
        except Exception as err:
            print(f"-> ERROR: {err}")
            results.append({
                "track_id": track["track_id"],
                "title": track["title"],
                "artist": track["artist"],
                "genre": track["genre"],
                "release_year": track["release_year"],
                "album_type": track["album_type"],
                "days_since_added": track["days_since_added"],
                "artist_cooccurrence_score": track["artist_cooccurrence_score"],
                "verdict": "ERROR",
                "raw_response": str(err),
                "elapsed_sec": 0.0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            })

    total_wall_time = time.perf_counter() - total_start_time

    # 3. Export CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_name}.csv"

    fieldnames = [
        "track_id",
        "title",
        "artist",
        "genre",
        "release_year",
        "album_type",
        "days_since_added",
        "artist_cooccurrence_score",
        "verdict",
        "raw_response",
        "elapsed_sec",
        "completion_tokens",
        "prompt_tokens",
        "total_tokens",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Results saved to: {csv_path}")

    # 4. Summary metrics
    n_successful = len([r for r in results if r["verdict"] != "ERROR"])
    total_elapsed_sec = sum(r["elapsed_sec"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)

    avg_sec_per_decision = (total_elapsed_sec / n_successful) if n_successful > 0 else 0.0
    tokens_per_sec = (total_completion_tokens / total_elapsed_sec) if total_elapsed_sec > 0 else 0.0

    keep_count = sum(1 for r in results if r["verdict"] == "Keep")
    skip_count = sum(1 for r in results if r["verdict"] == "Skip")
    unclear_count = sum(1 for r in results if r["verdict"] == "UNCLEAR")
    error_count = sum(1 for r in results if r["verdict"] == "ERROR")

    print("\n" + "=" * 70)
    print(f"Benchmark Summary: {model_name}")
    print("=" * 70)
    print(f"  Total Tracks Evaluated : {len(results)}")
    print(f"  Total Wall Clock Time  : {total_wall_time:.2f}s")
    print(f"  Cumulative Model Time  : {total_elapsed_sec:.2f}s")
    print(f"  Avg Time / Decision    : {avg_sec_per_decision:.2f}s")
    print(f"  Overall Generation     : {tokens_per_sec:.1f} tok/s ({total_completion_tokens} total tokens)")
    print()
    print("  Verdict Breakdown:")
    print(f"    • Keep    : {keep_count:>3} ({keep_count / len(results) * 100:.1f}%)")
    print(f"    • Skip    : {skip_count:>3} ({skip_count / len(results) * 100:.1f}%)")
    print(f"    • UNCLEAR : {unclear_count:>3} ({unclear_count / len(results) * 100:.1f}%)")
    if error_count > 0:
        print(f"    • ERROR   : {error_count:>3} ({error_count / len(results) * 100:.1f}%)")
    print("=" * 70 + "\n")

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local GGUF models on skip-prediction classification."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(config.DB_PATH),
        help=f"Path to SQLite database (default: {config.DB_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of tracks to sample and benchmark (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible track sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark CSV results (default: benchmark_results)",
    )
    args = parser.parse_args()

    run_benchmark(
        db_path=Path(args.db),
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
