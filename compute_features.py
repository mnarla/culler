"""Feature engineering pipeline for Skip-Prediction Playlist Agent (Culler).

Computes artist co-occurrence frequencies, days since added, filtered top tags,
release years, and popularity proxy metrics, storing results in the SQLite 'features' table.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_DB_PATH = "skip_predictor.db"
ALT_DB_PATH = "playlist_agent.db"
CACHE_DIR = Path(".cache")
TRACK_ARTISTS_CACHE = CACHE_DIR / "track_artists.json"
METADATA_CACHE = CACHE_DIR / "track_metadata.json"
LASTFM_ARTISTS_CACHE = CACHE_DIR / "lastfm_artists.json"
LASTFM_TRACKS_CACHE = CACHE_DIR / "lastfm_tracks.json"

GENERIC_NOISE_TAGS: Set[str] = {
    "seen live",
    "favorites",
    "favourite",
    "my favorites",
    "owned",
    "cool",
    "love",
    "favorite",
    "favourite tracks",
    "all",
    "awesome",
    "good",
    "beautiful",
    "tracks",
    "songs",
    "spotify",
    "under 2000 listeners",
}


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Establish SQLite connection with foreign key enforcement."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def load_json_cache(cache_file: Path) -> Dict[str, Any]:
    """Safely load JSON cache file from disk."""
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def normalize_artist_name(name: str) -> str:
    """Clean and normalize artist name for frequency and co-occurrence computation."""
    if not name:
        return ""
    # Strip common noise like (feat. ...), ft. ...
    cleaned = re.sub(r"\s*[\(\[](?:feat\.|ft\.|featuring).*?[\)\]]", "", name, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:feat\.|ft\.|featuring)\s+.*", "", cleaned, flags=re.IGNORECASE)
    # Strip punctuation/excess whitespace
    cleaned = re.sub(r"[^\w\s&]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def extract_artists_for_track(
    spotify_uri: str,
    raw_artist_str: str,
    track_artists_cache: Dict[str, List[str]],
) -> List[str]:
    """Retrieve full list of artists for a track from cache or split string."""
    if spotify_uri in track_artists_cache and track_artists_cache[spotify_uri]:
        artists = track_artists_cache[spotify_uri]
    else:
        # Fallback splitting by comma, slash, ampersand
        artists = re.split(r",|/|&|\band\b", raw_artist_str, flags=re.IGNORECASE)

    normalized = [normalize_artist_name(a) for a in artists if normalize_artist_name(a)]
    return normalized if normalized else [normalize_artist_name(raw_artist_str) or "unknown"]


def compute_days_since_added(added_at_str: Optional[str]) -> int:
    """Calculate integer days between added_at and current UTC date."""
    if not added_at_str or not added_at_str.strip():
        return 0
    try:
        clean_str = added_at_str.strip().replace("Z", "+00:00")
        if "T" in clean_str:
            dt = datetime.datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            return max(0, (now - dt).days)
        else:
            d = datetime.date.fromisoformat(clean_str[:10])
            today = datetime.datetime.now(datetime.timezone.utc).date()
            return max(0, (today - d).days)
    except Exception:
        return 0


def extract_release_year(album_str: Optional[str], added_at_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit release year from album or added_at string."""
    if album_str:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", album_str)
        if match:
            return int(match.group(1))
    if added_at_str:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", added_at_str)
        if match:
            return int(match.group(1))
    return None


def infer_album_type(album_str: Optional[str], duration_ms: Optional[int]) -> str:
    """Infer album type (album, single, compilation) from metadata heuristics."""
    if not album_str:
        return "album"
    album_lower = album_str.lower()
    if any(k in album_lower for k in ["greatest hits", "anthology", "collection", "best of", "remastered box set"]):
        return "compilation"
    if " - single" in album_lower or "(single)" in album_lower:
        return "single"
    return "album"


def clean_and_filter_tags(raw_tags_input: Any) -> List[str]:
    """Filter out noise tags, deduplicate, lowercase, and return top 5 tags."""
    tag_candidates: List[str] = []
    if isinstance(raw_tags_input, list):
        tag_candidates = [str(t) for t in raw_tags_input]
    elif isinstance(raw_tags_input, str) and raw_tags_input.strip():
        # Check if JSON encoded
        if raw_tags_input.strip().startswith("["):
            try:
                parsed = json.loads(raw_tags_input)
                if isinstance(parsed, list):
                    tag_candidates = [str(t) for t in parsed]
            except Exception:
                tag_candidates = [t.strip() for t in raw_tags_input.split(",") if t.strip()]
        else:
            tag_candidates = [t.strip() for t in raw_tags_input.split(",") if t.strip()]

    cleaned_tags: List[str] = []
    seen: Set[str] = set()

    for tag in tag_candidates:
        norm = tag.strip().lower()
        if not norm or norm in GENERIC_NOISE_TAGS:
            continue
        if len(norm) < 2 or len(norm) > 40:
            continue
        if norm not in seen:
            seen.add(norm)
            cleaned_tags.append(norm)
        if len(cleaned_tags) >= 5:
            break

    return cleaned_tags


def clean_track_title(title: str) -> str:
    """Clean track title by removing remaster, live, bonus, and feat annotations."""
    cleaned = title
    cleaned = re.sub(r"\s*[-–—]\s*\d{4}\s*[-–—]?\s*Remaster.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[-–—]\s*Remaster.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:Remastered|Remaster|Live|Mono|Stereo|Deluxe).*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[(?:Remastered|Remaster|Live|Mono|Stereo|Deluxe).*?\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:feat\.|ft\.|featuring).*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[(?:feat\.|ft\.|featuring).*?\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() if cleaned.strip() else title


def calculate_popularity_proxy(listeners: Any, playcount: Any) -> Optional[float]:
    """Calculate log-scaled popularity proxy based on Last.fm listeners count."""
    try:
        if listeners is not None and str(listeners).strip() not in ("", "None", "null"):
            val = float(listeners)
            if val > 0:
                return round(math.log10(val + 1), 2)
    except (ValueError, TypeError):
        pass
    try:
        if playcount is not None and str(playcount).strip() not in ("", "None", "null"):
            val = float(playcount)
            if val > 0:
                return round(math.log10(val + 1), 2)
    except (ValueError, TypeError):
        pass
    return None


def compute_all_features(db_path: str | Path) -> Tuple[int, Counter, List[Dict[str, Any]]]:
    """Compute features for all tracks and populate the features table in SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Read existing tracks
    cursor.execute(
        """
        SELECT id, spotify_uri, title, artist, album, duration_ms, added_at, playlist_name
        FROM tracks
        ORDER BY id ASC;
        """
    )
    tracks = cursor.fetchall()
    total_tracks = len(tracks)

    if total_tracks == 0:
        conn.close()
        return 0, Counter(), []

    # Read existing features if any to preserve raw Last.fm fetches
    cursor.execute(
        """
        SELECT track_id, genre, release_year, album_type, days_since_added,
               artist_cooccurrence_score, lastfm_playcount, lastfm_listeners, lastfm_track_tags
        FROM features;
        """
    )
    existing_features_rows = {row[0]: row for row in cursor.fetchall()}

    # Load cache files
    track_artists_cache = load_json_cache(TRACK_ARTISTS_CACHE)
    track_metadata_cache = load_json_cache(METADATA_CACHE)
    lastfm_artists_cache = load_json_cache(LASTFM_ARTISTS_CACHE)
    lastfm_tracks_cache = load_json_cache(LASTFM_TRACKS_CACHE)

    # 1. Compute artist global occurrences across the library
    artist_counts: Counter[str] = Counter()
    track_artist_map: Dict[int, List[str]] = {}

    for t_id, uri, title, artist, album, dur, added_at, pl_name in tracks:
        artists = extract_artists_for_track(uri, artist, track_artists_cache)
        track_artist_map[t_id] = artists
        for a in artists:
            artist_counts[a] += 1

    # 2. Build feature records for batch update
    feature_records_to_insert = []
    preview_samples: List[Dict[str, Any]] = []

    for t_id, uri, title, artist, album, dur, added_at, pl_name in tracks:
        artists = track_artist_map.get(t_id, [normalize_artist_name(artist)])
        primary_artist_norm = artists[0] if artists else normalize_artist_name(artist)

        # a. Artist Co-occurrence / Frequency score (normalized count / total tracks)
        artist_track_count = artist_counts[primary_artist_norm]
        cooccurrence_score = round(artist_track_count / total_tracks, 4) if total_tracks > 0 else 0.0

        # b. Days since added
        days_since = compute_days_since_added(added_at)

        # Existing raw DB values
        existing = existing_features_rows.get(t_id)
        raw_genre = existing[1] if existing else None
        db_release_year = existing[2] if existing else None
        db_album_type = existing[3] if existing else None
        lastfm_playcount = existing[6] if existing else None
        lastfm_listeners = existing[7] if existing else None
        raw_track_tags = existing[8] if existing else None

        meta_info = track_metadata_cache.get(uri, {})

        # Check Last.fm disk caches if DB was empty or incomplete for this track
        possible_track_keys = [
            f"{artist.strip().lower()}::{title.strip().lower()}",
            f"{artist.strip().lower()}::{clean_track_title(title).lower()}",
            f"{primary_artist_norm}::{title.strip().lower()}",
            f"{primary_artist_norm}::{clean_track_title(title).lower()}",
            uri.lower(),
        ]
        cache_info = {}
        for k in possible_track_keys:
            if k in lastfm_tracks_cache and isinstance(lastfm_tracks_cache[k], dict):
                cache_info = lastfm_tracks_cache[k]
                break

        if cache_info:
            if lastfm_playcount is None:
                lastfm_playcount = cache_info.get("playcount")
            if lastfm_listeners is None:
                lastfm_listeners = cache_info.get("listeners")
            if not raw_track_tags or raw_track_tags in ("[]", "null", ""):
                raw_track_tags = cache_info.get("tags")

        # Check artist cache
        for a_key in [primary_artist_norm, artist.strip().lower()]:
            if a_key in lastfm_artists_cache and not raw_genre:
                raw_genre = lastfm_artists_cache[a_key]
                break

        # c. Clean and Filter Top Tags (JSON list string)
        # Track-level tags first, falling back to artist genre tags, then CSV genres
        has_valid_track_tags = raw_track_tags and str(raw_track_tags).strip() not in ("[]", "null", '""', "")
        tag_source = raw_track_tags if has_valid_track_tags else (raw_genre or meta_info.get("genres"))
        top_tags_list = clean_and_filter_tags(tag_source)
        top_tags_json = json.dumps(top_tags_list, ensure_ascii=False)
        genre_str = top_tags_list[0] if top_tags_list else (raw_genre or None)

        # d. Popularity proxy (log10 scaled listeners)
        pop_proxy = calculate_popularity_proxy(lastfm_listeners, lastfm_playcount)

        # e. Release Year & Album Type
        release_year = meta_info.get("release_year") or extract_release_year(album, None) or db_release_year
        album_type = db_album_type or infer_album_type(album, dur)

        feature_records_to_insert.append(
            (
                t_id,
                genre_str,
                release_year,
                album_type,
                days_since,
                cooccurrence_score,
                lastfm_playcount,
                lastfm_listeners,
                top_tags_json,
            )
        )

        if len(preview_samples) < 3:
            preview_samples.append(
                {
                    "title": title,
                    "artist": artist,
                    "days_since_added": days_since,
                    "top_tags": top_tags_list,
                    "cooccurrence_score": cooccurrence_score,
                    "popularity_proxy": pop_proxy,
                    "listeners": lastfm_listeners,
                }
            )

    # 3. Batch insert/replace inside a single transaction
    with conn:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO features (
                track_id,
                genre,
                release_year,
                album_type,
                days_since_added,
                artist_cooccurrence_score,
                lastfm_playcount,
                lastfm_listeners,
                lastfm_track_tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            feature_records_to_insert,
        )

    conn.close()
    return total_tracks, artist_counts, preview_samples


def resolve_db_path(custom_path: Optional[str]) -> Path:
    """Resolve database path with fallback checks."""
    if custom_path:
        return Path(custom_path)
    if Path(DEFAULT_DB_PATH).exists():
        return Path(DEFAULT_DB_PATH)
    if Path(ALT_DB_PATH).exists():
        return Path(ALT_DB_PATH)
    return Path(DEFAULT_DB_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute engineered features for playlist tracks and populate the SQLite features table."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )

    args = parser.parse_args()
    db_path = resolve_db_path(args.db)

    if not db_path.exists():
        print(f"[!] Error: SQLite database '{db_path}' does not exist. Please run init_db.py and ingest_exportify.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Computing features from '{db_path}'...")
    total_processed, artist_counts, preview_samples = compute_all_features(db_path)

    if total_processed == 0:
        print("[!] No tracks found in the database. Ingest CSVs first with ingest_exportify.py.")
        return

    print("\n================ FEATURE COMPUTATION SUMMARY ================")
    print(f"  • Total tracks processed       : {total_processed}")
    print(f"  • Total features populated     : {total_processed}")
    print(f"  • Unique artists identified    : {len(artist_counts)}")

    print("\n--- Top 5 Most Frequent Artists & Co-occurrence ---")
    for artist, count in artist_counts.most_common(5):
        ratio = count / total_processed
        print(f"  • {artist.title():<25} : {count:>3} tracks ({ratio * 100:>5.1f}%)")

    print("\n--- Sample Feature Previews (First 3 Records) ---")
    for i, s in enumerate(preview_samples, 1):
        tags_display = json.dumps(s['top_tags']) if s['top_tags'] else "[]"
        print(f"  [{i}] {s['artist']} - \"{s['title']}\"")
        print(f"      Days Since Added  : {s['days_since_added']} days")
        print(f"      Co-occurrence     : {s['cooccurrence_score']}")
        print(f"      Top Tags          : {tags_display}")
        print(f"      Popularity Proxy  : {s['popularity_proxy']} (raw listeners: {s['listeners']})")
        print()
    print("=============================================================\n")


if __name__ == "__main__":
    main()
