"""Ingest Exportify CSV exports into the SQLite database.

Parses Exportify playlist CSV files and stores tracks in the 'tracks' table,
while caching multi-artist relations for downstream artist co-occurrence calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = "skip_predictor.db"
DEFAULT_EXPORTS_DIR = "exports"
CACHE_DIR = Path(".cache")
ARTIST_CACHE_FILE = CACHE_DIR / "track_artists.json"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Get SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def normalize_column_name(col: str) -> str:
    """Normalize CSV header strings for flexible column matching."""
    return re.sub(r"[^a-z0-9]", "", col.lower())


def extract_column_mapping(fieldnames: List[str]) -> Dict[str, str]:
    """Map canonical field names to actual CSV column headers."""
    canonical_aliases = {
        "uri": ["trackuri", "spotifyuri", "uri", "trackid"],
        "title": ["trackname", "title", "name", "songname"],
        "artist": ["artistnames", "artistname", "artists", "artist"],
        "album": ["albumname", "album", "albumtitle"],
        "duration": ["trackdurationms", "durationms", "duration"],
        "added_at": ["addedat", "addeddate", "dateadded", "added"],
        "playlist_name": ["playlistname", "playlist"],
        "playlist_id": ["playlistid"],
    }

    mapping: Dict[str, str] = {}
    normalized_headers = {normalize_column_name(h): h for h in fieldnames if h}

    for key, aliases in canonical_aliases.items():
        for alias in aliases:
            if alias in normalized_headers:
                mapping[key] = normalized_headers[alias]
                break

    return mapping


def parse_artists(raw_artist_str: str) -> Tuple[str, List[str]]:
    """Parse comma-separated artist string into primary artist and full list."""
    if not raw_artist_str or not raw_artist_str.strip():
        return "Unknown Artist", ["Unknown Artist"]

    # Split by comma while stripping whitespace
    artists = [a.strip() for a in raw_artist_str.split(",") if a.strip()]
    if not artists:
        return "Unknown Artist", ["Unknown Artist"]

    primary_artist = artists[0]
    return primary_artist, artists


def normalize_spotify_uri(raw_uri: str) -> Optional[str]:
    """Validate and normalize Spotify track URI or URL."""
    if not raw_uri:
        return None
    raw_uri = raw_uri.strip()

    # If it's already a standard spotify:track:URI
    if raw_uri.startswith("spotify:track:"):
        track_id = raw_uri.split(":")[-1]
        if len(track_id) >= 15:
            return raw_uri
        return None

    # If it's an HTTP URL (e.g. https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT)
    url_match = re.search(r"spotify\.com/track/([a-zA-Z0-9]+)", raw_uri)
    if url_match:
        return f"spotify:track:{url_match.group(1)}"

    # If it's just the 22-character alphanumeric Spotify ID
    if re.fullmatch(r"[a-zA-Z0-9]{22}", raw_uri):
        return f"spotify:track:{raw_uri}"

    return None


def load_artist_cache() -> Dict[str, List[str]]:
    """Load existing track->artists mapping cache from disk."""
    if ARTIST_CACHE_FILE.exists():
        try:
            with open(ARTIST_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_artist_cache(cache: Dict[str, List[str]]) -> None:
    """Save track->artists mapping cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def infer_playlist_name(csv_path: Path, csv_row_playlist: Optional[str] = None) -> str:
    """Infer playlist name from row metadata or file name."""
    if csv_row_playlist and csv_row_playlist.strip():
        return csv_row_playlist.strip()
    # Use stem and replace underscores/hyphens with spaces
    stem = csv_path.stem
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned if cleaned else "Exported Playlist"


def ingest_csv_file(
    csv_path: Path,
    conn: sqlite3.Connection,
    playlist_name_override: Optional[str] = None,
    artist_cache: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, int]:
    """Ingest a single Exportify CSV file into the database.

    Returns:
        (inserted_count, skipped_count)
    """
    if artist_cache is None:
        artist_cache = {}

    inserted_count = 0
    skipped_count = 0

    if not csv_path.exists():
        print(f"[!] File not found: {csv_path}")
        return 0, 0

    with open(csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"[!] Warning: No header found in {csv_path}")
            return 0, 0

        col_map = extract_column_mapping(reader.fieldnames)
        if "uri" not in col_map:
            print(f"[!] Error: Could not identify Track URI column in {csv_path}")
            return 0, 0

        rows_to_insert = []
        for row in reader:
            raw_uri = row.get(col_map.get("uri", ""), "")
            spotify_uri = normalize_spotify_uri(raw_uri)

            if not spotify_uri:
                skipped_count += 1
                continue

            raw_title = row.get(col_map.get("title", ""), "").strip()
            title = raw_title if raw_title else "Unknown Title"

            raw_artists = row.get(col_map.get("artist", ""), "")
            primary_artist, all_artists = parse_artists(raw_artists)

            raw_album = row.get(col_map.get("album", ""), "").strip()
            album = raw_album if raw_album else None

            raw_duration = row.get(col_map.get("duration", ""), "")
            try:
                duration_ms = int(float(raw_duration)) if raw_duration else None
            except ValueError:
                duration_ms = None

            raw_added_at = row.get(col_map.get("added_at", ""), "").strip()
            added_at = raw_added_at if raw_added_at else None

            row_playlist = row.get(col_map.get("playlist_name", ""), "")
            playlist_name = (
                playlist_name_override
                if playlist_name_override
                else infer_playlist_name(csv_path, row_playlist)
            )

            playlist_id = row.get(col_map.get("playlist_id", ""), "").strip() or None

            rows_to_insert.append(
                (
                    spotify_uri,
                    title,
                    primary_artist,
                    album,
                    duration_ms,
                    added_at,
                    playlist_name,
                    playlist_id,
                )
            )
            artist_cache[spotify_uri] = all_artists

        if rows_to_insert:
            cursor = conn.cursor()
            initial_rows = conn.total_changes
            cursor.executemany(
                """
                INSERT OR IGNORE INTO tracks (
                    spotify_uri, title, artist, album, duration_ms, added_at, playlist_name, playlist_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows_to_insert,
            )
            conn.commit()
            inserted_count = conn.total_changes - initial_rows

    return inserted_count, skipped_count


def find_csv_files(target_path: Path) -> List[Path]:
    """Resolve target path into a list of CSV files."""
    if target_path.is_file():
        return [target_path]
    elif target_path.is_dir():
        return sorted(list(target_path.glob("*.csv")))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Exportify CSV playlist exports into SQLite tracks table."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_EXPORTS_DIR,
        help=f"Path to an Exportify CSV file or directory of CSV files (default: {DEFAULT_EXPORTS_DIR})",
    )
    parser.add_argument(
        "--playlist-name",
        type=str,
        default=None,
        help="Override the playlist name stored for all rows in this import.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )

    args = parser.parse_args()
    target_path = Path(args.path)

    csv_files = find_csv_files(target_path)
    if not csv_files:
        print(f"[!] No CSV files found at '{target_path}'.")
        print(f"    Create a CSV or place Exportify exports in './{DEFAULT_EXPORTS_DIR}/'")
        return

    conn = get_connection(args.db)
    artist_cache = load_artist_cache()

    total_inserted = 0
    total_skipped = 0

    print(f"[*] Ingesting {len(csv_files)} CSV file(s) into '{args.db}'...")

    for csv_file in csv_files:
        print(f"  -> Processing: {csv_file.name}")
        inserted, skipped = ingest_csv_file(
            csv_file,
            conn,
            playlist_name_override=args.playlist_name,
            artist_cache=artist_cache,
        )
        total_inserted += inserted
        total_skipped += skipped
        print(f"     Inserted: {inserted} | Skipped/Malformed: {skipped}")

    save_artist_cache(artist_cache)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tracks;")
    total_in_table = cursor.fetchone()[0]
    conn.close()

    print("\n--- Ingestion Summary ---")
    print(f"  • Tracks inserted this run : {total_inserted}")
    print(f"  • Rows skipped (malformed) : {total_skipped}")
    print(f"  • Total tracks in table    : {total_in_table}")
    print(f"  • Cached artist relations  : {len(artist_cache)} tracks in {ARTIST_CACHE_FILE}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
