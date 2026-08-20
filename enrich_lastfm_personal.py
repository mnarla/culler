"""Enrich tracks in SQLite database with personal Last.fm user playcounts.

Queries Last.fm's track.getInfo API with a specific username (default: 'mnarla')
to fetch personal scrobble history (userplaycount), adds the user_playcount column
to the 'features' table if missing, and updates all tracks with persistent disk caching.

Usage:
    python enrich_lastfm_personal.py
    python enrich_lastfm_personal.py --username mnarla --db skip_predictor.db
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = "skip_predictor.db"
DEFAULT_USERNAME = "mnarla"
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
REQUEST_DELAY_SEC = 0.25  # ~4 requests/second, under Last.fm's 5 req/sec limit
CACHE_DIR = Path(".cache")
PERSONAL_TRACKS_CACHE = CACHE_DIR / "lastfm_personal_tracks.json"


def load_json_cache(cache_file: Path) -> Dict[str, Any]:
    """Safely load JSON cache file from disk."""
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json_cache(cache_file: Path, data: Dict[str, Any]) -> None:
    """Safely save JSON cache file to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_env_file(env_path: Path = Path(".env")) -> None:
    """Load key-value pairs from .env file into os.environ if present."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Get SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def clean_track_title(title: str) -> str:
    """Clean track title by removing remaster, live, bonus, and feat annotations for better Last.fm matching."""
    cleaned = title
    cleaned = re.sub(r"\s*[-–—]\s*\d{4}\s*[-–—]?\s*Remaster.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[-–—]\s*Remaster.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:Remastered|Remaster|Live|Mono|Stereo|Deluxe).*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[(?:Remastered|Remaster|Live|Mono|Stereo|Deluxe).*?\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:feat\.|ft\.|featuring).*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[(?:feat\.|ft\.|featuring).*?\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() if cleaned.strip() else title


class LastFmPersonalClient:
    """Client for querying Last.fm user scrobble counts with disk caching."""

    def __init__(self, api_key: str, username: str = DEFAULT_USERNAME):
        self.api_key = api_key
        self.username = username.strip()
        self.track_cache: Dict[str, Dict[str, Any]] = load_json_cache(PERSONAL_TRACKS_CACHE)

    def save_cache(self) -> None:
        """Persist track cache to disk."""
        save_json_cache(PERSONAL_TRACKS_CACHE, self.track_cache)

    def _make_request(self, params: Dict[str, str], retry_on_fail: bool = True) -> Optional[Dict[str, Any]]:
        """Make an HTTP GET request to Last.fm API with rate limiting and retry."""
        time.sleep(REQUEST_DELAY_SEC)
        query_params = {**params, "api_key": self.api_key, "format": "json"}
        url = f"{LASTFM_BASE_URL}?{urllib.parse.urlencode(query_params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "CullerPlaylistAgent/1.0"})

        for attempt in range(2 if retry_on_fail else 1):
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        if "error" in data:
                            return None
                        return data
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                if attempt == 0 and retry_on_fail:
                    time.sleep(2.0)
                else:
                    return None
            except Exception:
                return None
        return None

    def get_user_playcount(self, artist: str, track: str) -> Tuple[int, bool]:
        """Fetch user scrobble count for a track.
        
        Returns:
            Tuple of (user_playcount: int, from_cache: bool)
        """
        user_key = self.username.lower()
        track_key = f"{user_key}::{artist.strip().lower()}::{track.strip().lower()}"

        if track_key in self.track_cache:
            cached_val = self.track_cache[track_key].get("userplaycount")
            if cached_val is not None:
                return int(cached_val), True

        cleaned_title = clean_track_title(track)
        data = self._make_request({
            "method": "track.getInfo",
            "artist": artist,
            "track": cleaned_title,
            "username": self.username,
        })

        if not data and cleaned_title != track:
            # Fallback to uncleaned title
            data = self._make_request({
                "method": "track.getInfo",
                "artist": artist,
                "track": track,
                "username": self.username,
            })

        userplaycount = 0
        if data and "track" in data:
            t = data["track"]
            raw_count = t.get("userplaycount")
            if raw_count is not None:
                try:
                    userplaycount = max(0, int(raw_count))
                except (ValueError, TypeError):
                    userplaycount = 0

        self.track_cache[track_key] = {"userplaycount": userplaycount}
        return userplaycount, False


def ensure_user_playcount_column(conn: sqlite3.Connection) -> None:
    """Add user_playcount column to features table if it does not already exist."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(features);")
    columns = [row[1] for row in cursor.fetchall()]
    if "user_playcount" not in columns:
        print("[*] Adding 'user_playcount' column to 'features' table...")
        cursor.execute("ALTER TABLE features ADD COLUMN user_playcount INTEGER DEFAULT 0;")
        conn.commit()


def fetch_all_tracks(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[Tuple[int, str, str]]:
    """Fetch all tracks from the database."""
    cursor = conn.cursor()
    query = "SELECT id, title, artist FROM tracks ORDER BY id ASC"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"
    cursor.execute(query)
    return cursor.fetchall()


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Enrich tracks in SQLite database with personal Last.fm user playcounts."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=DEFAULT_USERNAME,
        help=f"Last.fm username to fetch scrobbles for (default: {DEFAULT_USERNAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks to process in this run (useful for testing).",
    )

    args = parser.parse_args()

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print("[!] Error: LASTFM_API_KEY environment variable is not set.", file=sys.stderr)
        print("    Please set it in your environment or in a .env file:", file=sys.stderr)
        print("    export LASTFM_API_KEY=\"your_lastfm_api_key_here\"", file=sys.stderr)
        sys.exit(1)

    if not Path(args.db).exists():
        print(f"[!] Error: Database file '{args.db}' does not exist. Run init_db.py first.", file=sys.stderr)
        sys.exit(1)

    conn = get_connection(args.db)
    ensure_user_playcount_column(conn)

    tracks = fetch_all_tracks(conn, limit=args.limit)
    if not tracks:
        print(f"[*] No tracks found in '{args.db}'.")
        conn.close()
        return

    print(f"[*] Processing {len(tracks)} tracks for Last.fm user '{args.username}'...")
    client = LastFmPersonalClient(api_key=api_key, username=args.username)

    played_count = 0
    zero_count = 0
    error_count = 0

    cursor = conn.cursor()

    for idx, (track_id, title, artist) in enumerate(tracks, 1):
        print(f"  [{idx:>3}/{len(tracks)}] {artist} - \"{title}\" ...", end="", flush=True)

        try:
            userplaycount, from_cache = client.get_user_playcount(artist, title)

            if userplaycount > 0:
                played_count += 1
                cache_tag = " (cached)" if from_cache else ""
                status_str = f" [user_playcount: {userplaycount}{cache_tag}]"
            else:
                zero_count += 1
                cache_tag = " (cached)" if from_cache else ""
                status_str = f" [user_playcount: 0{cache_tag}]"

            # Upsert into features table
            cursor.execute(
                """
                INSERT INTO features (track_id, user_playcount)
                VALUES (?, ?)
                ON CONFLICT(track_id) DO UPDATE SET user_playcount = excluded.user_playcount;
                """,
                (track_id, userplaycount),
            )
            conn.commit()
            print(status_str)

        except Exception as e:
            error_count += 1
            print(f" [ERROR: {e}]")

    client.save_cache()
    conn.close()

    print("\n" + "=" * 60)
    print(f"Personal Last.fm Scrobble Enrichment Summary ({args.username})")
    print("=" * 60)
    print(f"  • Total Tracks Processed   : {len(tracks)}")
    print(f"  • Tracks with Plays (> 0)  : {played_count}")
    print(f"  • Tracks Defaulted to 0    : {zero_count}")
    print(f"  • Errors Encountered       : {error_count}")
    print(f"  • Total Cache Entries      : {len(client.track_cache)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
