"""Enrich tracks in SQLite database with Last.fm metadata and computed features.

Pulls genre tags, playcounts, and listener counts from Last.fm, calculates
days_since_added, and populates the 'features' table.
"""

from __future__ import annotations

import argparse
import datetime
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
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
REQUEST_DELAY_SEC = 0.25  # ~4 requests/second, well under Last.fm's 5 req/sec limit
CACHE_DIR = Path(".cache")
LASTFM_ARTISTS_CACHE = CACHE_DIR / "lastfm_artists.json"
LASTFM_TRACKS_CACHE = CACHE_DIR / "lastfm_tracks.json"


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
    # Remove (- YYYY Remaster), (Remastered YYYY), [Remastered], etc.
    cleaned = re.sub(r"\s*[-–—]\s*\d{4}\s*[-–—]?\s*Remaster.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[-–—]\s*Remaster.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:Remastered|Remaster|Live|Mono|Stereo|Deluxe).*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[(?:Remastered|Remaster|Live|Mono|Stereo|Deluxe).*?\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:feat\.|ft\.|featuring).*?\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[(?:feat\.|ft\.|featuring).*?\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() if cleaned.strip() else title


def calculate_days_since_added(added_at_str: Optional[str]) -> Optional[int]:
    """Calculate the number of days between tracks.added_at and today."""
    if not added_at_str:
        return None
    try:
        # Handle ISO formats: 2025-12-05T06:09:19Z or 2025-12-05
        date_part = added_at_str.split("T")[0]
        added_date = datetime.date.fromisoformat(date_part)
        today = datetime.date.today()
        delta = (today - added_date).days
        return max(0, delta)
    except Exception:
        return None


def parse_release_year(raw_val: Optional[str]) -> Optional[int]:
    """Extract 4-digit release year from date strings if present."""
    if not raw_val:
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", str(raw_val))
    if match:
        return int(match.group(1))
    return None


class LastFmClient:
    """Client for querying the Last.fm read-only API with persistent disk caching."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.artist_tag_cache: Dict[str, Optional[str]] = load_json_cache(LASTFM_ARTISTS_CACHE)
        self.track_cache: Dict[str, Dict[str, Any]] = load_json_cache(LASTFM_TRACKS_CACHE)

    def save_caches(self) -> None:
        """Persist in-memory caches to disk."""
        save_json_cache(LASTFM_ARTISTS_CACHE, self.artist_tag_cache)
        save_json_cache(LASTFM_TRACKS_CACHE, self.track_cache)

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

    def get_artist_top_tags(self, artist: str) -> Optional[str]:
        """Fetch top 1-3 tags for an artist as comma-separated string, using disk/memory cache."""
        artist_key = artist.strip().lower()
        if artist_key in self.artist_tag_cache and self.artist_tag_cache[artist_key] is not None:
            return self.artist_tag_cache[artist_key]

        data = self._make_request({"method": "artist.getTopTags", "artist": artist})
        tags: List[str] = []
        if data and "toptags" in data and "tag" in data["toptags"]:
            raw_tags = data["toptags"]["tag"]
            if isinstance(raw_tags, dict):
                raw_tags = [raw_tags]
            for t in raw_tags[:3]:
                name = t.get("name", "").strip().lower()
                if name:
                    tags.append(name)

        result = ", ".join(tags) if tags else None
        self.artist_tag_cache[artist_key] = result
        return result

    def get_track_info(self, artist: str, track: str) -> Tuple[Optional[int], Optional[int]]:
        """Fetch playcount and listeners count for a track."""
        track_key = f"{artist.strip().lower()}::{track.strip().lower()}"
        if track_key in self.track_cache:
            cache_info = self.track_cache[track_key]
            if cache_info.get("listeners") is not None or cache_info.get("playcount") is not None:
                return cache_info.get("playcount"), cache_info.get("listeners")

        cleaned_title = clean_track_title(track)
        data = self._make_request({"method": "track.getInfo", "artist": artist, "track": cleaned_title})

        if not data and cleaned_title != track:
            # Fallback to original uncleaned title
            data = self._make_request({"method": "track.getInfo", "artist": artist, "track": track})

        playcount = None
        listeners = None

        if data and "track" in data:
            t = data["track"]
            try:
                playcount = int(t.get("playcount", 0)) if t.get("playcount") is not None else None
            except (ValueError, TypeError):
                playcount = None

            try:
                listeners = int(t.get("listeners", 0)) if t.get("listeners") is not None else None
            except (ValueError, TypeError):
                listeners = None

        # Store in track cache
        if track_key not in self.track_cache:
            self.track_cache[track_key] = {}
        self.track_cache[track_key]["playcount"] = playcount
        self.track_cache[track_key]["listeners"] = listeners

        return playcount, listeners

    def get_track_top_tags(self, artist: str, track: str) -> Optional[str]:
        """Fetch top track-level tags (comma-separated) for a track."""
        track_key = f"{artist.strip().lower()}::{track.strip().lower()}"
        if track_key in self.track_cache and "tags" in self.track_cache[track_key]:
            return self.track_cache[track_key]["tags"]

        cleaned_title = clean_track_title(track)
        data = self._make_request({"method": "track.getTopTags", "artist": artist, "track": cleaned_title})

        if not data and cleaned_title != track:
            data = self._make_request({"method": "track.getTopTags", "artist": artist, "track": track})

        tags: List[str] = []
        if data and "toptags" in data and "tag" in data["toptags"]:
            raw_tags = data["toptags"]["tag"]
            if isinstance(raw_tags, dict):
                raw_tags = [raw_tags]
            for t in raw_tags[:5]:
                name = t.get("name", "").strip().lower()
                if name:
                    tags.append(name)

        result = ", ".join(tags) if tags else None
        if track_key not in self.track_cache:
            self.track_cache[track_key] = {}
        self.track_cache[track_key]["tags"] = result
        return result


def fetch_unenriched_tracks(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[Tuple[int, str, str, str, Optional[str], Optional[str]]]:
    """Find tracks that either have no entry in features or have NULL Last.fm data.

    Returns:
        List of (track_id, spotify_uri, title, artist, album, added_at)
    """
    cursor = conn.cursor()
    query = """
        SELECT t.id, t.spotify_uri, t.title, t.artist, t.album, t.added_at
        FROM tracks t
        LEFT JOIN features f ON t.id = f.track_id
        WHERE f.track_id IS NULL 
           OR (f.lastfm_listeners IS NULL AND f.lastfm_track_tags IS NULL AND f.genre IS NULL)
        ORDER BY t.id ASC
    """
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"

    cursor.execute(query)
    return cursor.fetchall()


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Enrich tracks in SQLite database with Last.fm tags, playcounts, and listeners."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks to enrich in this run (useful for testing).",
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
    tracks_to_process = fetch_unenriched_tracks(conn, limit=args.limit)

    if not tracks_to_process:
        print(f"[*] No unenriched tracks found in '{args.db}'. All tracks already have features.")
        conn.close()
        return

    print(f"[*] Found {len(tracks_to_process)} unenriched track(s). Starting Last.fm enrichment...")
    client = LastFmClient(api_key=api_key)

    enriched_count = 0
    no_match_count = 0
    error_count = 0

    cursor = conn.cursor()

    for idx, (track_id, uri, title, artist, album, added_at) in enumerate(tracks_to_process, 1):
        print(f"  [{idx}/{len(tracks_to_process)}] Processing: {artist} - {title} ...", end="", flush=True)

        try:
            # 1. Artist-level genre tags (cached)
            genre = client.get_artist_top_tags(artist)

            # 2. Track info (playcount, listeners)
            playcount, listeners = client.get_track_info(artist, title)

            # 3. Track-level top tags
            track_tags = client.get_track_top_tags(artist, title)

            # 4. Computed features
            days_since_added = calculate_days_since_added(added_at)
            
            # TODO: Infer album_type (album/single/compilation) from album metadata heuristics
            album_type = None

            # Attempt release year extraction if album/track string has year annotation
            release_year = parse_release_year(album)

            # Artist cooccurrence score default (to be computed across playlists)
            artist_cooccurrence_score = None

            has_lastfm_data = bool(genre or playcount or listeners or track_tags)
            if has_lastfm_data:
                enriched_count += 1
                status_str = f" [OK] (Genre: {genre or 'N/A'}, Listeners: {listeners or 'N/A'})"
            else:
                no_match_count += 1
                status_str = " [No Last.fm match - inserted default metadata]"

            cursor.execute(
                """
                INSERT OR REPLACE INTO features (
                    track_id, genre, release_year, album_type, days_since_added,
                    artist_cooccurrence_score, lastfm_playcount, lastfm_listeners,
                    lastfm_track_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    track_id,
                    genre,
                    release_year,
                    album_type,
                    days_since_added,
                    artist_cooccurrence_score,
                    playcount,
                    listeners,
                    track_tags,
                ),
            )
            conn.commit()
            print(status_str)

        except Exception as e:
            error_count += 1
            print(f" [ERROR: {e}]")

    client.save_caches()
    cursor.execute("SELECT COUNT(*) FROM features;")
    total_features = cursor.fetchone()[0]
    conn.close()

    print("\n--- Last.fm Enrichment Summary ---")
    print(f"  • Processed in this run    : {len(tracks_to_process)}")
    print(f"  • Enriched with Last.fm    : {enriched_count}")
    print(f"  • No Last.fm match (NULLs) : {no_match_count}")
    print(f"  • Errored out              : {error_count}")
    print(f"  • Total features in DB     : {total_features}")
    print(f"  • Cached artists in memory : {len(client.artist_tag_cache)}")
    print("----------------------------------\n")


if __name__ == "__main__":
    main()
