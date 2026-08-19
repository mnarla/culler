"""Database initialization script for Skip-Prediction Playlist Agent (Culler).

Creates and manages the local SQLite database schema for tracks, features,
labels, rules, and run history.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB_PATH = "skip_predictor.db"

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        spotify_uri TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        artist TEXT NOT NULL,
        album TEXT,
        duration_ms INTEGER,
        added_at TEXT,
        playlist_name TEXT NOT NULL,
        playlist_id TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS features (
        track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
        genre TEXT,
        release_year INTEGER,
        album_type TEXT,
        days_since_added INTEGER,
        artist_cooccurrence_score REAL,
        lastfm_playcount INTEGER,
        lastfm_listeners INTEGER,
        lastfm_track_tags TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
        label TEXT CHECK(label IN ('skip', 'keep')) NOT NULL,
        confidence REAL,
        reasoning TEXT,
        time_of_day TEXT,
        day_of_week TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_text TEXT NOT NULL,
        active INTEGER DEFAULT 1,
        times_applied INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_type TEXT NOT NULL,
        rules_before INTEGER,
        rules_after INTEGER,
        summary TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_tracks_playlist_name ON tracks(playlist_name);",
    "CREATE INDEX IF NOT EXISTS idx_labels_track_id ON labels(track_id);",
    "CREATE INDEX IF NOT EXISTS idx_features_track_id ON features(track_id);",
]

DROP_TABLES_STATEMENTS = [
    "DROP TABLE IF EXISTS runs;",
    "DROP TABLE IF EXISTS rules;",
    "DROP TABLE IF EXISTS labels;",
    "DROP TABLE IF EXISTS features;",
    "DROP TABLE IF EXISTS tracks;",
]

TABLE_NAMES = ["tracks", "features", "labels", "rules", "runs"]


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Establish a connection to the SQLite database with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path, reset: bool = False) -> None:
    """Initialize database tables and indexes."""
    conn = get_connection(db_path)
    try:
        with conn:
            if reset:
                print(f"[!] Resetting database: dropping existing tables in '{db_path}'...")
                for drop_stmt in DROP_TABLES_STATEMENTS:
                    conn.execute(drop_stmt)

            for schema_stmt in SCHEMA_STATEMENTS:
                conn.execute(schema_stmt)

            for index_stmt in INDEX_STATEMENTS:
                conn.execute(index_stmt)
    finally:
        conn.close()


def print_table_summary(db_path: str | Path) -> None:
    """Print the row counts for each table in the database."""
    conn = get_connection(db_path)
    try:
        print(f"\n--- Database Summary: {db_path} ---")
        cursor = conn.cursor()
        for table in TABLE_NAMES:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"  • {table:<12} : {count:>6} rows")
        print("---------------------------------------\n")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the SQLite database for the Skip-Prediction Playlist Agent."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables and indexes.",
    )

    args = parser.parse_args()

    action = "Recreating" if args.reset else "Initializing"
    print(f"[*] {action} schema in '{args.db}'...")
    init_db(args.db, reset=args.reset)
    print("[+] Schema initialization complete.")
    print_table_summary(args.db)


if __name__ == "__main__":
    main()
