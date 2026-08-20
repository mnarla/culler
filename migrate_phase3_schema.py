"""Database schema migration for Phase 3: Self-Calibration and Rule Evolution.

This script updates skip_predictor.db with new columns on the 'rules' and 'runs'
tables to support rule performance tracking, consolidation runs, and calibration history.

Schema Changes:
    1. 'rules' table:
        - times_correct (INTEGER DEFAULT 0): Count of times this rule's prediction matched ground-truth.
        - created_by_run_id (INTEGER): Nullable FK to runs.id, consolidation run that created this rule.
        - superseded_by (INTEGER): Nullable self-referencing FK to rules.id when merged.
        - retirement_reason (TEXT): Explanation when pruned or superseded.

    2. 'runs' table:
        - accuracy_before (REAL): Overall accuracy on batch before consolidation run.
        - skip_recall_before (REAL): Skip recall on batch before consolidation run.
        - label_ids (TEXT): JSON array string of evaluated label IDs (e.g. "[14, 22, 37]").

runs.summary Contract (JSON Contract):
    Every write to runs.summary must be a JSON string following this structure:
    {
        "rule_changes": [
            {"rule_id": 7, "action": "pruned", "reason": "0% correct over 12 applications"},
            {"rule_id": 12, "action": "created", "reason": "systematic Skip miss on live-genre tracks"},
            {"rule_id": 3, "action": "merged", "into_rule_id": 12, "reason": "near-duplicate condition"},
            {"rule_id": 5, "action": "modified", "reason": "tightened release_year bound"}
        ]
    }

    Valid 'action' values:
        - "created"  : A new rule was synthesized by the agent.
        - "pruned"   : An underperforming or harmful rule was deactivated/retired.
        - "merged"   : A rule was merged into another rule (must include 'into_rule_id').
        - "modified" : An existing rule's condition text was adjusted.

Usage:
    python migrate_phase3_schema.py
    python migrate_phase3_schema.py --db skip_predictor.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

DEFAULT_DB_PATH = "skip_predictor.db"

# Column definitions to add: (table_name, column_name, sql_definition)
RULES_NEW_COLUMNS: List[Tuple[str, str]] = [
    (
        "times_correct",
        "INTEGER DEFAULT 0",
    ),
    (
        "created_by_run_id",
        "INTEGER REFERENCES runs(id)",
    ),
    (
        "superseded_by",
        "INTEGER REFERENCES rules(id)",
    ),
    (
        "retirement_reason",
        "TEXT",
    ),
]

RUNS_NEW_COLUMNS: List[Tuple[str, str]] = [
    (
        "accuracy_before",
        "REAL",
    ),
    (
        "skip_recall_before",
        "REAL",
    ),
    (
        "label_ids",
        "TEXT",
    ),
]


def get_existing_columns(conn: sqlite3.Connection, table_name: str) -> Set[str]:
    """Retrieve set of existing column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row[1] for row in cursor.fetchall()}


def print_table_schema(conn: sqlite3.Connection, table_name: str) -> None:
    """Print the formatted schema and column info for a given table."""
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    row = cursor.fetchone()
    print(f"\n{'=' * 60}")
    print(f"Table Schema: {table_name}")
    print(f"{'=' * 60}")
    if row and row[0]:
        print(row[0].strip())
    print("\nColumns:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    for col in cursor.fetchall():
        cid, name, col_type, notnull, dflt_value, pk = col
        pk_str = " [PRIMARY KEY]" if pk else ""
        notnull_str = " [NOT NULL]" if notnull else ""
        dflt_str = f" [DEFAULT {dflt_value}]" if dflt_value is not None else ""
        print(f"  • {name:<24} {col_type:<10}{notnull_str}{dflt_str}{pk_str}")
    print(f"{'=' * 60}")


def migrate(db_path: Path | str) -> bool:
    """Execute schema migration in a single transaction."""
    path_obj = Path(db_path)
    if not path_obj.exists():
        print(f"[!] Error: Database '{db_path}' not found.", file=sys.stderr)
        return False

    print(f"[*] Connecting to SQLite database: '{db_path}'...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        # Check existing columns
        rules_cols = get_existing_columns(conn, "rules")
        runs_cols = get_existing_columns(conn, "runs")

        changes_made = 0

        with conn:
            # 1. Alter rules table
            for col_name, col_def in RULES_NEW_COLUMNS:
                if col_name not in rules_cols:
                    stmt = f"ALTER TABLE rules ADD COLUMN {col_name} {col_def};"
                    print(f"  [+] rules: adding column '{col_name}' ({col_def})...")
                    conn.execute(stmt)
                    changes_made += 1
                else:
                    print(f"  [-] rules: column '{col_name}' already exists (skipped).")

            # 2. Alter runs table
            for col_name, col_def in RUNS_NEW_COLUMNS:
                if col_name not in runs_cols:
                    stmt = f"ALTER TABLE runs ADD COLUMN {col_name} {col_def};"
                    print(f"  [+] runs: adding column '{col_name}' ({col_def})...")
                    conn.execute(stmt)
                    changes_made += 1
                else:
                    print(f"  [-] runs: column '{col_name}' already exists (skipped).")

        print(f"\n[✓] Migration successful! Total alterations applied: {changes_made}")

        # Print new schema verification
        print_table_schema(conn, "rules")
        print_table_schema(conn, "runs")

        return True

    except Exception as e:
        print(f"\n[!] Migration failed with error: {e}", file=sys.stderr)
        print("[!] Transaction rolled back automatically.", file=sys.stderr)
        return False

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate skip_predictor.db schema for Phase 3 self-calibration fields."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    success = migrate(db_path=args.db)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
