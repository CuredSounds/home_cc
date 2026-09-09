"""
Database Initialization and Testing Tool for Media & ML Library.

Initializes, provisions, seeds, and verifies the SQLite database schema
supporting the full 26-column ML and media metadata taxonomy.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

DEFAULT_DB_PATH = Path("books.db")

# Full 26-column schema definition
SCHEMA_COLUMNS: List[Tuple[str, str]] = [
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("clean_title", "TEXT NOT NULL UNIQUE"),
    ("subtitle", "TEXT"),
    ("raw_title", "TEXT NOT NULL"),
    ("author", "TEXT"),
    ("publisher", "TEXT"),
    ("published", "TEXT"),
    ("edition", "TEXT"),
    ("revision", "INTEGER DEFAULT 1"),
    ("language", "TEXT DEFAULT 'en'"),
    ("domain", "TEXT"),
    ("subject", "TEXT"),
    ("subfield", "TEXT"),
    ("genre", "TEXT"),
    ("topics", "TEXT"),
    ("key_concepts", "TEXT"),
    ("target_audience", "TEXT"),
    ("media_type", "TEXT"),
    ("is_collection", "INTEGER DEFAULT 0"),
    ("summary", "TEXT"),
    ("file_type", "TEXT"),
    ("file_extension", "TEXT"),
    ("source_path", "TEXT"),
    ("in_directory", "INTEGER DEFAULT 0"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]

INDEXES: List[Tuple[str, str]] = [
    ("idx_clean_title", "clean_title"),
    ("idx_domain", "domain"),
    ("idx_subject", "subject"),
    ("idx_media_type", "media_type"),
    ("idx_in_directory", "in_directory"),
    ("idx_file_type", "file_type"),
]

SAMPLE_RECORDS: List[Dict[str, Any]] = [
    {
        "clean_title": "Electric Motors and Control Systems",
        "subtitle": "Industrial and Applied Motor Controls",
        "raw_title": "[Frank D. Petruzella] Electric Motors.pdf",
        "author": "Frank D. Petruzella",
        "publisher": "McGraw-Hill",
        "published": "2010",
        "edition": "1st Edition",
        "revision": 1,
        "language": "en",
        "domain": "Engineering & Applied Physics",
        "subject": "Electrical & Embedded Engineering",
        "subfield": "Circuit Design, Embedded Systems & Hardware",
        "genre": "Textbook / Technical Reference",
        "topics": "electronics, electrical-engineering, circuits, motors, hardware",
        "key_concepts": "Analog & Digital Circuits, Motor Controls, Industrial Systems",
        "target_audience": "Intermediate / Practitioner",
        "media_type": "Book / Ebook",
        "is_collection": 0,
        "summary": "Comprehensive guide covering electric motor principles, control circuits, and industrial troubleshooting.",
        "file_type": "PDF",
        "file_extension": ".pdf",
        "source_path": "/Volumes/Drive2/Books_2/[Frank D. Petruzella] Electric Motors.pdf",
        "in_directory": 1,
    },
    {
        "clean_title": "Python for Data Science and Machine Learning Bootcamp",
        "subtitle": "Comprehensive Hands-On Machine Learning in Python",
        "raw_title": "[FreeCourseLab.com] Udemy - Python for Data Science and Machine Learning Bootcamp",
        "author": "Jose Portilla",
        "publisher": "Udemy",
        "published": "2024",
        "edition": "Bootcamp",
        "revision": 1,
        "language": "en",
        "domain": "Computer Science & Artificial Intelligence",
        "subject": "Data Science & Statistical Analysis",
        "subfield": "Applied Data Science & Mining",
        "genre": "Courseware / Video Curriculum",
        "topics": "data-science, python, statistics, analytics, machine-learning",
        "key_concepts": "Exploratory Data Analysis, Statistical Modeling, Feature Extraction, Scikit-Learn",
        "target_audience": "Intermediate / Practitioner",
        "media_type": "Course / Video Series",
        "is_collection": 0,
        "summary": "A full video course curriculum focusing on Python data analysis, NumPy, Pandas, Matplotlib, and Scikit-Learn.",
        "file_type": "Directory / Folder",
        "file_extension": "",
        "source_path": "/Volumes/Drive2/Books_2/[FreeCourseLab.com] Udemy - Python for Data Science",
        "in_directory": 1,
    },
    {
        "clean_title": "Project Manifest: The Cybernetic Instrument & Advanced Diagnostic Ecosystem",
        "subtitle": "System Specification and Architectural Blueprint",
        "raw_title": "📂 Project Manifest_ The Cybernetic Instrument & Advanced Diagnostic Ecosystem.md",
        "author": "System Architect",
        "publisher": "Internal",
        "published": "2026",
        "edition": "v1.0",
        "revision": 1,
        "language": "en",
        "domain": "Systems Architecture & Engineering",
        "subject": "Systems Architecture & Ecosystems",
        "subfield": "Cybernetic Diagnostics & Ecosystem Architecture",
        "genre": "Project Documentation",
        "topics": "system-architecture, cybernetics, diagnostics, technical-specification",
        "key_concepts": "Cybernetic Feedback Loops, Diagnostic Pipelines, Multi-Agent Instrumentation",
        "target_audience": "Architect / Engineer",
        "media_type": "Project Documentation / Manifest",
        "is_collection": 0,
        "summary": "Technical blueprint detailing cybernetic telemetry, feedback loops, and automated diagnostics for the platform.",
        "file_type": "Markdown",
        "file_extension": ".md",
        "source_path": "/Volumes/Drive2/Books_2/Project Manifest.md",
        "in_directory": 1,
    },
    {
        "clean_title": "130+ Math Learning Books",
        "subtitle": "Curated Mathematics Resource Pack",
        "raw_title": "130+ Math Learning Books",
        "author": "Various Authors",
        "publisher": "Academic Compendium",
        "published": "2023",
        "edition": "Compendium",
        "revision": 1,
        "language": "en",
        "domain": "Formal Sciences & Mathematics",
        "subject": "Mathematics & Applied Statistics",
        "subfield": "Foundational & Advanced Mathematics",
        "genre": "Anthology / Resource Collection",
        "topics": "mathematics, statistics, calculus, linear-algebra, probability",
        "key_concepts": "Vector Spaces, Differential Calculus, Linear Algebra, Probability Theory",
        "target_audience": "Academic / Practitioner",
        "media_type": "Collection / Resource Pack",
        "is_collection": 1,
        "summary": "Curated repository collection comprising over 130 foundational and advanced mathematics textbooks and references.",
        "file_type": "Directory / Folder",
        "file_extension": "",
        "source_path": "/Volumes/Drive2/Books_2/130+ Math Learning Books",
        "in_directory": 1,
    },
]


def init_database(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    drop_existing: bool = False,
    table_name: str = "books",
) -> Dict[str, Any]:
    """
    Initializes the SQLite database with the full schema and indexes.

    Args:
        db_path: Path to the SQLite database file.
        drop_existing: If True, drops existing table before creation.
        table_name: Name of the table to initialize (default: 'books').

    Returns:
        Dictionary containing status and schema statistics.
    """
    db_file = Path(db_path)
    if db_file.parent and not db_file.parent.exists():
        db_file.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        if drop_existing:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        col_defs = ", ".join([f"{name} {dtype}" for name, dtype in SCHEMA_COLUMNS])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

        # Create performance and search indexes
        for idx_name, col in INDEXES:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col});")

        # Create trigger for automatic updated_at timestamp updating
        trigger_name = f"trg_{table_name}_updated_at"
        cursor.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {trigger_name}
            AFTER UPDATE ON {table_name}
            FOR EACH ROW
            BEGIN
                UPDATE {table_name} SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
            """
        )

        conn.commit()

        # Query metadata
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

    return {
        "db_path": str(db_file.resolve()),
        "table_name": table_name,
        "column_count": len(columns),
        "columns": columns,
        "row_count": row_count,
        "indexes_count": len(INDEXES),
        "status": "initialized",
    }


def seed_sample_records(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    table_name: str = "books",
    records: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Seeds the database with representative sample records for testing.

    Returns:
        Number of inserted / seeded records.
    """
    db_file = Path(db_path)
    to_insert = records or SAMPLE_RECORDS

    inserted = 0
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        for rec in to_insert:
            cols = list(rec.keys())
            placeholders = ", ".join(["?"] * len(cols))
            update_clauses = [f"{c} = excluded.{c}" for c in cols if c != "clean_title"]

            sql = f"""
                INSERT INTO {table_name} ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(clean_title) DO UPDATE SET
                    {', '.join(update_clauses)}
            """
            cursor.execute(sql, list(rec.values()))
            inserted += 1
        conn.commit()

    return inserted


def verify_database(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    table_name: str = "books",
) -> Dict[str, Any]:
    """
    Runs schema integrity, constraint, and index verification tests on the database.

    Returns:
        Dictionary with test results and validation status.
    """
    db_file = Path(db_path)
    if not db_file.is_file():
        return {"valid": False, "error": f"Database file '{db_path}' does not exist"}

    results: Dict[str, Any] = {
        "valid": True,
        "tests": {},
        "errors": [],
    }

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # 1. PRAGMA integrity_check
        cursor.execute("PRAGMA integrity_check")
        integrity_row = cursor.fetchone()
        integrity_passed = integrity_row and integrity_row[0] == "ok"
        results["tests"]["sqlite_integrity"] = "PASSED" if integrity_passed else "FAILED"
        if not integrity_passed:
            results["valid"] = False
            results["errors"].append(f"Integrity check failed: {integrity_row}")

        # 2. Schema Column Verification
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {row[1] for row in cursor.fetchall()}
        expected_cols = {name for name, _ in SCHEMA_COLUMNS}
        missing_cols = expected_cols - existing_cols

        results["tests"]["column_schema"] = "PASSED" if not missing_cols else "FAILED"
        results["tests"]["total_columns"] = len(existing_cols)
        if missing_cols:
            results["valid"] = False
            results["errors"].append(f"Missing columns: {sorted(list(missing_cols))}")

        # 3. Index Verification
        cursor.execute(f"PRAGMA index_list({table_name})")
        existing_indexes = {row[1] for row in cursor.fetchall()}
        expected_indexes = {idx_name for idx_name, _ in INDEXES}
        missing_indexes = expected_indexes - existing_indexes

        results["tests"]["indexes"] = "PASSED" if not missing_indexes else "FAILED"
        results["tests"]["total_indexes"] = len(existing_indexes)
        if missing_indexes:
            results["valid"] = False
            results["errors"].append(f"Missing indexes: {sorted(list(missing_indexes))}")

        # 4. Trigger Verification
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        existing_triggers = {row[0] for row in cursor.fetchall()}
        trigger_name = f"trg_{table_name}_updated_at"
        results["tests"]["updated_at_trigger"] = "PASSED" if trigger_name in existing_triggers else "FAILED"

        # 5. Row Count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        results["row_count"] = cursor.fetchone()[0]

    return results


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize and verify SQLite database for Media & ML Library."
    )
    parser.add_argument(
        "--db-path",
        "-d",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Drop existing table and re-initialize a fresh schema",
    )
    parser.add_argument(
        "--seed",
        "-s",
        action="store_true",
        help="Seed database with diverse sample media records for immediate testing",
    )
    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Run comprehensive schema integrity and constraint tests",
    )
    parser.add_argument(
        "--export-csv",
        "-e",
        type=Path,
        default=None,
        help="Export initialized/seeded database to CSV file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Run in quiet mode with minimal console output",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    db_path = parsed.db_path

    if not parsed.quiet:
        print("=" * 60)
        print("         MEDIA & ML LIBRARY DATABASE INITIALIZER")
        print("=" * 60)
        print(f"Target Database: '{db_path.resolve()}'")
        if parsed.force:
            print("Mode: Fresh initialization (--force enabled, dropping existing table)")
        else:
            print("Mode: Safe initialization (preserving or creating schema)")

    try:
        init_stats = init_database(db_path=db_path, drop_existing=parsed.force)
        if not parsed.quiet:
            print(f"✓ Schema successfully initialized: {init_stats['column_count']} columns, {init_stats['indexes_count']} indexes.")

        if parsed.seed:
            seeded_count = seed_sample_records(db_path=db_path)
            if not parsed.quiet:
                print(f"✓ Seeded {seeded_count} sample records into '{db_path.name}'.")

        # Always run verification if requested or after seeding/init
        if parsed.verify or not parsed.quiet:
            v_report = verify_database(db_path=db_path)
            if v_report["valid"]:
                if not parsed.quiet:
                    print(f"✓ Database verification PASSED ({v_report['row_count']} total records).")
                    for t_name, t_res in v_report["tests"].items():
                        print(f"    - {t_name}: {t_res}")
            else:
                print("✗ Database verification FAILED:", file=sys.stderr)
                for err in v_report["errors"]:
                    print(f"    - {err}", file=sys.stderr)
                return 1

        if parsed.export_csv:
            from compare_books import export_db_to_csv
            export_db_to_csv(db_path=db_path, output_path=parsed.export_csv)
            if not parsed.quiet:
                print(f"✓ Exported database to CSV: '{parsed.export_csv}'")

        if not parsed.quiet:
            print("=" * 60)
            print("Database is fully initialized, indexed, and ready for use.")
            print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
