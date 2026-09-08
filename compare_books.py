"""
Book List and Directory Comparison Tool.

Compares a book list (e.g., _Book_list) against actual files in a directory,
identifying matched titles, missing titles, and unlisted directory files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# Common book/document/media extensions to strip during title cleaning
KNOWN_EXTENSIONS: Set[str] = {
    ".pdf",
    ".epub",
    ".mobi",
    ".azw",
    ".azw3",
    ".djvu",
    ".fb2",
    ".ibooks",
    ".cbr",
    ".cbz",
    ".txt",
    ".rtf",
    ".doc",
    ".docx",
    ".odt",
    ".md",
    ".markdown",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".csv",
    ".json",
    ".py",
    ".html",
    ".htm",
}

DEFAULT_LIST_PATH = Path("_Book_list")
DEFAULT_DIRECTORY = Path("/Volumes/Drive2/Books_2/")
DEFAULT_DB_PATH = Path("books.db")


@dataclass(frozen=True)
class BookRecord:
    clean_title: str
    raw_title: str
    normalized_title: str
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_title": self.clean_title,
            "raw_title": self.raw_title,
            "normalized_title": self.normalized_title,
            "source_path": self.source_path,
        }


@dataclass
class MatchedPair:
    list_record: BookRecord
    dir_record: BookRecord

    def to_dict(self) -> Dict[str, Any]:
        return {
            "list_title": self.list_record.clean_title,
            "list_raw": self.list_record.raw_title,
            "dir_title": self.dir_record.clean_title,
            "dir_raw": self.dir_record.raw_title,
            "dir_path": self.dir_record.source_path,
        }


@dataclass
class ComparisonResult:
    total_list_items: int
    total_dir_items: int
    matched: List[MatchedPair] = field(default_factory=list)
    missing_from_dir: List[BookRecord] = field(default_factory=list)
    unlisted_in_dir: List[BookRecord] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def missing_count(self) -> int:
        return len(self.missing_from_dir)

    @property
    def unlisted_count(self) -> int:
        return len(self.unlisted_in_dir)

    @property
    def match_rate(self) -> float:
        if self.total_list_items == 0:
            return 0.0
        return (self.matched_count / self.total_list_items) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_list_items": self.total_list_items,
                "total_dir_items": self.total_dir_items,
                "matched_count": self.matched_count,
                "missing_count": self.missing_count,
                "unlisted_count": self.unlisted_count,
                "match_rate_percent": round(self.match_rate, 2),
            },
            "matched": [m.to_dict() for m in self.matched],
            "missing_from_dir": [r.to_dict() for r in self.missing_from_dir],
            "unlisted_in_dir": [r.to_dict() for r in self.unlisted_in_dir],
        }


def strip_all_extensions(filename: str) -> str:
    """Repeatedly removes known extensions or trailing file extensions."""
    current = filename.strip()
    while True:
        p = Path(current)
        suffix = p.suffix.lower()
        if suffix and (suffix in KNOWN_EXTENSIONS or re.match(r"^\.[a-z0-9]{2,5}$", suffix)):
            current = p.stem.strip()
        else:
            break
    return current


def clean_title(raw_name: str) -> str:
    """
    Sanitizes raw filenames/lines into human-readable clean titles.
    - Strips folder/status emojis (e.g. 📂).
    - Removes file extensions.
    - Trims leading/trailing whitespace and dashes.
    """
    text = raw_name.strip()
    if not text:
        return ""

    # Remove leading decorative symbols/emojis
    text = re.sub(r"^[📂📁📄\-_*•\s]+", "", text)
    # Strip file extensions
    cleaned = strip_all_extensions(text)
    # Strip residual whitespace
    return cleaned.strip()


def normalize_title(raw_name: str) -> str:
    """
    Produces a strict normalized key for case-insensitive, punctuation-insensitive matching.
    - Cleans extensions and prefixes.
    - Strips duplicate copy markers (e.g. ' (1)', ' 2', '_1').
    - Converts to lowercase.
    - Replaces non-alphanumeric characters with spaces.
    - Condenses multiple whitespace characters.
    """
    cleaned = clean_title(raw_name)
    if not cleaned:
        return ""

    # Strip duplicate copy suffixes like ' (1)', ' (2)', ' 2', '_1' at end of string
    normalized = re.sub(r"[\s_]+(\(\d+\)|\d+)$", "", cleaned).strip()

    # Lowercase
    normalized = normalized.lower()

    # Replace punctuation / non-alphanumeric with spaces
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    # Condense whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized or cleaned.lower().strip()


def load_list_titles(file_path: Union[str, Path]) -> List[BookRecord]:
    """
    Parses titles from a text file, ignoring empty lines and comments.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")

    records: List[BookRecord] = []
    seen_normalized: Set[str] = set()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            # Ignore empty lines, pure comment lines, or file header echo
            if not raw or raw.startswith("#") or raw == "_Book_list":
                continue

            cleaned = clean_title(raw)
            norm = normalize_title(raw)
            if not norm:
                continue

            # Avoid adding exact normalized duplicates within the same list
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                records.append(
                    BookRecord(
                        clean_title=cleaned or raw,
                        raw_title=raw,
                        normalized_title=norm,
                        source_path=str(path.resolve()),
                    )
                )

    return records


def load_directory_titles(
    dir_path: Union[str, Path], recursive: bool = False
) -> List[BookRecord]:
    """
    Scans a directory and returns BookRecords for all valid files.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return []

    records: List[BookRecord] = []
    seen_normalized: Set[str] = set()

    iterator = path.rglob("*") if recursive else path.iterdir()

    for entry in iterator:
        if not entry.is_file():
            continue
        # Skip hidden files and macOS metadata files
        if entry.name.startswith(".") or entry.name.startswith("._"):
            continue

        raw = entry.name
        cleaned = clean_title(raw)
        norm = normalize_title(raw)
        if not norm:
            continue

        if norm not in seen_normalized:
            seen_normalized.add(norm)
            records.append(
                BookRecord(
                    clean_title=cleaned or raw,
                    raw_title=raw,
                    normalized_title=norm,
                    source_path=str(entry.resolve()),
                )
            )

    return records


def compare_sources(
    list_records: List[BookRecord], dir_records: List[BookRecord]
) -> ComparisonResult:
    """
    Compares records from list and directory sources.
    Categorizes into matched, missing_from_dir, and unlisted_in_dir.
    """
    dir_by_norm: Dict[str, BookRecord] = {r.normalized_title: r for r in dir_records}
    list_by_norm: Dict[str, BookRecord] = {r.normalized_title: r for r in list_records}

    matched: List[MatchedPair] = []
    missing_from_dir: List[BookRecord] = []
    unlisted_in_dir: List[BookRecord] = []

    for list_rec in list_records:
        if list_rec.normalized_title in dir_by_norm:
            matched.append(
                MatchedPair(
                    list_record=list_rec,
                    dir_record=dir_by_norm[list_rec.normalized_title],
                )
            )
        else:
            missing_from_dir.append(list_rec)

    for dir_rec in dir_records:
        if dir_rec.normalized_title not in list_by_norm:
            unlisted_in_dir.append(dir_rec)

    return ComparisonResult(
        total_list_items=len(list_records),
        total_dir_items=len(dir_records),
        matched=matched,
        missing_from_dir=missing_from_dir,
        unlisted_in_dir=unlisted_in_dir,
    )


def generate_report(
    result: ComparisonResult, verbose: bool = False, max_preview: int = 10
) -> str:
    """
    Generates a structured human-readable comparison report.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("           BOOK LIST VS DIRECTORY COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append(f"Total Unique Titles in List:      {result.total_list_items}")
    lines.append(f"Total Unique Files in Directory:   {result.total_dir_items}")
    lines.append("-" * 60)
    lines.append(f"Matched Titles (Available):       {result.matched_count} ({result.match_rate:.1f}%)")
    lines.append(f"Missing from Directory:           {result.missing_count}")
    lines.append(f"Unlisted Files in Directory:      {result.unlisted_count}")
    lines.append("=" * 60)

    if result.missing_from_dir:
        lines.append(f"\n[!] Missing from Directory ({result.missing_count} titles):")
        items = result.missing_from_dir if verbose else result.missing_from_dir[:max_preview]
        for item in items:
            lines.append(f"  - {item.clean_title}")
        if not verbose and result.missing_count > max_preview:
            lines.append(f"  ... and {result.missing_count - max_preview} more (use --verbose to see all)")

    if result.unlisted_in_dir:
        lines.append(f"\n[+] Unlisted Files in Directory ({result.unlisted_count} files):")
        items = result.unlisted_in_dir if verbose else result.unlisted_in_dir[:max_preview]
        for item in items:
            lines.append(f"  + {item.clean_title}")
        if not verbose and result.unlisted_count > max_preview:
            lines.append(f"  ... and {result.unlisted_count - max_preview} more (use --verbose to see all)")

    if result.matched and verbose:
        lines.append(f"\n[✓] Matched Titles ({result.matched_count} titles):")
        for m in result.matched:
            lines.append(f"  ✓ {m.list_record.clean_title} -> {Path(m.dir_record.source_path or '').name}")

    return "\n".join(lines)


def export_reports(
    result: ComparisonResult,
    export_dir: Optional[Union[str, Path]] = None,
    export_json_path: Optional[Union[str, Path]] = None,
    export_missing_path: Optional[Union[str, Path]] = None,
    export_unlisted_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Exports comparison results to text and/or JSON files.
    Returns dictionary of exported file paths.
    """
    exported: Dict[str, str] = {}

    if export_dir:
        target_dir = Path(export_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if not export_missing_path:
            export_missing_path = target_dir / "missing_books.txt"
        if not export_unlisted_path:
            export_unlisted_path = target_dir / "unlisted_books.txt"

    if export_missing_path:
        missing_file = Path(export_missing_path)
        missing_file.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_file, "w", encoding="utf-8") as f:
            for item in result.missing_from_dir:
                f.write(f"{item.clean_title}\n")
        exported["missing"] = str(missing_file.resolve())

    if export_unlisted_path:
        unlisted_file = Path(export_unlisted_path)
        unlisted_file.parent.mkdir(parents=True, exist_ok=True)
        with open(unlisted_file, "w", encoding="utf-8") as f:
            for item in result.unlisted_in_dir:
                f.write(f"{item.clean_title}\n")
        exported["unlisted"] = str(unlisted_file.resolve())

    if export_json_path:
        json_file = Path(export_json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        exported["json"] = str(json_file.resolve())

    return exported


def sync_to_database(
    result: ComparisonResult, db_path: Union[str, Path] = DEFAULT_DB_PATH
) -> Dict[str, int]:
    """
    Synchronizes comparison availability into the SQLite database.
    Updates or inserts records and sets in_directory status.
    """
    db_file = Path(db_path)
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clean_title TEXT NOT NULL UNIQUE,
                raw_title TEXT NOT NULL,
                source_path TEXT,
                in_directory INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Check if in_directory column exists (in case table already existed previously without it)
        cursor.execute("PRAGMA table_info(books)")
        columns = [row[1] for row in cursor.fetchall()]
        if "in_directory" not in columns:
            cursor.execute("ALTER TABLE books ADD COLUMN in_directory INTEGER DEFAULT 0")
        if "updated_at" not in columns:
            cursor.execute("ALTER TABLE books ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_clean_title ON books(clean_title);")

        matched_updates = 0
        inserted_records = 0

        # 1. Update matched records
        for pair in result.matched:
            cursor.execute(
                """
                INSERT INTO books (clean_title, raw_title, source_path, in_directory, updated_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(clean_title) DO UPDATE SET
                    in_directory = 1,
                    source_path = excluded.source_path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    pair.list_record.clean_title,
                    pair.list_record.raw_title,
                    pair.dir_record.source_path,
                ),
            )
            matched_updates += 1

        # 2. Update missing records
        for missing in result.missing_from_dir:
            cursor.execute(
                """
                INSERT INTO books (clean_title, raw_title, source_path, in_directory, updated_at)
                VALUES (?, ?, ?, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(clean_title) DO UPDATE SET
                    in_directory = 0,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    missing.clean_title,
                    missing.raw_title,
                    missing.source_path,
                ),
            )
            inserted_records += 1

        # 3. Insert unlisted directory files
        for unlisted in result.unlisted_in_dir:
            cursor.execute(
                """
                INSERT INTO books (clean_title, raw_title, source_path, in_directory, updated_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(clean_title) DO UPDATE SET
                    in_directory = 1,
                    source_path = excluded.source_path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    unlisted.clean_title,
                    unlisted.raw_title,
                    unlisted.source_path,
                ),
            )

        conn.commit()

    return {
        "matched_synced": matched_updates,
        "missing_synced": len(result.missing_from_dir),
        "unlisted_synced": len(result.unlisted_in_dir),
    }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare book list with directory contents and report differences."
    )
    parser.add_argument(
        "--list",
        "-l",
        type=Path,
        default=DEFAULT_LIST_PATH,
        help=f"Path to the book list text file (default: {DEFAULT_LIST_PATH})",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help=f"Path to the books directory (default: {DEFAULT_DIRECTORY})",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively scan the target directory and subdirectories",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full detailed lists instead of short previews",
    )
    parser.add_argument(
        "--export-dir",
        "-e",
        type=Path,
        default=None,
        help="Directory to save missing_books.txt and unlisted_books.txt",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Path to export the full comparison result as JSON",
    )
    parser.add_argument(
        "--export-missing",
        type=Path,
        default=None,
        help="Path to export missing book titles as a text file",
    )
    parser.add_argument(
        "--export-unlisted",
        type=Path,
        default=None,
        help="Path to export unlisted directory file titles as a text file",
    )
    parser.add_argument(
        "--sync-db",
        action="store_true",
        help="Synchronize results and availability status into SQLite database",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    list_path = parsed.list
    dir_path = parsed.dir

    if not list_path.exists():
        print(f"Error: Book list file '{list_path}' not found.", file=sys.stderr)
        return 1

    print(f"Loading book list from: '{list_path}'...")
    try:
        list_records = load_list_titles(list_path)
    except Exception as e:
        print(f"Error loading book list: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(list_records)} unique title(s) from list.")

    if not dir_path.exists():
        print(f"Warning: Directory '{dir_path}' does not exist or is not mounted.", file=sys.stderr)
        dir_records = []
    else:
        print(f"Scanning directory: '{dir_path}' (recursive={parsed.recursive})...")
        dir_records = load_directory_titles(dir_path, recursive=parsed.recursive)
        print(f"Found {len(dir_records)} unique file(s) in directory.")

    result = compare_sources(list_records, dir_records)

    # Print human-readable report
    report = generate_report(result, verbose=parsed.verbose)
    print("\n" + report + "\n")

    # Exports
    exported = export_reports(
        result,
        export_dir=parsed.export_dir,
        export_json_path=parsed.export_json,
        export_missing_path=parsed.export_missing,
        export_unlisted_path=parsed.export_unlisted,
    )
    for export_type, file_path in exported.items():
        print(f"Exported {export_type} report to: {file_path}")

    # Database sync
    if parsed.sync_db:
        sync_stats = sync_to_database(result, db_path=parsed.db_path)
        print(
            f"Database '{parsed.db_path}' synchronized successfully: "
            f"{sync_stats['matched_synced']} matched, {sync_stats['missing_synced']} missing, "
            f"{sync_stats['unlisted_synced']} unlisted."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
