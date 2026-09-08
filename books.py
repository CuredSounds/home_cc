import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_DIRECTORY = Path("/Volumes/Drive2/Books_2/")
DEFAULT_DB_PATH = Path("books.db")


def init_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initializes the database schema with necessary indexes and constraints."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clean_title TEXT NOT NULL UNIQUE,
                raw_title TEXT NOT NULL,
                source_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_clean_title ON books(clean_title);"
        )
        conn.commit()


def clean_title(raw_name: str) -> str:
    """Sanitizes raw filenames by removing file extensions and trimming whitespace."""
    stem = Path(raw_name).stem.strip()
    return stem if stem else raw_name.strip()


def get_titles_from_directory(directory_path: Path) -> List[Tuple[str, str, str]]:
    """Retrieves file entries directly from a directory."""
    target_dir = Path(directory_path)
    if not target_dir.is_dir():
        print(f"Directory not found: '{target_dir}'")
        return []

    records = []
    for entry in target_dir.iterdir():
        if entry.is_file():
            raw = entry.name
            clean = clean_title(raw)
            if clean:
                records.append((clean, raw, str(entry.resolve())))
    return records


def get_titles_from_file(file_path: Path) -> List[Tuple[str, str, str]]:
    """Parses titles from an existing line-separated text list file."""
    path = Path(file_path)
    if not path.is_file():
        print(f"File not found: '{path}'")
        return []

    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                clean = clean_title(raw)
                if clean:
                    records.append((clean, raw, str(path.resolve())))
    return records


def insert_titles_batch(
    records: Iterable[Tuple[str, str, str]], db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Inserts records in a single batch transaction to maximize throughput."""
    if not records:
        return 0

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR IGNORE INTO books (clean_title, raw_title, source_path)
            VALUES (?, ?, ?)
            """,
            records,
        )
        conn.commit()
        return cursor.rowcount


def main() -> None:
    # 1. Initialize DB
    db_file = DEFAULT_DB_PATH
    init_database(db_file)

    # 2. Extract entries from command-line argument, fallback to _Book_list, or directory
    if len(sys.argv) > 1:
        source_target = Path(sys.argv[1])
    elif Path("_Book_list").exists():
        source_target = Path("_Book_list")
    else:
        source_target = DEFAULT_DIRECTORY

    print(f"Reading from source: '{source_target}'")

    if source_target.is_file():
        records = get_titles_from_file(source_target)
    else:
        records = get_titles_from_directory(source_target)

    print(f"Processed {len(records)} candidate record(s).")

    # 3. Store batch in database
    inserted_count = insert_titles_batch(records, db_file)
    print(
        f"Database updated: {inserted_count} new row(s) inserted into '{db_file}'."
    )


if __name__ == "__main__":
    main()