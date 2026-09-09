from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re
import sys
import pandas as pd

from compare_books import clean_title, strip_all_extensions

DEFAULT_DIRECTORY = Path("/Volumes/Drive2/Books_2/")
DEFAULT_BOOK_LIST = Path("_Book_list")

DOUBLE_EXT_PATTERN = re.compile(r"(\.[a-zA-Z0-9]+)\1+$")


def extract_extension_and_stem(raw_name: str) -> Tuple[str, str]:
    """Extracts the base stem and standardized lowercase extension, handling double extensions."""
    text = raw_name.strip()
    # Handle double extensions like .pdf.pdf
    text = DOUBLE_EXT_PATTERN.sub(r"\1", text)
    path = Path(text)
    extension = path.suffix.lower()
    stem = path.stem if extension else text
    return stem, extension


def clean_filename_record(raw_name: str) -> Dict[str, str]:
    """
    Parses and sanitizes a filename according to Home Central Command rules.
    Returns a dictionary with 'Original Filename', 'Cleaned Title', and 'Extension'.
    """
    raw_str = raw_name.strip()
    stem, extension = extract_extension_and_stem(raw_str)
    cleaned_title = clean_title(raw_name)

    return {
        "Original Filename": raw_name,
        "Cleaned Title": cleaned_title,
        "Extension": extension,
    }


def create_books_dataframe(filenames: Iterable[str]) -> pd.DataFrame:
    """Converts a collection of raw filenames into a structured pandas DataFrame."""
    records = [
        clean_filename_record(name)
        for name in filenames
        if name and not name.strip().startswith("#")
    ]
    return pd.DataFrame(records)


def get_titles_from_file(file_path: Path) -> List[str]:
    """Reads raw title strings from a line-separated text file."""
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: File not found at '{path}'.")
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def get_titles_from_directory(directory_path: Path) -> List[str]:
    """Retrieves file names from the specified directory if it exists."""
    target_dir = Path(directory_path)
    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a valid directory.")
        return []
    return [entry.name for entry in target_dir.iterdir() if entry.is_file()]


def load_titles(source_path: Path) -> List[str]:
    """Loads titles from either a file or directory based on source path type."""
    if source_path.is_file():
        return get_titles_from_file(source_path)
    return get_titles_from_directory(source_path)


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    elif DEFAULT_BOOK_LIST.exists():
        target_path = DEFAULT_BOOK_LIST
    else:
        target_path = DEFAULT_DIRECTORY

    print(f"Loading titles from: '{target_path}'")
    raw_titles = load_titles(target_path)

    if not raw_titles:
        print("No files found to process.")
        return

    df = create_books_dataframe(raw_titles)
    print(f"Processed {len(df)} title(s):\n")
    print(df)


if __name__ == "__main__":
    main()
