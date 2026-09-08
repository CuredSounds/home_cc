from pathlib import Path
from typing import List, Optional

DEFAULT_DIRECTORY = Path("/Volumes/Drive2/Books_2/")


def clean_titles(raw_titles: List[str]) -> List[str]:
    """Sanitizes raw filenames by removing file extensions and trimming whitespace."""
    return [Path(title).stem.strip() for title in raw_titles if title]


def get_titles(directory_path: Path = DEFAULT_DIRECTORY) -> Optional[List[str]]:
    """Retrieves file names from the specified directory if it exists."""
    target_dir = Path(directory_path)

    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a valid directory.")
        return None

    return [entry.name for entry in target_dir.iterdir() if entry.is_file()]

from pathlib import Path
from typing import List, Optional
import sys

DEFAULT_DIRECTORY = Path("/Volumes/Drive2/Books_2/")


def clean_titles(raw_titles: List[str]) -> List[str]:
    """Sanitizes raw filenames by removing file extensions and trimming whitespace."""
    return [Path(title).stem.strip() for title in raw_titles if title]


def get_titles(directory_path: Path = DEFAULT_DIRECTORY) -> Optional[List[str]]:
    """Retrieves file names from the specified directory if it exists."""
    target_dir = Path(directory_path)

    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a valid directory.")
        return None

    return [entry.name for entry in target_dir.iterdir() if entry.is_file()]


def main() -> None:
    # Use command-line argument if provided, otherwise default directory
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIRECTORY

    raw_titles = get_titles(target_path)
    if raw_titles is not None:
        cleaned = clean_titles(raw_titles)
        print(f"Found {len(cleaned)} file(s):")
        for title in cleaned:
            print(f"- {title}")


if __name__ == "__main__":
    main()