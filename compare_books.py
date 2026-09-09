"""
Book List and Directory Comparison Tool.

Compares a book list (e.g., _Book_list) against actual files in a directory,
identifying matched titles, missing titles, and unlisted directory files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import wordninja
from titlecase import titlecase
from front_matter import extract_front_matter, is_uninformative_title

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

# Domain-specific terms and acronyms to improve unspaced token segmentation
EXTRA_VOCABULARY = [
    "ai", "ml", "llm", "nlp", "sql", "aws", "api", "iot", "cad", "autocad", "pdf", "csv", "json", "bom",
    "usb", "html", "css", "php", "ui", "ux", "diy", "esp32", "esp8266", "stm32", "arm", "fpga", "dsp",
    "pcb", "rf", "cnc", "ic", "led", "oled", "lcd", "ssmp", "edm", "bacto", "midi", "wifi",
    "os", "dos", "ip", "tcp", "udp", "http", "https", "ssh", "gpu", "cpu", "ram", "rom", "eeprom", "flash",
    "lidar", "gps", "3d", "2d", "4d", "5g", "4g", "lte", "cplusplus", "csharp", "dotnet", "python",
    "javascript", "typescript", "nodejs", "react", "vue", "linux", "ubuntu", "debian", "macos", "ios",
    "android", "docker", "kubernetes", "git", "github", "gitlab", "raspberry", "pi", "arduino", "teensy",
    "oreilly", "mcgraw", "hill", "wiley", "packt", "apress", "manning", "sybex", "nostarch", "pragmatic",
    "devops", "cybersecurity", "datascience", "deeplearning", "machinelearning", "microservices", "blockchain",
    "microprocessor", "synthesizer", "synthesizers", "calisthenics", "bushcraft", "bushcrafting",
    "pentesters", "pentesting", "pentest", "prepper", "preppers", "homesteading", "permaculture",
    "foraging", "forager", "woodworking", "metalworking", "blacksmithing", "lockpicking",
    "electromagnetic", "robotics", "mechatronics", "biomedical", "bioinformatics", "quantum",
    "neuroscience", "astronomy", "astrophysics", "microbiology", "pharmacology", "toxicology",
    "virology", "pathology", "physiology", "immunology", "biochemistry", "histology", "genetics",
    "optics", "thermodynamics", "nanotechnology", "semiconductors", "photonics", "mycology",
    "entomology", "rocketry", "aerospace", "aeronautics", "avionics", "propulsion", "ballistics",
    "firstaid", "paramedic", "trauma", "orienteering", "cordage", "firemaking", "apiculture",
    "beekeeping", "husbandry", "charcuterie", "biohacking", "nootropics", "bodyweight", "kinesiology",
    "biomechanics", "ergonomics", "stoicism", "epistemology", "metaphysics", "econometrics", "copywriting",
    "cartography", "surveying", "mineralogy", "petrology", "meteorology", "climatology", "oceanography",
    "paleontology", "linguistics", "semantics", "etymology", "cryptanalysis", "steganography", "musicology",
    "ansible", "kubernetes", "docker", "pentester", "pentesters", "adversarial", "mitigations",
    "blackanddecker", "cyber", "security", "data", "scientist", "edible", "in", "an", "and", "the",
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "edition", "guide", "manual",
    "handbook", "bible", "cookbook", "pocket", "reference", "basics", "projects", "complete", "ultimate",
    "mastering", "beginning", "advanced", "practical", "applied", "modern", "essential", "essentials",
    "algorithms", "algorithmic", "structures", "engineering", "design", "development", "architecture",
    "systems", "adafruit", "udemy", "backyard", "gardening", "greenhouses", "landscape", "carpentry",
    "plumbing", "masonry", "stonework", "treehouses", "workshops", "furniture", "outdoor", "grills",
    "smokers", "wiring", "clutter", "bathrooms", "midwest", "northeast", "northwest", "homerepair",
    "homeimprovement", "playstructures", "pichkalyov", "reedited", "for", "with", "from", "by", "of"
]

for _w in EXTRA_VOCABULARY:
    wordninja.DEFAULT_LANGUAGE_MODEL._wordcost[_w.lower()] = 0.5
    wordninja.DEFAULT_LANGUAGE_MODEL._maxword = max(
        wordninja.DEFAULT_LANGUAGE_MODEL._maxword, len(_w)
    )

EXACT_WORD_CASING: Dict[str, str] = {
    "ai": "AI",
    "ml": "ML",
    "llm": "LLM",
    "nlp": "NLP",
    "sql": "SQL",
    "aws": "AWS",
    "api": "API",
    "iot": "IoT",
    "cad": "CAD",
    "autocad": "AutoCAD",
    "pdf": "PDF",
    "csv": "CSV",
    "json": "JSON",
    "bom": "BOM",
    "usb": "USB",
    "html": "HTML",
    "css": "CSS",
    "php": "PHP",
    "ui": "UI",
    "ux": "UX",
    "diy": "DIY",
    "esp32": "ESP32",
    "esp8266": "ESP8266",
    "stm32": "STM32",
    "arm": "ARM",
    "fpga": "FPGA",
    "dsp": "DSP",
    "asic": "ASIC",
    "pcb": "PCB",
    "rf": "RF",
    "cnc": "CNC",
    "ic": "IC",
    "led": "LED",
    "oled": "OLED",
    "lcd": "LCD",
    "ssmp": "SSMP",
    "bacto": "Bacto",
    "edm": "EDM",
    "midi": "MIDI",
    "wifi": "Wi-Fi",
    "wi-fi": "Wi-Fi",
    "os": "OS",
    "dos": "DOS",
    "ip": "IP",
    "tcp": "TCP",
    "udp": "UDP",
    "http": "HTTP",
    "https": "HTTPS",
    "ssh": "SSH",
    "gpu": "GPU",
    "cpu": "CPU",
    "ram": "RAM",
    "rom": "ROM",
    "eeprom": "EEPROM",
    "flash": "Flash",
    "lidar": "LiDAR",
    "gps": "GPS",
    "3d": "3D",
    "2d": "2D",
    "4d": "4D",
    "5g": "5G",
    "4g": "4G",
    "lte": "LTE",
    "c++": "C++",
    "c#": "C#",
    ".net": ".NET",
    "dotnet": ".NET",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "node.js": "Node.js",
    "react": "React",
    "vue": "Vue",
    "linux": "Linux",
    "ubuntu": "Ubuntu",
    "debian": "Debian",
    "macos": "macOS",
    "ios": "iOS",
    "android": "Android",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "raspberry": "Raspberry",
    "pi": "Pi",
    "arduino": "Arduino",
    "teensy": "Teensy",
    "adafruit": "Adafruit",
    "oreilly": "O'Reilly",
    "o'reilly": "O'Reilly",
    "mcgraw-hill": "McGraw-Hill",
    "wiley": "Wiley",
    "packt": "Packt",
    "apress": "Apress",
    "manning": "Manning",
    "sybex": "Sybex",
    "devops": "DevOps",
    "cybersecurity": "Cybersecurity",
    "datascience": "Data Science",
    "deeplearning": "Deep Learning",
    "machinelearning": "Machine Learning",
    "microservices": "Microservices",
    "blockchain": "Blockchain",
    "microprocessor": "Microprocessor",
    "synthesizer": "Synthesizer",
    "calisthenics": "Calisthenics",
    "bushcraft": "Bushcraft",
    "pentesters": "Pentesters",
    "pentesting": "Pentesting",
    "pentest": "Pentest",
    "prepper": "Prepper",
    "preppers": "Preppers",
    "blackanddecker": "Black & Decker",
    "ebook": "eBook",
    "bi": "BI",
    "it": "IT",
    "udemy": "Udemy",
    "pichkalyov": "Pichkalyov",
}

LOWERCASE_WORDS: Set[str] = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "as", "at", "by", "for", "in", "of", "off", "on", "per", "to", "up",
    "via", "with", "from", "into", "onto", "over", "than"
}

HYPHENATED_COMPOUNDS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bhands\s*on\b", re.IGNORECASE), "Hands-On"),
    (re.compile(r"\bstep\s*by\s*step\b", re.IGNORECASE), "Step-by-Step"),
    (re.compile(r"\ball\s*in\s*one\b", re.IGNORECASE), "All-in-One"),
    (re.compile(r"\breal\s*time\b", re.IGNORECASE), "Real-Time"),
    (re.compile(r"\bopen\s*source\b", re.IGNORECASE), "Open-Source"),
    (re.compile(r"\bself\s*taught\b", re.IGNORECASE), "Self-Taught"),
    (re.compile(r"\bend\s*to\s*end\b", re.IGNORECASE), "End-to-End"),
    (re.compile(r"\bobject\s*oriented\b", re.IGNORECASE), "Object-Oriented"),
    (re.compile(r"\bfull\s*stack\b", re.IGNORECASE), "Full-Stack"),
    (re.compile(r"\bpeer\s*to\s*peer\b", re.IGNORECASE), "Peer-to-Peer"),
    (re.compile(r"\bstate\s*of\s*the\s*art\b", re.IGNORECASE), "State-of-the-Art"),
    (re.compile(r"\bplug\s*and\s*play\b", re.IGNORECASE), "Plug-and-Play"),
    (re.compile(r"\bcommand\s*line\b", re.IGNORECASE), "Command-Line"),
    (re.compile(r"\bclutter\s*free\b", re.IGNORECASE), "Clutter-Free"),
    (re.compile(r"\buser\s*guide\b", re.IGNORECASE), "User Guide"),
    (re.compile(r"\bcheat\s*sheet\b", re.IGNORECASE), "Cheat Sheet"),
    (re.compile(r"\bauto\s*cad\b", re.IGNORECASE), "AutoCAD"),
    (re.compile(r"\braspberry\s*pi\b", re.IGNORECASE), "Raspberry Pi"),
    (re.compile(r"\bblack\s*and\s*decker\b", re.IGNORECASE), "Black & Decker"),
    (re.compile(r"\b(\d+)\s*(st|nd|rd|th)\s*ed\b", re.IGNORECASE), r"\1\2 Edition"),
    (re.compile(r"\b(\d+)\s*(st|nd|rd|th)\s*edition\b", re.IGNORECASE), r"\1\2 Edition"),
    (re.compile(r"\b3\s*rd\b", re.IGNORECASE), "3rd"),
    (re.compile(r"\b2\s*nd\b", re.IGNORECASE), "2nd"),
    (re.compile(r"\b1\s*st\b", re.IGNORECASE), "1st"),
    (re.compile(r"\b(\d+)\s*th\b", re.IGNORECASE), r"\1th"),
]

BRACKET_PREFIX_PATTERN = re.compile(r"^\[[^\]]+\]\s*")
LEADING_SYMBOLS_PATTERN = re.compile(r"^[📂📁📄\-_*•\s]+")

DEFAULT_LIST_PATH = Path("/Users/sonic.design/Brain/01_Projects/CuredSounds/_app_devs/home_cc/docs/_Book_list")
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
        if suffix and (
            suffix in KNOWN_EXTENSIONS
            or re.match(
                r"^\.(pdf|epub|mobi|azw3?|djvu|fb2|cbr|cbz|docx?|odt|rtf|zip|rar|7z|tar|gz|txt|md|csv|json|py|html?)$",
                suffix,
            )
        ):
            current = p.stem.strip()
        else:
            break
    return current


def strip_copy_suffix(text: str) -> str:
    """
    Strips trailing duplicate copy designations (e.g. ' (1)', ' (2)', ' 2')
    while preserving actual title parts like 'Volume 2', 'Pack 5', 'Part 1'.
    """
    # Strip (1), (2), etc. (1-2 digits only to preserve 4-digit years like '(2026)')
    text = re.sub(r"[\s_]*\([1-9]\d{0,1}\)$", "", text).strip()
    # Strip trailing standalone duplicate copy number like "file 2" or "file_2"
    m = re.search(r"[\s_]+([1-9]\d{0,1})$", text)
    if m:
        prefix = text[:m.start()].strip()
        last_word = prefix.split()[-1].lower().rstrip(".") if prefix.split() else ""
        if last_word not in {
            "volume",
            "vol",
            "part",
            "pack",
            "book",
            "chapter",
            "level",
            "phase",
            "step",
            "grade",
            "class",
            "no",
            "v",
            "edition",
            "issue",
        }:
            return prefix
    return text


def split_camel_case(s: str) -> str:
    """Splits CamelCase words into space-separated words."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s


def should_segment_token(token: str) -> bool:
    """Checks if a lowercase unspaced token should be segmented with wordninja."""
    if len(token) < 7:
        return False
    # Avoid splitting hex hashes or pure numbers
    if re.fullmatch(r"[a-f0-9]{20,}", token, re.IGNORECASE):
        return False
    if token.isdigit():
        return False
    if not token.islower():
        return False
    return True


def segment_token(chunk: str) -> str:
    """Segments concatenated lowercase words using probabilistic word segmentation."""
    if should_segment_token(chunk):
        tokens = wordninja.split(chunk)
        if len(tokens) > 1:
            return " ".join(tokens)
    return chunk


def clean_title(raw_name: str) -> str:
    """
    Sanitizes raw filenames/lines into human-readable, grammatically proper titles
    formatted like titles on Google Books / publisher listings.

    - Strips folder/status emojis, leading symbols, bracketed uploader tags, and extensions.
    - Strips duplicate copy markers, Notion hash suffixes, and preview tags.
    - Resolves concatenated lowercase strings into individual words.
    - Applies standard Title Casing with proper grammar for lowercase prepositions/articles.
    - Preserves exact casing for technical acronyms (e.g. AI, AWS, IoT, CAD, PCB, WiFi).
    - Formats subtitles and hyphenated compound adjectives cleanly.
    """
    text = raw_name.strip()
    if not text:
        return ""

    # 1. Unescape markdown formatting characters
    text = text.replace(r"\!", "!").replace(r"\+", "+").replace(r"\-", "-")

    # 2. Strip leading decorative symbols/emojis
    text = LEADING_SYMBOLS_PATTERN.sub("", text)

    # 3. Strip bracketed uploader / website tags (e.g. [ WebToolTip.com ], [FreeCourseLab.com])
    text = BRACKET_PREFIX_PATTERN.sub("", text)

    # 4. Strip known document/archive extensions
    text = strip_all_extensions(text)

    # 5. Strip duplicate copy suffixes
    text = strip_copy_suffix(text)

    # 6. Strip Notion 32-char hex hash suffixes
    text = re.sub(r"[\s_-]+[a-f0-9]{32}$", "", text, flags=re.IGNORECASE).strip()

    # 7. Strip trailing preview tags
    text = re.sub(r"[\s_-]*_?preview$", "", text, flags=re.IGNORECASE).strip()

    # 8. Strip repository master/main suffix if standalone
    text = re.sub(r"[-_](master|main)$", "", text, flags=re.IGNORECASE).strip()

    # 9. Handle plus signs used as word separators (e.g. "Attractive+Body+Language+BONUS")
    text = re.sub(r"(?<!\d)(?<![Cc])\+(?![+])", " ", text)

    # 10. Normalize subtitle underscores (e.g. "Manifest_ The" -> "Manifest: The")
    text = re.sub(r"([A-Za-z0-9])_\s+([A-Za-z0-9])", r"\1: \2", text)
    text = re.sub(r"([A-Za-z0-9])\s+_\s*([A-Za-z0-9])", r"\1: \2", text)

    # 11. Replace kebab hyphens that join words without spaces (e.g. "ai-engineer" -> "ai engineer")
    text = re.sub(r"(?<=\S)-(?=\S)", " ", text)

    # 12. Split CamelCase if string has no spaces
    if " " not in text and ("_" not in text):
        text = split_camel_case(text)

    # 13. Segment concatenated words and replace remaining underscores with spaces
    chunks = re.split(r"([_\s]+)", text)
    processed_chunks: List[str] = []
    for c in chunks:
        if "_" in c or " " in c:
            processed_chunks.append(" ")
        else:
            processed_chunks.append(segment_token(c))
    text = "".join(processed_chunks)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # 14. Title Case formatting with exact word overrides
    def custom_word(word: str, **kwargs: Any) -> Optional[str]:
        clean_w = word.lower().strip("(),:;\"'?!")
        if clean_w in EXACT_WORD_CASING:
            proper = EXACT_WORD_CASING[clean_w]
            lead_punct = re.match(r"^[\(\"']+", word)
            tail_punct = re.search(r"[\)\",:;!'\.\?]+$", word)
            res = proper
            if lead_punct:
                res = lead_punct.group(0) + res
            if tail_punct:
                res = res + tail_punct.group(0)
            return res
        return None

    formatted = titlecase(text, callback=custom_word)

    # 15. Enforce proper lowercase for prepositions/articles/conjunctions
    words = formatted.split()
    for i, w in enumerate(words):
        w_clean = re.sub(r"[^\w]", "", w).lower()
        if w_clean in LOWERCASE_WORDS:
            # Capitalize if first word, last word, or immediately follows colon/dash/question
            if 0 < i < len(words) - 1:
                prev_w = words[i - 1]
                if not (
                    prev_w.endswith(":")
                    or prev_w.endswith("-")
                    or prev_w.endswith("—")
                    or prev_w.endswith("?")
                    or prev_w.endswith("!")
                ):
                    lead = re.match(r"^[^\w]*", w).group(0)
                    tail = re.search(r"[^\w]*$", w).group(0)
                    words[i] = lead + w_clean + tail
            elif i == 0:
                lead = re.match(r"^[^\w]*", w).group(0)
                tail = re.search(r"[^\w]*$", w).group(0)
                words[i] = lead + w_clean.capitalize() + tail
    formatted = " ".join(words)

    # 16. Apply phrase replacements & compound word casing
    for pattern, repl in HYPHENATED_COMPOUNDS:
        formatted = pattern.sub(repl, formatted)

    # 17. Punctuation spacing & typography normalization
    formatted = re.sub(r"\s*:\s*", ": ", formatted)
    formatted = re.sub(r"\s*,\s*", ", ", formatted)
    formatted = re.sub(r"\s+-\s+", " - ", formatted)
    formatted = re.sub(r"\s*—\s*", " — ", formatted)
    formatted = re.sub(r"\s+", " ", formatted).strip()

    return formatted


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
    normalized = strip_copy_suffix(cleaned).strip()

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
            if (
                not raw
                or raw.startswith("#")
                or raw == "_Book_list"
                or ("@" in raw and ("%" in raw or "$" in raw or ">" in raw))
            ):
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
    dir_path: Union[str, Path],
    recursive: bool = False,
    include_dirs: bool = True,
    scan_front_matter: bool = True,
) -> List[BookRecord]:
    """
    Scans a directory and returns BookRecords for all valid files and optionally directories.
    If scan_front_matter is enabled and a file has an uninformative name (e.g. '20240319.pdf'),
    extracts the true title from embedded PDF/EPUB front matter.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return []

    records: List[BookRecord] = []
    seen_normalized: Set[str] = set()

    iterator = path.rglob("*") if recursive else path.iterdir()

    for entry in iterator:
        # Skip hidden files and macOS metadata files
        if entry.name.startswith(".") or entry.name.startswith("._"):
            continue

        # Include valid files or directories (such as book collections, packs, and course folders)
        if not entry.is_file() and not (include_dirs and entry.is_dir()):
            continue

        raw = entry.name
        cleaned = clean_title(raw)

        # If title is uninformative and entry is a file, inspect front matter
        if scan_front_matter and entry.is_file() and is_uninformative_title(cleaned):
            fm = extract_front_matter(entry)
            if fm.has_useful_title:
                cleaned = clean_title(fm.title)

        norm = normalize_title(cleaned)
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


def reclean_database(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    table_name: str = "books",
) -> int:
    """
    Re-processes and updates all book clean_title entries in the SQLite database
    using the latest natural language cleaning and proper case formatting rules.
    Preserves all existing metadata columns dynamically.
    """
    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols_info = cursor.fetchall()
        col_names = [r[1] for r in cols_info]
        if not col_names or "raw_title" not in col_names:
            return 0

        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id ASC")
        rows = cursor.fetchall()

        cursor.execute(f"DROP TABLE IF EXISTS {table_name}_recleaned")

        # Dynamically build CREATE TABLE statement preserving all column definitions and types
        col_defs = []
        for col_id, col_name, col_type, not_null, default_val, pk in cols_info:
            if pk:
                col_defs.append(f"{col_name} INTEGER PRIMARY KEY AUTOINCREMENT")
            elif col_name == "clean_title":
                col_defs.append("clean_title TEXT NOT NULL UNIQUE")
            elif col_name == "raw_title":
                col_defs.append("raw_title TEXT NOT NULL")
            else:
                default_clause = f" DEFAULT {default_val}" if default_val is not None else ""
                col_defs.append(f"{col_name} {col_type or 'TEXT'}{default_clause}")

        cursor.execute(f"CREATE TABLE {table_name}_recleaned ({', '.join(col_defs)})")

        non_pk_cols = [c for c in col_names if c != "id"]
        placeholders = ", ".join(["?"] * len(non_pk_cols))

        count = 0
        for row in rows:
            row_dict = dict(zip(col_names, row))
            raw = row_dict.get("raw_title") or ""
            new_clean = clean_title(raw)
            if not new_clean:
                continue

            row_dict["clean_title"] = new_clean
            row_dict["updated_at"] = row_dict.get("updated_at")

            vals = [row_dict.get(c) for c in non_pk_cols]

            update_clauses = []
            for c in non_pk_cols:
                if c == "clean_title":
                    continue
                elif c == "in_directory":
                    update_clauses.append("in_directory = MAX(in_directory, excluded.in_directory)")
                elif c == "source_path":
                    update_clauses.append("source_path = COALESCE(excluded.source_path, source_path)")
                elif c == "updated_at":
                    update_clauses.append("updated_at = CURRENT_TIMESTAMP")
                else:
                    update_clauses.append(f"{c} = COALESCE(excluded.{c}, {c})")

            cursor.execute(
                f"""
                INSERT INTO {table_name}_recleaned ({', '.join(non_pk_cols)})
                VALUES ({placeholders})
                ON CONFLICT(clean_title) DO UPDATE SET
                    {', '.join(update_clauses)}
                """,
                vals,
            )
            count += 1

        cursor.execute(f"DROP TABLE {table_name}")
        cursor.execute(f"ALTER TABLE {table_name}_recleaned RENAME TO {table_name}")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_clean_title ON {table_name}(clean_title);")
        conn.commit()

    return count


def export_db_to_csv(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    output_path: Optional[Union[str, Path]] = None,
    table_name: str = "books",
) -> str:
    """
    Exports the contents of an SQLite database table as CSV format.
    If output_path is provided, writes CSV to disk.
    Returns the CSV data as a string.
    """
    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    output_stream = io.StringIO()
    writer = csv.writer(output_stream, lineterminator="\n")

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist or has no columns in {db_file}")

        writer.writerow(columns)

        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id ASC")
        for row in cursor.fetchall():
            writer.writerow(row)

    csv_content = output_stream.getvalue()

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)

    return csv_content


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
        "--no-scan-front-matter",
        action="store_true",
        help="Disable automatic front matter inspection for uninformative filenames",
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
        "--init-db",
        action="store_true",
        help="Initialize the SQLite database schema and indexes (safe idempotency)",
    )
    parser.add_argument(
        "--reclean-db",
        action="store_true",
        help="Re-clean and update all title records in the SQLite database with proper casing and grammar",
    )
    parser.add_argument(
        "--enrich-db",
        action="store_true",
        help="Enrich database records with authors, publishers, ML taxonomy, summaries, and media metadata",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--export-db-csv",
        type=Path,
        nargs="?",
        const=Path("books_db.csv"),
        default=None,
        help="Export SQLite database table to CSV file (default filename: books_db.csv)",
    )
    parser.add_argument(
        "--print-db-csv",
        action="store_true",
        help="Print the entire SQLite database table as CSV directly to standard output",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    # 1. Handle database initialization if requested
    if parsed.init_db:
        try:
            from init_db import init_database, verify_database
            stats = init_database(db_path=parsed.db_path)
            v_report = verify_database(db_path=parsed.db_path)
            print(f"Database '{parsed.db_path}' initialized successfully: {stats['column_count']} columns, {stats['indexes_count']} indexes (integrity: {v_report['tests']['sqlite_integrity']}).")
        except Exception as e:
            print(f"Error initializing database: {e}", file=sys.stderr)
            return 1

    # 2. Handle database re-cleaning if requested
    if parsed.reclean_db:
        try:
            cleaned_count = reclean_database(db_path=parsed.db_path)
            print(f"Database '{parsed.db_path}' updated: {cleaned_count} title record(s) re-cleaned.")
        except Exception as e:
            print(f"Error re-cleaning database: {e}", file=sys.stderr)
            return 1

    # 2. Handle database metadata enrichment if requested
    if parsed.enrich_db:
        try:
            from enrich_books import enrich_database
            stats = enrich_database(db_path=parsed.db_path, export_csv_path=parsed.export_db_csv or DEFAULT_DB_PATH.with_suffix(".csv"))
            print(
                f"Database '{parsed.db_path}' enriched: {stats['total_records']} total records "
                f"({stats['online_enriched']} online, {stats['local_rule_enriched']} local/heuristics)."
            )
        except Exception as e:
            print(f"Error enriching database: {e}", file=sys.stderr)
            return 1

    # 3. Handle direct database print/export requests without requiring directory comparison
    if parsed.print_db_csv:
        try:
            csv_data = export_db_to_csv(db_path=parsed.db_path)
            sys.stdout.write(csv_data)
            sys.stdout.flush()
            return 0
        except BrokenPipeError:
            # Piped into a pager or tool (e.g. head/grep) that closed stdout
            sys.stderr.close()
            return 0
        except Exception as e:
            print(f"Error printing DB to CSV: {e}", file=sys.stderr)
            return 1

    if parsed.export_db_csv and not parsed.sync_db:
        try:
            export_db_to_csv(db_path=parsed.db_path, output_path=parsed.export_db_csv)
            print(f"Exported database table to CSV: {Path(parsed.export_db_csv).resolve()}")
            return 0
        except Exception as e:
            print(f"Error exporting DB to CSV: {e}", file=sys.stderr)
            return 1

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
        dir_records = load_directory_titles(
            dir_path,
            recursive=parsed.recursive,
            scan_front_matter=not parsed.no_scan_front_matter,
        )
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

    # Database CSV Export
    if parsed.export_db_csv:
        try:
            export_db_to_csv(db_path=parsed.db_path, output_path=parsed.export_db_csv)
            print(f"Exported database table to CSV: {Path(parsed.export_db_csv).resolve()}")
        except Exception as e:
            print(f"Error exporting DB to CSV: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
