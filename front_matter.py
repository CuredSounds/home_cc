"""
Front Matter & Embedded Metadata Extractor for Books and Technical Media.

Extracts titles, authors, publishers, publication dates, editions, ISBNs, and abstracts
directly from the internal structures of PDF, EPUB, and document files.
Provides seamless recovery for uninformative/obfuscated filenames (e.g., '20240319.pdf',
hashes, UUIDs, daily Packt Free Learning downloads).
"""

from __future__ import annotations

import html
import io
import json
import logging
import os
import re
import ssl
import urllib.parse
import urllib.request
import warnings
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Suppress verbose pypdf warnings during batch scanning
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pypdf")

# SSL Context Setup with fallback
SSL_CONTEXT = ssl.create_default_context()
try:
    import certifi
    SSL_CONTEXT.load_verify_locations(certifi.where())
except Exception:
    SSL_CONTEXT = ssl._create_unverified_context()

USER_AGENT = "BookCatalogFrontMatterScanner/1.0 (https://github.com/curedsounds/home_cc)"

# Uninformative title patterns (dates, timestamps, hashes, generic names)
UNINFORMATIVE_PATTERNS = [
    re.compile(r"^\d{6,14}$"),                           # e.g. 20240319, 12251123
    re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}$"),            # e.g. 2024-03-19, 2024_03_19
    re.compile(r"^[0-9a-fA-F]{16,}$"),                   # Hashes / UUIDs
    re.compile(r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$"),  # UUID
    re.compile(r"^(book|ebook|document|doc|untitled|scan|file|temp|download|sample|test|pdf|free[-_]?learning)$", re.I),
    re.compile(r"^packt[-_]?\d{6,14}$", re.I),
]

ISBN_13_PATTERN = re.compile(r"\b(97[89][-\s]?[0-9]{1,5}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1,7}[-\s]?[0-9])\b")
ISBN_10_PATTERN = re.compile(r"\b([0-9]{1,5}[-\s]?[0-9]{1,7}[-\s]?[0-9]{1,7}[-\s]?[0-9xX])\b")

PUBLISHER_PATTERNS = [
    ("Packt Publishing", re.compile(r"\b(packt\s+publishing|packtpub|packt>|packt)\b", re.I)),
    ("O'Reilly Media", re.compile(r"\b(o'reilly\s+media|o'reilly|oreilly)\b", re.I)),
    ("Manning Publications", re.compile(r"\b(manning\s+publications|manning)\b", re.I)),
    ("Wiley", re.compile(r"\b(john\s+wiley\s+&\s+sons|wiley)\b", re.I)),
    ("Apress", re.compile(r"\b(apress|springer\s+nature\s+apress)\b", re.I)),
    ("No Starch Press", re.compile(r"\b(no\s+starch\s+press|nostarch)\b", re.I)),
    ("McGraw-Hill", re.compile(r"\b(mcgraw[- ]?hill|mcgraw[- ]?hill\s+education)\b", re.I)),
    ("Addison-Wesley", re.compile(r"\b(addison[- ]?wesley|pearson\s+education)\b", re.I)),
    ("Prentice Hall", re.compile(r"\b(prentice[- ]?hall)\b", re.I)),
    ("MIT Press", re.compile(r"\b(mit\s+press)\b", re.I)),
    ("Cambridge University Press", re.compile(r"\b(cambridge\s+university\s+press|cambridge)\b", re.I)),
    ("Oxford University Press", re.compile(r"\b(oxford\s+university\s+press|oxford)\b", re.I)),
    ("Springer", re.compile(r"\b(springer\s+science|springer\s+nature|springer)\b", re.I)),
    ("CRC Press", re.compile(r"\b(crc\s+press|taylor\s+&\s+francis)\b", re.I)),
    ("Sybex", re.compile(r"\b(sybex)\b", re.I)),
    ("Raspberry Pi Press", re.compile(r"\b(raspberry\s+pi\s+press)\b", re.I)),
    ("Black & Decker", re.compile(r"\b(black\s*(&|and)\s*decker)\b", re.I)),
]

EDITION_WORD_MAP = {
    "first": (1, "1st Edition"),
    "second": (2, "2nd Edition"),
    "third": (3, "3rd Edition"),
    "fourth": (4, "4th Edition"),
    "fifth": (5, "5th Edition"),
    "sixth": (6, "6th Edition"),
    "seventh": (7, "7th Edition"),
    "eighth": (8, "8th Edition"),
    "ninth": (9, "9th Edition"),
    "tenth": (10, "10th Edition"),
}


@dataclass
class FrontMatterResult:
    """Encapsulates extracted bibliographic metadata from document front matter."""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    published: Optional[str] = None
    edition: Optional[str] = None
    revision: Optional[int] = None
    isbn: Optional[str] = None
    summary: Optional[str] = None
    language: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "front_matter"

    @property
    def has_useful_title(self) -> bool:
        return bool(self.title and not is_uninformative_title(self.title))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_uninformative_title(title: str) -> bool:
    """
    Evaluates whether a title string is an uninformative placeholder,
    raw date, hash, or generic document name requiring front matter recovery.
    """
    cleaned = title.strip()
    if not cleaned or len(cleaned) <= 2:
        return True

    # Strip extension if present
    p = Path(cleaned)
    stem = p.stem.strip() if p.suffix else cleaned

    for pat in UNINFORMATIVE_PATTERNS:
        if pat.match(stem) or pat.match(cleaned):
            return True

    # If it consists purely of digits, hyphens, underscores, spaces
    if re.match(r"^[\d\s\-_.]+$", stem):
        return True

    return False


def clean_isbn(raw_isbn: str) -> str:
    """Removes hyphens, whitespace, and formatting from an ISBN string."""
    return re.sub(r"[\s\-]", "", raw_isbn).strip().upper()


def validate_isbn(isbn_str: str) -> bool:
    """Validates length and checksum format for ISBN-10 or ISBN-13."""
    cleaned = clean_isbn(isbn_str)
    if len(cleaned) == 13 and cleaned.isdigit() and (cleaned.startswith("978") or cleaned.startswith("979")):
        return True
    if len(cleaned) == 10 and (cleaned[:-1].isdigit()) and (cleaned[-1].isdigit() or cleaned[-1] == "X"):
        return True
    return False


def parse_pdf_date(date_str: Optional[str]) -> Optional[str]:
    """Parses PDF creation/modification date string (e.g. 'D:20240319120000Z') into year 'YYYY'."""
    if not date_str:
        return None
    match = re.search(r"\b(19\d\d|20\d\d)", date_str)
    if match:
        return match.group(1)
    return None


def extract_pdf_front_matter(
    file_path: Union[str, Path], max_pages: int = 5
) -> FrontMatterResult:
    """
    Extracts embedded XMP metadata and first-page front matter text from a PDF file.
    """
    path = Path(file_path)
    if not path.is_file():
        return FrontMatterResult(source="pdf_not_found")

    try:
        import pypdf
    except ImportError:
        return FrontMatterResult(source="pypdf_missing")

    result = FrontMatterResult(source="pdf_front_matter")

    try:
        reader = pypdf.PdfReader(str(path), strict=False)
    except Exception as e:
        result.source = f"pdf_read_error: {e}"
        return result

    # 1. Inspect Document Info / Metadata Dictionary
    metadata = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            if v:
                key = str(k).lstrip("/")
                metadata[key] = str(v).strip()

    result.raw_metadata = metadata

    # Title from Metadata
    meta_title = metadata.get("Title")
    if meta_title and not is_uninformative_title(meta_title):
        # Ignore generic PDF generator titles like "Microsoft Word - Document1" or "untitled"
        if not re.search(r"(microsoft\s+word|untitled|latex\s+to\s+pdf|adobe\s+acrobat|indd)", meta_title, re.I):
            result.title = meta_title.strip()

    # Author from Metadata
    meta_author = metadata.get("Author")
    if meta_author and not is_uninformative_title(meta_author):
        # Filter out generic tool names
        if not re.search(r"(adobe|acrobat|word|latex|canva|microsoft|print|publisher)", meta_author, re.I):
            result.author = meta_author.strip()

    # Subject / Description from Metadata
    meta_subject = metadata.get("Subject")
    if meta_subject and len(meta_subject) > 5 and not is_uninformative_title(meta_subject):
        result.summary = meta_subject.strip()

    # Creation Date
    date_cand = parse_pdf_date(metadata.get("CreationDate") or metadata.get("ModDate"))
    if date_cand:
        result.published = date_cand

    # Producer / Creator check for publisher
    producer = f"{metadata.get('Producer', '')} {metadata.get('Creator', '')}"
    for pub_name, pub_pat in PUBLISHER_PATTERNS:
        if pub_pat.search(producer):
            result.publisher = pub_name
            break

    # If filename is informative and metadata provided title/author/publisher, return early
    if not is_uninformative_title(path.name) and result.title and result.author and result.publisher:
        return result

    # 2. Extract Text from First N Front Matter Pages
    pages_text: List[str] = []
    num_pages_to_scan = min(len(reader.pages), 3 if not is_uninformative_title(path.name) else max_pages)

    for i in range(num_pages_to_scan):
        try:
            p_text = reader.pages[i].extract_text()
            if p_text:
                pages_text.append(p_text)
        except Exception:
            continue

    combined_text = "\n---PAGE---\n".join(pages_text)

    if not combined_text.strip():
        return result

    # 3. Detect ISBN in Front Matter (Copyright / Colophon Pages)
    isbn_13_matches = ISBN_13_PATTERN.findall(combined_text)
    for match in isbn_13_matches:
        if validate_isbn(match):
            result.isbn = clean_isbn(match)
            break

    if not result.isbn:
        isbn_10_matches = ISBN_10_PATTERN.findall(combined_text)
        for match in isbn_10_matches:
            if validate_isbn(match):
                result.isbn = clean_isbn(match)
                break

    # 4. Detect Publisher in Text
    if not result.publisher:
        for pub_name, pub_pat in PUBLISHER_PATTERNS:
            if pub_pat.search(combined_text):
                result.publisher = pub_name
                break

    # 5. Detect Publication Year in Text
    if not result.published:
        pub_year_match = re.search(
            r"(?:First\s+published|Published\s+in|Published|Copyright\s+©|Copyright|©)\s*(?:in)?\s*(?:[A-Za-z]+\s+)?(19\d\d|20\d\d)",
            combined_text,
            re.I,
        )
        if pub_year_match:
            result.published = pub_year_match.group(1)

    # 6. Detect Edition in Text
    ed_match = re.search(
        r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|\d+(?:st|nd|rd|th))\s+Edition\b",
        combined_text,
        re.I,
    )
    if ed_match:
        ed_val = ed_match.group(1).lower()
        if ed_val in EDITION_WORD_MAP:
            result.revision, result.edition = EDITION_WORD_MAP[ed_val]
        elif ed_match.group(1)[0].isdigit():
            num_part = int(re.match(r"\d+", ed_match.group(1)).group(0))
            result.revision = num_part
            result.edition = f"{num_part}th Edition" if num_part not in {1, 2, 3} else (
                "1st Edition" if num_part == 1 else ("2nd Edition" if num_part == 2 else "3rd Edition")
            )

    # 7. Extract Title & Author from First Page Text if Missing or Uninformative
    if not result.title or is_uninformative_title(result.title):
        first_page = pages_text[0] if pages_text else ""
        lines = [line.strip() for line in first_page.splitlines() if line.strip()]

        # Filter out publisher watermarks, date lines, or single characters
        meaningful_lines = [
            l for l in lines
            if len(l) > 2
            and not re.match(r"^(packt>|packt|o'reilly|wiley|manning|chapter|\d+|page\s+\d+|www\..+)$", l, re.I)
        ]

        if meaningful_lines:
            candidate_title = meaningful_lines[0]
            # Check if subtitle is on the next line
            if len(meaningful_lines) > 1 and len(meaningful_lines[1]) > 5 and not meaningful_lines[1].lower().startswith("by "):
                result.subtitle = meaningful_lines[1]
            result.title = candidate_title

    # 8. Extract Author byline if missing
    if not result.author:
        byline_match = re.search(r"\b(?:Written\s+by|By)\s+([A-Z][A-Za-z\s.,&]+?)(?:\r|\n|Published|Packt|O'Reilly|Wiley|$)", combined_text)
        if byline_match:
            cand_author = byline_match.group(1).strip()
            if 3 < len(cand_author) < 60 and not re.search(r"(packt|press|wiley|media)", cand_author, re.I):
                result.author = cand_author

    return result


def extract_epub_front_matter(file_path: Union[str, Path]) -> FrontMatterResult:
    """
    Extracts Dublin Core metadata and structural info from an EPUB container.
    """
    path = Path(file_path)
    if not path.is_file():
        return FrontMatterResult(source="epub_not_found")

    result = FrontMatterResult(source="epub_opf")

    try:
        with zipfile.ZipFile(path, "r") as z:
            opf_path = None
            # Find container.xml
            if "META-INF/container.xml" in z.namelist():
                container_data = z.read("META-INF/container.xml")
                root = ET.fromstring(container_data)
                for rootfile in root.findall(".//{*}rootfile"):
                    opf_path = rootfile.get("full-path")
                    if opf_path:
                        break

            # Fallback search for any .opf
            if not opf_path:
                for name in z.namelist():
                    if name.endswith(".opf"):
                        opf_path = name
                        break

            if not opf_path or opf_path not in z.namelist():
                return result

            opf_data = z.read(opf_path)
            opf_root = ET.fromstring(opf_data)

            # Namespaces
            for elem in opf_root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                val = elem.text.strip() if elem.text else ""
                if not val:
                    continue

                if tag == "title" and not result.title:
                    result.title = val
                elif tag == "creator" and not result.author:
                    result.author = val
                elif tag == "publisher" and not result.publisher:
                    result.publisher = val
                elif tag == "date" and not result.published:
                    year_match = re.search(r"\b(19\d\d|20\d\d)\b", val)
                    result.published = year_match.group(1) if year_match else val
                elif tag == "language" and not result.language:
                    result.language = val
                elif tag == "description" and not result.summary:
                    # Clean HTML tags from EPUB description
                    cleaned_desc = re.sub(r"<[^>]+>", " ", val)
                    cleaned_desc = html.unescape(cleaned_desc)
                    cleaned_desc = re.sub(r"\s+", " ", cleaned_desc).strip()
                    result.summary = cleaned_desc
                elif tag == "identifier" and not result.isbn:
                    match13 = ISBN_13_PATTERN.search(val)
                    if match13 and validate_isbn(match13.group(1)):
                        result.isbn = clean_isbn(match13.group(1))
                    else:
                        match10 = ISBN_10_PATTERN.search(val)
                        if match10 and validate_isbn(match10.group(1)):
                            result.isbn = clean_isbn(match10.group(1))

    except Exception as e:
        result.source = f"epub_read_error: {e}"

    return result


def extract_front_matter(
    file_path: Union[str, Path], max_pages: int = 5
) -> FrontMatterResult:
    """
    Unified entrypoint to extract front matter from PDF, EPUB, or document files.
    """
    path = Path(file_path)
    if not path.is_file():
        return FrontMatterResult(source="file_not_found")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_front_matter(path, max_pages=max_pages)
    elif ext == ".epub":
        return extract_epub_front_matter(path)
    else:
        return FrontMatterResult(source="unsupported_format")


def query_online_by_isbn(
    isbn: str, timeout: float = 3.5
) -> Optional[Dict[str, Any]]:
    """
    Queries Open Library or Google Books API directly by ISBN for 100% precision lookup.
    """
    cleaned_isbn = clean_isbn(isbn)
    if not validate_isbn(cleaned_isbn):
        return None

    # 1. Try Open Library ISBN API
    ol_url = f"https://openlibrary.org/search.json?isbn={cleaned_isbn}&fields=title,subtitle,author_name,first_publish_year,publisher,subject,first_sentence,edition_count,language&limit=1"
    req = urllib.request.Request(ol_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            docs = data.get("docs", [])
            if docs:
                doc = docs[0]
                title = doc.get("title")
                subtitle = doc.get("subtitle")
                authors = doc.get("author_name", [])
                author_str = ", ".join(authors[:3]) if authors else None
                pub_year = str(doc.get("first_publish_year")) if doc.get("first_publish_year") else None
                publishers = doc.get("publisher", [])
                publisher_str = publishers[0] if publishers else None
                languages = doc.get("language", [])
                lang_str = languages[0] if languages else "en"
                first_sent = doc.get("first_sentence")
                summary = first_sent.get("value") if isinstance(first_sent, dict) else first_sent

                return {
                    "title": title,
                    "subtitle": subtitle,
                    "author": author_str,
                    "publisher": publisher_str,
                    "published": pub_year,
                    "language": lang_str,
                    "summary": summary,
                    "source": "OpenLibrary:ISBN",
                }
    except Exception:
        pass

    # 2. Try Google Books ISBN API fallback
    gb_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{cleaned_isbn}&maxResults=1"
    req = urllib.request.Request(gb_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items", [])
            if items:
                v_info = items[0].get("volumeInfo", {})
                title = v_info.get("title")
                subtitle = v_info.get("subtitle")
                authors = v_info.get("authors", [])
                author_str = ", ".join(authors[:3]) if authors else None
                publisher_str = v_info.get("publisher")
                pub_date = v_info.get("publishedDate")
                pub_year = pub_date[:4] if pub_date and len(pub_date) >= 4 else pub_date
                lang_str = v_info.get("language", "en")
                desc = v_info.get("description")

                return {
                    "title": title,
                    "subtitle": subtitle,
                    "author": author_str,
                    "publisher": publisher_str,
                    "published": pub_year,
                    "language": lang_str,
                    "summary": desc,
                    "source": "GoogleBooks:ISBN",
                }
    except Exception:
        pass

    return None
