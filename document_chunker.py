"""
Document Extraction and Sliding Window Semantic Text Chunker.

Extracts text from PDF (via pypdf), EPUB (via zipfile), Markdown, and Plain Text files,
preserving page numbers and section headers, and segments text into token-aware overlapping chunks.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    import pypdf
except ImportError:
    pypdf = None


@dataclass
class PageText:
    page_number: int
    text: str
    section_title: Optional[str] = None


@dataclass
class DocumentChunk:
    chunk_index: int
    content: str
    page_number: Optional[int]
    section_title: Optional[str]
    token_count: int
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "content": self.content,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }


def estimate_token_count(text: str) -> int:
    """
    Approximates token count based on whitespace words and character heuristics.
    Standard English text: ~0.75 words per token (1 word ≈ 1.33 tokens) or 4 characters per token.
    """
    if not text:
        return 0
    words = text.split()
    # Hybrid word/character estimation: max of word-count*1.3 and char-count/4
    by_words = int(len(words) * 1.33) + 1
    by_chars = int(len(text) / 4.0) + 1
    return max(1, int((by_words + by_chars) / 2.0))


def clean_extracted_text(text: str) -> str:
    """Normalizes whitespace and removes null bytes or unprintable characters."""
    if not text:
        return ""
    # Replace null bytes
    text = text.replace("\x00", " ")
    # Condense excessive newlines/spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html_tags(html_content: str) -> str:
    """Strips HTML markup and unescapes entities."""
    clean = re.sub(r"<script.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = html.unescape(clean)
    return clean_extracted_text(clean)


class DocumentExtractor:
    """Extracts page/chapter-indexed text from documents."""

    @staticmethod
    def extract_pdf(file_path: Union[str, Path], max_pages: Optional[int] = None) -> List[PageText]:
        if pypdf is None:
            raise ImportError("pypdf is required for PDF extraction. Install with `pip install pypdf`.")

        path = Path(file_path)
        if not path.is_file():
            return []

        pages: List[PageText] = []
        try:
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
            limit = min(total_pages, max_pages) if max_pages else total_pages

            for page_idx in range(limit):
                try:
                    page = reader.pages[page_idx]
                    raw_text = page.extract_text() or ""
                    cleaned = clean_extracted_text(raw_text)
                    if cleaned:
                        pages.append(
                            PageText(
                                page_number=page_idx + 1,
                                text=cleaned,
                                section_title=None,
                            )
                        )
                except Exception:
                    # Skip corrupt page gracefully
                    continue
        except Exception:
            return []

        return pages

    @staticmethod
    def extract_epub(file_path: Union[str, Path]) -> List[PageText]:
        path = Path(file_path)
        if not path.is_file():
            return []

        pages: List[PageText] = []
        try:
            with zipfile.ZipFile(path, "r") as z:
                # Find all HTML/XHTML chapter files
                names = [n for n in z.namelist() if n.lower().endswith((".html", ".xhtml", ".htm"))]
                names.sort()

                for idx, name in enumerate(names, start=1):
                    try:
                        raw_html = z.read(name).decode("utf-8", errors="replace")
                        # Try to extract chapter heading if available
                        h_match = re.search(r"<h[1-3][^>]*>(.*?)</h[1-3]>", raw_html, re.I | re.S)
                        heading = strip_html_tags(h_match.group(1)) if h_match else None

                        plain = strip_html_tags(raw_html)
                        if plain:
                            pages.append(
                                PageText(
                                    page_number=idx,
                                    text=plain,
                                    section_title=heading,
                                )
                            )
                    except Exception:
                        continue
        except Exception:
            return []

        return pages

    @staticmethod
    def extract_text_file(file_path: Union[str, Path]) -> List[PageText]:
        path = Path(file_path)
        if not path.is_file():
            return []

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            cleaned = clean_extracted_text(content)
            if not cleaned:
                return []

            # Check if file has markdown headings
            sections = re.split(r"(^#+\s+.+$)", cleaned, flags=re.M)
            if len(sections) > 1:
                pages: List[PageText] = []
                current_heading: Optional[str] = None
                page_idx = 1
                for part in sections:
                    part = part.strip()
                    if not part:
                        continue
                    if part.startswith("#"):
                        current_heading = part.lstrip("#").strip()
                    else:
                        pages.append(
                            PageText(
                                page_number=page_idx,
                                text=part,
                                section_title=current_heading,
                            )
                        )
                        page_idx += 1
                return pages if pages else [PageText(page_number=1, text=cleaned)]

            return [PageText(page_number=1, text=cleaned)]
        except Exception:
            return []

    @classmethod
    def extract_document(
        cls, file_path: Union[str, Path], max_pages: Optional[int] = None
    ) -> List[PageText]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return cls.extract_pdf(path, max_pages=max_pages)
        elif suffix in (".epub", ".ibooks"):
            return cls.extract_epub(path)
        elif suffix in (".txt", ".md", ".markdown", ".rst", ".json", ".csv", ".py"):
            return cls.extract_text_file(path)
        return []


class DocumentChunker:
    """
    Segments page-indexed text into overlapping chunks optimized for vector embeddings and RAG retrieval.
    """

    def __init__(
        self,
        target_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        min_chunk_tokens: int = 30,
    ) -> None:
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = min(overlap_tokens, target_chunk_tokens // 2)
        self.min_chunk_tokens = min_chunk_tokens

    def chunk_pages(self, pages: Sequence[PageText]) -> List[DocumentChunk]:
        """
        Chunks a sequence of document pages with sliding window overlap,
        preserving page numbers and section headers.
        """
        if not pages:
            return []

        # Split into paragraph units annotated with original page and section
        units: List[Tuple[str, int, Optional[str]]] = []

        for p in pages:
            # Split page text into paragraphs
            paragraphs = re.split(r"\n\s*\n", p.text)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                units.append((para, p.page_number, p.section_title))

        if not units:
            return []

        chunks: List[DocumentChunk] = []
        current_paras: List[str] = []
        current_tokens = 0
        current_page = units[0][1]
        current_section = units[0][2]
        chunk_idx = 0

        unit_idx = 0
        while unit_idx < len(units):
            para_text, page_num, sec_title = units[unit_idx]
            para_tokens = estimate_token_count(para_text)

            # If a single paragraph exceeds target chunk size, split by sentences
            if para_tokens > self.target_chunk_tokens:
                sentences = re.split(r"(?<=[.!?])\s+", para_text)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_tokens = estimate_token_count(sent)
                    if current_tokens + sent_tokens > self.target_chunk_tokens and current_paras:
                        chunk_text = "\n\n".join(current_paras).strip()
                        if current_tokens >= self.min_chunk_tokens:
                            chunks.append(
                                DocumentChunk(
                                    chunk_index=chunk_idx,
                                    content=chunk_text,
                                    page_number=current_page,
                                    section_title=current_section,
                                    token_count=current_tokens,
                                    char_count=len(chunk_text),
                                )
                            )
                            chunk_idx += 1

                        # Carry overlap
                        overlap_paras, overlap_toks = self._compute_overlap(current_paras)
                        current_paras = overlap_paras
                        current_tokens = overlap_toks

                    current_paras.append(sent)
                    current_tokens += sent_tokens
                    current_page = current_page or page_num
                    current_section = current_section or sec_title

                unit_idx += 1
                continue

            if current_tokens + para_tokens > self.target_chunk_tokens and current_paras:
                chunk_text = "\n\n".join(current_paras).strip()
                if current_tokens >= self.min_chunk_tokens:
                    chunks.append(
                        DocumentChunk(
                            chunk_index=chunk_idx,
                            content=chunk_text,
                            page_number=current_page,
                            section_title=current_section,
                            token_count=current_tokens,
                            char_count=len(chunk_text),
                        )
                    )
                    chunk_idx += 1

                overlap_paras, overlap_toks = self._compute_overlap(current_paras)
                current_paras = overlap_paras
                current_tokens = overlap_toks
                current_page = page_num
                current_section = sec_title

            current_paras.append(para_text)
            current_tokens += para_tokens
            if not current_section:
                current_section = sec_title
            unit_idx += 1

        # Emit remaining text
        if current_paras:
            chunk_text = "\n\n".join(current_paras).strip()
            if current_tokens >= self.min_chunk_tokens or not chunks:
                chunks.append(
                    DocumentChunk(
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        page_number=current_page,
                        section_title=current_section,
                        token_count=current_tokens,
                        char_count=len(chunk_text),
                    )
                )

        return chunks

    def _compute_overlap(self, paras: List[str]) -> Tuple[List[str], int]:
        """Retains trailing paragraphs that fit within overlap_tokens budget."""
        overlap: List[str] = []
        tok_count = 0
        for p in reversed(paras):
            t = estimate_token_count(p)
            if tok_count + t <= self.overlap_tokens or not overlap:
                overlap.insert(0, p)
                tok_count += t
            else:
                break
        return overlap, tok_count

    def chunk_document(
        self, file_path: Union[str, Path], max_pages: Optional[int] = None
    ) -> List[DocumentChunk]:
        """High-level entrypoint to extract and chunk a document file."""
        pages = DocumentExtractor.extract_document(file_path, max_pages=max_pages)
        return self.chunk_pages(pages)
