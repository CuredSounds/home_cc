"""
Unit tests for Document Extraction and Sliding Window Semantic Text Chunker.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import pypdf
from document_chunker import (
    DocumentChunk,
    DocumentChunker,
    DocumentExtractor,
    PageText,
    clean_extracted_text,
    estimate_token_count,
    strip_html_tags,
)


class TestDocumentChunker(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_estimate_token_count_and_clean_text(self) -> None:
        text = "Hello world! This is a test sentence for vector embeddings."
        tokens = estimate_token_count(text)
        self.assertTrue(5 <= tokens <= 20)
        self.assertEqual(estimate_token_count(""), 0)

        dirty = "Hello \x00 world!  \n\n\n\n  Multiple lines."
        cleaned = clean_extracted_text(dirty)
        self.assertNotIn("\x00", cleaned)
        self.assertIn("\n\n", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_strip_html_tags(self) -> None:
        html_input = "<p>This is <strong>important</strong> text.<script>alert(1)</script></p>"
        self.assertEqual(strip_html_tags(html_input), "This is important text.")

    def test_extract_pdf_pages(self) -> None:
        pdf_path = Path(self.temp_dir) / "sample.pdf"
        writer = pypdf.PdfWriter()
        # Page 1
        page1 = writer.add_blank_page(width=300, height=400)
        # Page 2
        page2 = writer.add_blank_page(width=300, height=400)

        with open(pdf_path, "wb") as f:
            writer.write(f)

        # Extraction on blank pages returns []
        pages = DocumentExtractor.extract_pdf(pdf_path)
        self.assertIsInstance(pages, list)

    def test_extract_epub_chapters(self) -> None:
        epub_path = Path(self.temp_dir) / "book.epub"
        with zipfile.ZipFile(epub_path, "w") as z:
            z.writestr("ch1.xhtml", "<html><body><h1>Chapter 1: Intro</h1><p>Welcome to RAG.</p></body></html>")
            z.writestr("ch2.xhtml", "<html><body><h1>Chapter 2: Vector DB</h1><p>Postgres with pgvector.</p></body></html>")

        pages = DocumentExtractor.extract_epub(epub_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].section_title, "Chapter 1: Intro")
        self.assertIn("Welcome to RAG", pages[0].text)
        self.assertEqual(pages[1].section_title, "Chapter 2: Vector DB")
        self.assertIn("Postgres with pgvector", pages[1].text)

    def test_extract_markdown_sections(self) -> None:
        md_path = Path(self.temp_dir) / "notes.md"
        content = """# Section 1: Overview
Here is the overview text explaining vector databases.

# Section 2: Implementation
Here is how we implement cosine similarity search.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        pages = DocumentExtractor.extract_text_file(md_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].section_title, "Section 1: Overview")
        self.assertEqual(pages[1].section_title, "Section 2: Implementation")

    def test_sliding_window_chunking(self) -> None:
        # Create multi-page document with multiple paragraphs
        pages = [
            PageText(
                page_number=1,
                text="Paragraph 1 on page 1.\n\nParagraph 2 on page 1 with more detail.\n\nParagraph 3 on page 1.",
                section_title="Introduction",
            ),
            PageText(
                page_number=2,
                text="Paragraph 4 on page 2 discussing neural networks.\n\nParagraph 5 on page 2 covering transformers.",
                section_title="Deep Learning",
            ),
        ]

        # Use small target token count to force multiple chunks
        chunker = DocumentChunker(target_chunk_tokens=15, overlap_tokens=5, min_chunk_tokens=5)
        chunks = chunker.chunk_pages(pages)

        self.assertTrue(len(chunks) >= 2)
        for c in chunks:
            self.assertIsNotNone(c.page_number)
            self.assertTrue(c.token_count >= 5)
            self.assertTrue(len(c.content) > 0)

        # First chunk should start on page 1
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].section_title, "Introduction")

    def test_oversized_paragraph_sentence_splitting(self) -> None:
        long_para = "Sentence one is clear. Sentence two has more information. Sentence three adds extra context. Sentence four concludes."
        pages = [PageText(page_number=5, text=long_para, section_title="Architecture")]

        chunker = DocumentChunker(target_chunk_tokens=10, overlap_tokens=3, min_chunk_tokens=3)
        chunks = chunker.chunk_pages(pages)

        self.assertTrue(len(chunks) >= 2)
        for c in chunks:
            self.assertEqual(c.page_number, 5)
            self.assertEqual(c.section_title, "Architecture")


if __name__ == "__main__":
    unittest.main()
