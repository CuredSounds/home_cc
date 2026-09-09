"""
Unit tests for Front Matter & Embedded Metadata Extraction (PDF, EPUB, ISBN).
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import pypdf
from front_matter import (
    FrontMatterResult,
    clean_isbn,
    extract_epub_front_matter,
    extract_front_matter,
    extract_pdf_front_matter,
    is_uninformative_title,
    validate_isbn,
)
from enrich_books import MetadataCache, OnlineEnricher


class TestFrontMatter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_uninformative_title(self) -> None:
        uninformative = [
            "20240319",
            "20240319.pdf",
            "2024-03-19",
            "2024_03_19",
            "12251",
            "d41d8cd98f00b204e9800998ecf8427e",
            "c02b3df5-18ab-4f27-9c8e-327cba84e621",
            "doc",
            "untitled",
            "temp",
            "packt-20240319",
            "123",
            "",
        ]
        for name in uninformative:
            self.assertTrue(is_uninformative_title(name), f"Expected '{name}' to be uninformative")

        informative = [
            "The Art of Electronics",
            "Deep Learning",
            "Clean Code",
            "Mastering PyTorch",
            "Raspberry Pi Cookbook",
            "20 Chemistry Books Collection",
        ]
        for name in informative:
            self.assertFalse(is_uninformative_title(name), f"Expected '{name}' to be informative")

    def test_isbn_validation_and_cleaning(self) -> None:
        valid_13 = "978-1-80461-123-4"
        valid_10 = "0-13-110362-8"
        invalid = "12345"

        self.assertEqual(clean_isbn(valid_13), "9781804611234")
        self.assertEqual(clean_isbn(valid_10), "0131103628")

        self.assertTrue(validate_isbn(valid_13))
        self.assertTrue(validate_isbn(valid_10))
        self.assertFalse(validate_isbn(invalid))

    def test_extract_pdf_metadata(self) -> None:
        pdf_path = Path(self.temp_dir) / "test_doc.pdf"

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=300, height=400)
        writer.add_metadata({
            "/Title": "Mastering Linux Administration",
            "/Author": "Paul Cobbaut",
            "/Subject": "Enterprise Linux System Engineering",
            "/Producer": "Packt Publishing Ltd.",
            "/CreationDate": "D:20240319120000Z",
        })

        with open(pdf_path, "wb") as f:
            writer.write(f)

        result = extract_pdf_front_matter(pdf_path)
        self.assertEqual(result.title, "Mastering Linux Administration")
        self.assertEqual(result.author, "Paul Cobbaut")
        self.assertEqual(result.summary, "Enterprise Linux System Engineering")
        self.assertEqual(result.publisher, "Packt Publishing")
        self.assertEqual(result.published, "2024")

    def test_extract_epub_front_matter(self) -> None:
        epub_path = Path(self.temp_dir) / "test_book.epub"

        opf_content = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title>Generative AI with Python and PyTorch</dc:title>
    <dc:creator>Jane Doe</dc:creator>
    <dc:publisher>Packt Publishing</dc:publisher>
    <dc:date>2024-03-15</dc:date>
    <dc:language>en</dc:language>
    <dc:identifier opf:scheme="ISBN">978-1-80461-999-9</dc:identifier>
    <dc:description>&lt;p&gt;A practical guide to LLMs and diffusion models.&lt;/p&gt;</dc:description>
  </metadata>
</package>
"""
        container_content = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
        with zipfile.ZipFile(epub_path, "w") as z:
            z.writestr("META-INF/container.xml", container_content)
            z.writestr("OEBPS/content.opf", opf_content)

        result = extract_epub_front_matter(epub_path)
        self.assertEqual(result.title, "Generative AI with Python and PyTorch")
        self.assertEqual(result.author, "Jane Doe")
        self.assertEqual(result.publisher, "Packt Publishing")
        self.assertEqual(result.published, "2024")
        self.assertEqual(result.isbn, "9781804619999")
        self.assertEqual(result.summary, "A practical guide to LLMs and diffusion models.")

    def test_end_to_end_enricher_with_date_stamped_file(self) -> None:
        # Create date-stamped PDF: 20240319.pdf
        pdf_path = Path(self.temp_dir) / "20240319.pdf"

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=300, height=400)
        writer.add_metadata({
            "/Title": "Hands-On Artificial Intelligence for IoT",
            "/Author": "Amita Kapoor",
            "/Subject": "Edge computing and neural network deployment",
            "/Producer": "Packt Publishing",
            "/CreationDate": "D:20240319000000Z",
        })

        with open(pdf_path, "wb") as f:
            writer.write(f)

        cache = MetadataCache(cache_file=Path(self.temp_dir) / ".cache.json")
        enricher = OnlineEnricher(cache=cache, offline_only=True, scan_front_matter=True)

        record = enricher.enrich(
            clean_title="20240319",
            raw_title="20240319.pdf",
            source_path=str(pdf_path),
            in_dir=1,
        )

        # Confirm the uninformative title 20240319 was automatically recovered to the true book title
        self.assertEqual(record.clean_title, "Hands-On Artificial Intelligence for IoT")
        self.assertEqual(record.author, "Amita Kapoor")
        self.assertEqual(record.publisher, "Packt Publishing")
        self.assertEqual(record.published, "2024")
        self.assertEqual(record.file_type, "PDF")
        self.assertIn("Artificial Intelligence", record.domain)


if __name__ == "__main__":
    unittest.main()
