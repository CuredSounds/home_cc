"""
Unit tests for Vector Database Ingestion Pipeline.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from embedding_provider import BaseEmbeddingProvider
from ingest_vector_db import (
    VectorDBIngester,
    fetch_sqlite_books,
    format_book_metadata_text,
)
from pg_vector_db import PGConfig


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__(model_name="mock-provider", dimension=dimension)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class TestIngestion(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.sqlite_path = Path(self.temp_dir) / "test_books.db"

        # Create a sample SQLite DB with books table
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE books (
                    id INTEGER PRIMARY KEY,
                    clean_title TEXT,
                    subtitle TEXT,
                    raw_title TEXT,
                    author TEXT,
                    publisher TEXT,
                    published TEXT,
                    edition TEXT,
                    language TEXT,
                    domain TEXT,
                    subject TEXT,
                    subfield TEXT,
                    genre TEXT,
                    topics TEXT,
                    key_concepts TEXT,
                    target_audience TEXT,
                    media_type TEXT,
                    summary TEXT,
                    file_type TEXT,
                    file_extension TEXT,
                    source_path TEXT,
                    in_directory INTEGER
                );
                """
            )
            conn.execute(
                """
                INSERT INTO books (
                    id, clean_title, subtitle, author, domain, subject, summary, source_path
                ) VALUES (
                    1, 'Deep Learning Essentials', 'A Hands-On Guide', 'Ian Goodfellow',
                    'Computer Science', 'Artificial Intelligence', 'Comprehensive deep learning guide.',
                    NULL
                );
                """
            )
            conn.commit()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_book_metadata_text(self) -> None:
        book = {
            "clean_title": "Reinforcement Learning",
            "author": "Richard Sutton",
            "domain": "Artificial Intelligence",
            "summary": "An introduction to RL algorithms.",
        }
        text = format_book_metadata_text(book)
        self.assertIn("Title: Reinforcement Learning", text)
        self.assertIn("Author: Richard Sutton", text)
        self.assertIn("Domain: Artificial Intelligence", text)
        self.assertIn("Summary: An introduction to RL algorithms.", text)

    def test_fetch_sqlite_books(self) -> None:
        books = fetch_sqlite_books(self.sqlite_path)
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["clean_title"], "Deep Learning Essentials")
        self.assertEqual(books[0]["author"], "Ian Goodfellow")

    @patch("ingest_vector_db.get_db_connection")
    def test_ingest_catalog_records(self, mock_get_conn: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (101, 1)  # (pg_id, sqlite_id)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        books = fetch_sqlite_books(self.sqlite_path)
        provider = MockEmbeddingProvider(dimension=4)
        ingester = VectorDBIngester(
            pg_config=PGConfig(),
            embedding_provider=provider,
            batch_size=10,
        )

        sqlite_to_pg = ingester.ingest_catalog_records(books)
        self.assertEqual(sqlite_to_pg, {1: 101})
        self.assertEqual(mock_cursor.execute.call_count, 1)

    @patch("ingest_vector_db.get_db_connection")
    def test_ingest_document_chunks(self, mock_get_conn: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Create a sample text file
        doc_path = Path(self.temp_dir) / "sample_book.txt"
        doc_path.write_text("Chapter 1: The beginning.\n\nChapter 2: The continuation of learning.")

        provider = MockEmbeddingProvider(dimension=4)
        ingester = VectorDBIngester(
            pg_config=PGConfig(),
            embedding_provider=provider,
            batch_size=10,
        )

        count = ingester.ingest_document_chunks(pg_book_id=101, file_path=doc_path)
        self.assertTrue(count >= 1)
        # Verify deletion and executemany insert were called
        mock_cursor.execute.assert_called_with("DELETE FROM book_chunks WHERE book_id = %s;", (101,))
        self.assertTrue(mock_cursor.executemany.called)


if __name__ == "__main__":
    unittest.main()
