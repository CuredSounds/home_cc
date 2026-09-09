"""
Batch Vectorizer & Database Ingestion Pipeline.

Migrates catalog metadata and full-text document chunks (PDF/EPUB) into PostgreSQL + pgvector
with semantic dense embeddings and sparse tsvector indexes.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg
from psycopg import Connection, sql

from document_chunker import DocumentChunk, DocumentChunker
from embedding_provider import BaseEmbeddingProvider, get_embedding_provider
from pg_vector_db import DEFAULT_VECTOR_DIM, PGConfig, get_db_connection, init_vector_database, verify_schema

DEFAULT_SQLITE_PATH = Path("books.db")


def format_book_metadata_text(book: Dict[str, Any]) -> str:
    """Formats structured catalog fields into a rich text passage for macro book embedding."""
    parts: List[str] = []
    title = book.get("clean_title") or book.get("raw_title") or "Unknown"
    parts.append(f"Title: {title}")

    if book.get("subtitle"):
        parts.append(f"Subtitle: {book['subtitle']}")
    if book.get("author"):
        parts.append(f"Author: {book['author']}")
    if book.get("publisher"):
        parts.append(f"Publisher: {book['publisher']}")
    if book.get("published"):
        parts.append(f"Published: {book['published']}")
    if book.get("edition"):
        parts.append(f"Edition: {book['edition']}")
    if book.get("domain"):
        parts.append(f"Domain: {book['domain']}")
    if book.get("subject"):
        parts.append(f"Subject: {book['subject']}")
    if book.get("subfield"):
        parts.append(f"Subfield: {book['subfield']}")
    if book.get("genre"):
        parts.append(f"Genre: {book['genre']}")
    if book.get("topics"):
        parts.append(f"Topics: {book['topics']}")
    if book.get("key_concepts"):
        parts.append(f"Key Concepts: {book['key_concepts']}")
    if book.get("target_audience"):
        parts.append(f"Target Audience: {book['target_audience']}")
    if book.get("summary"):
        parts.append(f"Summary: {book['summary']}")

    return " | ".join(parts)


def fetch_sqlite_books(sqlite_path: Union[str, Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads records from the SQLite books table as dictionaries."""
    path = Path(sqlite_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    books: List[Dict[str, Any]] = []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT * FROM books ORDER BY id"
        if limit:
            query += f" LIMIT {int(limit)}"
        cursor.execute(query)
        for row in cursor.fetchall():
            books.append(dict(row))

    return books


class VectorDBIngester:
    """
    Coordinates batch embedding generation and insertion into PostgreSQL pgvector.
    """

    def __init__(
        self,
        pg_config: Optional[PGConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        batch_size: int = 32,
    ) -> None:
        self.pg_config = pg_config or PGConfig.from_env()
        self.provider = embedding_provider or get_embedding_provider()
        self.batch_size = batch_size
        self.chunker = DocumentChunker()

    def ingest_catalog_records(
        self,
        books: List[Dict[str, Any]],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[int, int]:
        """
        Embeds book metadata passages and upserts catalog records into PostgreSQL `books` table.
        Returns mapping of sqlite_id -> postgres_id.
        """
        if not books:
            return {}

        total = len(books)
        sqlite_to_pg_id: Dict[int, int] = {}

        # Prepare metadata passages
        metadata_texts = [format_book_metadata_text(b) for b in books]

        # Generate dense vector embeddings in batches
        all_embeddings: List[List[float]] = []
        for i in range(0, total, self.batch_size):
            batch_texts = metadata_texts[i : i + self.batch_size]
            batch_embs = self.provider.embed_batch(batch_texts, batch_size=self.batch_size)
            all_embeddings.extend(batch_embs)
            if progress_cb:
                progress_cb(min(i + len(batch_texts), total), total, "Embedding catalog records")

        # Upsert into PostgreSQL
        with get_db_connection(self.pg_config) as conn:
            with conn.cursor() as cur:
                for idx, (book, emb) in enumerate(zip(books, all_embeddings)):
                    sql_query = """
                    INSERT INTO books (
                        sqlite_id, clean_title, subtitle, raw_title, author, publisher, published,
                        edition, language, domain, subject, subfield, genre, topics, key_concepts,
                        target_audience, media_type, summary, file_type, file_extension,
                        source_path, in_directory, metadata_embedding, updated_at
                    ) VALUES (
                        %(sqlite_id)s, %(clean_title)s, %(subtitle)s, %(raw_title)s, %(author)s,
                        %(publisher)s, %(published)s, %(edition)s, %(language)s, %(domain)s,
                        %(subject)s, %(subfield)s, %(genre)s, %(topics)s, %(key_concepts)s,
                        %(target_audience)s, %(media_type)s, %(summary)s, %(file_type)s,
                        %(file_extension)s, %(source_path)s, %(in_directory)s, %(metadata_embedding)s,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (sqlite_id) DO UPDATE SET
                        clean_title = EXCLUDED.clean_title,
                        subtitle = EXCLUDED.subtitle,
                        raw_title = EXCLUDED.raw_title,
                        author = EXCLUDED.author,
                        publisher = EXCLUDED.publisher,
                        published = EXCLUDED.published,
                        edition = EXCLUDED.edition,
                        language = EXCLUDED.language,
                        domain = EXCLUDED.domain,
                        subject = EXCLUDED.subject,
                        subfield = EXCLUDED.subfield,
                        genre = EXCLUDED.genre,
                        topics = EXCLUDED.topics,
                        key_concepts = EXCLUDED.key_concepts,
                        target_audience = EXCLUDED.target_audience,
                        media_type = EXCLUDED.media_type,
                        summary = EXCLUDED.summary,
                        file_type = EXCLUDED.file_type,
                        file_extension = EXCLUDED.file_extension,
                        source_path = EXCLUDED.source_path,
                        in_directory = EXCLUDED.in_directory,
                        metadata_embedding = EXCLUDED.metadata_embedding,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id, sqlite_id;
                    """

                    params = {
                        "sqlite_id": book.get("id"),
                        "clean_title": book.get("clean_title") or book.get("raw_title") or "Unknown",
                        "subtitle": book.get("subtitle"),
                        "raw_title": book.get("raw_title") or book.get("clean_title") or "Unknown",
                        "author": book.get("author"),
                        "publisher": book.get("publisher"),
                        "published": str(book["published"]) if book.get("published") is not None else None,
                        "edition": book.get("edition"),
                        "language": book.get("language") or "en",
                        "domain": book.get("domain"),
                        "subject": book.get("subject"),
                        "subfield": book.get("subfield"),
                        "genre": book.get("genre"),
                        "topics": book.get("topics"),
                        "key_concepts": book.get("key_concepts"),
                        "target_audience": book.get("target_audience"),
                        "media_type": book.get("media_type"),
                        "summary": book.get("summary"),
                        "file_type": book.get("file_type"),
                        "file_extension": book.get("file_extension"),
                        "source_path": book.get("source_path"),
                        "in_directory": book.get("in_directory", 0),
                        "metadata_embedding": emb,
                    }

                    cur.execute(sql_query, params)
                    res = cur.fetchone()
                    if res:
                        pg_id, sq_id = res[0], res[1]
                        sqlite_to_pg_id[sq_id] = pg_id

        return sqlite_to_pg_id

    def ingest_document_chunks(
        self,
        pg_book_id: int,
        file_path: Union[str, Path],
        max_pages: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ) -> int:
        """
        Extracts, embeds, and stores text chunks for a single book file.
        Returns the number of ingested chunks.
        """
        path = Path(file_path)
        if not path.is_file():
            return 0

        chunks = self.chunker.chunk_document(path, max_pages=max_pages)
        if not chunks:
            return 0

        if max_chunks:
            chunks = chunks[:max_chunks]

        chunk_texts = [c.content for c in chunks]
        chunk_embeddings = self.provider.embed_batch(chunk_texts, batch_size=self.batch_size)

        with get_db_connection(self.pg_config) as conn:
            with conn.cursor() as cur:
                # Clear existing chunks for this book if re-ingesting
                cur.execute("DELETE FROM book_chunks WHERE book_id = %s;", (pg_book_id,))

                # Insert chunk records
                insert_sql = """
                INSERT INTO book_chunks (
                    book_id, chunk_index, page_number, section_title, content, token_count, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                """

                rows_to_insert = [
                    (
                        pg_book_id,
                        c.chunk_index,
                        c.page_number,
                        c.section_title,
                        c.content,
                        c.token_count,
                        emb,
                    )
                    for c, emb in zip(chunks, chunk_embeddings)
                ]

                cur.executemany(insert_sql, rows_to_insert)

        return len(chunks)

    def ingest_all(
        self,
        sqlite_path: Union[str, Path] = DEFAULT_SQLITE_PATH,
        max_books: Optional[int] = None,
        max_chunks_per_book: Optional[int] = None,
        catalog_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end ingestion pipeline:
        1. Reads books from SQLite.
        2. Ingests catalog records + metadata embeddings.
        3. If not catalog_only, parses disk documents and indexes granular chunks.
        """
        start_time = time.time()
        books = fetch_sqlite_books(sqlite_path, limit=max_books)
        total_books = len(books)

        print(f"Loaded {total_books} book records from '{sqlite_path}'.")
        print(f"Generating embeddings with provider '{self.provider.model_name}' (dim: {self.provider.get_dimension()})...")

        # 1. Ingest catalog
        sqlite_to_pg_map = self.ingest_catalog_records(
            books,
            progress_cb=lambda cur, tot, msg: print(f"[{cur}/{tot}] {msg}...", end="\r", flush=True),
        )
        print(f"\nSuccessfully ingested {len(sqlite_to_pg_map)} catalog records into PostgreSQL 'books' table.")

        total_chunks = 0
        documents_processed = 0

        # 2. Ingest document chunks
        if not catalog_only:
            print("\nProcessing physical documents for page/passage chunking...")
            for idx, book in enumerate(books, start=1):
                src_path = book.get("source_path")
                sq_id = book.get("id")
                pg_id = sqlite_to_pg_map.get(sq_id)

                if not pg_id or not src_path or not Path(src_path).is_file():
                    continue

                try:
                    num_chunks = self.ingest_document_chunks(
                        pg_book_id=pg_id,
                        file_path=src_path,
                        max_chunks=max_chunks_per_book,
                    )
                    if num_chunks > 0:
                        total_chunks += num_chunks
                        documents_processed += 1
                        print(f"[{idx}/{total_books}] Chunked '{book.get('clean_title', '')[:40]}' -> {num_chunks} chunk(s)")
                except Exception as e:
                    print(f"[{idx}/{total_books}] Warning: Failed to chunk '{src_path}': {e}", file=sys.stderr)

        elapsed = round(time.time() - start_time, 2)

        return {
            "catalog_records_ingested": len(sqlite_to_pg_map),
            "documents_chunked": documents_processed,
            "total_chunks_ingested": total_chunks,
            "elapsed_seconds": elapsed,
            "vector_dimension": self.provider.get_dimension(),
        }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest book catalog and document chunks into PostgreSQL pgvector."
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=DEFAULT_SQLITE_PATH,
        help=f"Path to source SQLite database (default: {DEFAULT_SQLITE_PATH})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Embedding provider (fastembed, ollama, openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vector embedding generation (default: 32)",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=None,
        help="Limit number of books to ingest (for testing/benchmarking)",
    )
    parser.add_argument(
        "--max-chunks-per-book",
        type=int,
        default=None,
        help="Limit number of chunks extracted per book",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Ingest catalog records and metadata embeddings only (skip file chunking)",
    )
    parser.add_argument(
        "--force-init",
        action="store_true",
        help="Reinitialize PostgreSQL schema and drop existing tables before ingestion",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    try:
        # Check provider & schema
        provider = get_embedding_provider(
            provider_type=parsed.provider,
            model_name=parsed.model,
        )
        vector_dim = provider.get_dimension()
        cfg = PGConfig.from_env(vector_dim=vector_dim)

        if parsed.force_init:
            print(f"Force initializing database schema with vector dimension {vector_dim}...")
            init_vector_database(config=cfg, force=True, vector_dim=vector_dim)

        ingester = VectorDBIngester(
            pg_config=cfg,
            embedding_provider=provider,
            batch_size=parsed.batch_size,
        )

        stats = ingester.ingest_all(
            sqlite_path=parsed.sqlite_db,
            max_books=parsed.max_books,
            max_chunks_per_book=parsed.max_chunks_per_book,
            catalog_only=parsed.catalog_only,
        )

        print("\n================ Ingestion Summary ================")
        print(f"Catalog Records Ingested: {stats['catalog_records_ingested']}")
        print(f"Documents Chunked:        {stats['documents_chunked']}")
        print(f"Total Passage Chunks:     {stats['total_chunks_ingested']}")
        print(f"Vector Dimension:         {stats['vector_dimension']}")
        print(f"Elapsed Time:             {stats['elapsed_seconds']}s")
        print("===================================================")
        return 0

    except psycopg.OperationalError as e:
        print(f"\n[!] PostgreSQL Connection Error: {e}", file=sys.stderr)
        print("    Ensure PostgreSQL is running (e.g. `docker compose up -d`) and initialized.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[!] Ingestion Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
