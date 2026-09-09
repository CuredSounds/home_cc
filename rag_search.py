"""
Hybrid Vector & Keyword Search Engine and RAG Context Retrieval Platform.

Combines PostgreSQL + pgvector dense HNSW cosine similarity search with sparse tsvector
full-text search using Reciprocal Rank Fusion (RRF), hierarchical catalog retrieval,
and RAG prompt synthesis.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row

from embedding_provider import BaseEmbeddingProvider, get_embedding_provider
from pg_vector_db import PGConfig, get_db_connection


@dataclass
class BookSearchResult:
    book_id: int
    clean_title: str
    subtitle: Optional[str]
    author: Optional[str]
    domain: Optional[str]
    subject: Optional[str]
    summary: Optional[str]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "clean_title": self.clean_title,
            "subtitle": self.subtitle,
            "author": self.author,
            "domain": self.domain,
            "subject": self.subject,
            "summary": self.summary,
            "score": round(self.score, 4),
        }


@dataclass
class SearchResult:
    chunk_id: int
    book_id: int
    clean_title: str
    author: Optional[str]
    domain: Optional[str]
    subject: Optional[str]
    page_number: Optional[int]
    section_title: Optional[str]
    content: str
    score: float
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "book_id": self.book_id,
            "clean_title": self.clean_title,
            "author": self.author,
            "domain": self.domain,
            "subject": self.subject,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "content": self.content,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
            "text_score": round(self.text_score, 4) if self.text_score is not None else None,
            "source_path": self.source_path,
        }

    @property
    def citation(self) -> str:
        parts = [f"《{self.clean_title}》"]
        if self.author:
            parts.append(f"by {self.author}")
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.page_number:
            parts.append(f"p. {self.page_number}")
        return " | ".join(parts)


def compute_rrf_score(rank: int, k: int = 60, weight: float = 1.0) -> float:
    """Computes Reciprocal Rank Fusion component: weight / (k + rank)."""
    return float(weight) / (float(k) + float(rank))


class RAGSearchEngine:
    """
    Hybrid search and retrieval-augmented generation engine.
    """

    def __init__(
        self,
        pg_config: Optional[PGConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self.pg_config = pg_config or PGConfig.from_env()
        self.provider = embedding_provider or get_embedding_provider()

    def _build_filter_clauses(
        self, filters: Optional[Dict[str, Any]] = None, table_alias: str = "b"
    ) -> Tuple[List[str], List[Any]]:
        if not filters:
            return [], []

        clauses: List[str] = []
        params: List[Any] = []

        mapping = {
            "domain": f"{table_alias}.domain ILIKE %s",
            "subject": f"{table_alias}.subject ILIKE %s",
            "author": f"{table_alias}.author ILIKE %s",
            "media_type": f"{table_alias}.media_type ILIKE %s",
            "published": f"{table_alias}.published ILIKE %s",
            "file_type": f"{table_alias}.file_type ILIKE %s",
            "in_directory": f"{table_alias}.in_directory = %s",
        }

        for k, v in filters.items():
            if v is not None and k in mapping:
                clauses.append(mapping[k])
                if k == "in_directory":
                    params.append(int(v))
                else:
                    params.append(f"%{v}%")

        return clauses, params

    def search_books_dense(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BookSearchResult]:
        """
        Embeds query and executes dense vector cosine similarity search on the catalog `books` table.
        """
        query_vec = self.provider.embed_text(query)
        filter_clauses, filter_params = self._build_filter_clauses(filters, table_alias="b")

        where_sql = "WHERE b.metadata_embedding IS NOT NULL"
        if filter_clauses:
            where_sql += " AND " + " AND ".join(filter_clauses)

        query_sql = f"""
        SELECT 
            b.id as book_id,
            b.clean_title,
            b.subtitle,
            b.author,
            b.domain,
            b.subject,
            b.summary,
            1.0 - (b.metadata_embedding <=> %s::vector) AS similarity
        FROM books b
        {where_sql}
        ORDER BY b.metadata_embedding <=> %s::vector ASC
        LIMIT %s;
        """

        results: List[BookSearchResult] = []
        with get_db_connection(self.pg_config) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                all_params = [query_vec] + filter_params + [query_vec, top_k]
                cur.execute(query_sql, all_params)
                for row in cur.fetchall():
                    results.append(
                        BookSearchResult(
                            book_id=row["book_id"],
                            clean_title=row["clean_title"],
                            subtitle=row["subtitle"],
                            author=row["author"],
                            domain=row["domain"],
                            subject=row["subject"],
                            summary=row["summary"],
                            score=float(row["similarity"]) if row["similarity"] is not None else 0.0,
                        )
                    )

        return results

    def search_chunks_dense(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Executes dense vector search on passage-level `book_chunks` table using HNSW index.
        """
        query_vec = self.provider.embed_text(query)
        filter_clauses, filter_params = self._build_filter_clauses(filters, table_alias="b")

        where_sql = "WHERE c.embedding IS NOT NULL"
        if filter_clauses:
            where_sql += " AND " + " AND ".join(filter_clauses)

        query_sql = f"""
        SELECT 
            c.id as chunk_id,
            c.book_id,
            b.clean_title,
            b.author,
            b.domain,
            b.subject,
            b.source_path,
            c.page_number,
            c.section_title,
            c.content,
            1.0 - (c.embedding <=> %s::vector) AS similarity
        FROM book_chunks c
        JOIN books b ON b.id = c.book_id
        {where_sql}
        ORDER BY c.embedding <=> %s::vector ASC
        LIMIT %s;
        """

        results: List[SearchResult] = []
        with get_db_connection(self.pg_config) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                all_params = [query_vec] + filter_params + [query_vec, top_k]
                cur.execute(query_sql, all_params)
                for row in cur.fetchall():
                    sim = float(row["similarity"]) if row["similarity"] is not None else 0.0
                    results.append(
                        SearchResult(
                            chunk_id=row["chunk_id"],
                            book_id=row["book_id"],
                            clean_title=row["clean_title"],
                            author=row["author"],
                            domain=row["domain"],
                            subject=row["subject"],
                            source_path=row["source_path"],
                            page_number=row["page_number"],
                            section_title=row["section_title"],
                            content=row["content"],
                            score=sim,
                            vector_score=sim,
                        )
                    )

        return results

    def search_chunks_sparse(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Executes sparse full-text search against PostgreSQL tsvector/GIN index.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        filter_clauses, filter_params = self._build_filter_clauses(filters, table_alias="b")

        where_clauses = ["c.tsv @@ plainto_tsquery('english', %s)"]
        if filter_clauses:
            where_clauses.extend(filter_clauses)
        where_sql = "WHERE " + " AND ".join(where_clauses)

        query_sql = f"""
        SELECT 
            c.id as chunk_id,
            c.book_id,
            b.clean_title,
            b.author,
            b.domain,
            b.subject,
            b.source_path,
            c.page_number,
            c.section_title,
            c.content,
            ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS rank
        FROM book_chunks c
        JOIN books b ON b.id = c.book_id
        {where_sql}
        ORDER BY rank DESC
        LIMIT %s;
        """

        results: List[SearchResult] = []
        with get_db_connection(self.pg_config) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                all_params = [cleaned_query, cleaned_query] + filter_params + [top_k]
                cur.execute(query_sql, all_params)
                for row in cur.fetchall():
                    rnk = float(row["rank"]) if row["rank"] is not None else 0.0
                    results.append(
                        SearchResult(
                            chunk_id=row["chunk_id"],
                            book_id=row["book_id"],
                            clean_title=row["clean_title"],
                            author=row["author"],
                            domain=row["domain"],
                            subject=row["subject"],
                            source_path=row["source_path"],
                            page_number=row["page_number"],
                            section_title=row["section_title"],
                            content=row["content"],
                            score=rnk,
                            text_score=rnk,
                        )
                    )

        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Executes hybrid dense vector + sparse full-text search fused with Reciprocal Rank Fusion (RRF).
        """
        dense_results = self.search_chunks_dense(query, top_k=dense_top_k, filters=filters)
        sparse_results = self.search_chunks_sparse(query, top_k=sparse_top_k, filters=filters)

        # Merge results by chunk_id
        chunk_map: Dict[int, SearchResult] = {}
        rrf_scores: Dict[int, float] = {}

        # Accumulate dense RRF
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk_id
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + compute_rrf_score(rank, k=rrf_k, weight=dense_weight)

        # Accumulate sparse RRF
        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk_id
            if cid in chunk_map:
                chunk_map[cid].text_score = res.text_score
            else:
                chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + compute_rrf_score(rank, k=rrf_k, weight=sparse_weight)

        # Sort by final RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_results: List[SearchResult] = []
        for cid in sorted_chunk_ids[:top_k]:
            item = chunk_map[cid]
            item.score = rrf_scores[cid]
            final_results.append(item)

        return final_results

    def search_hierarchical(
        self,
        query: str,
        top_books: int = 3,
        chunks_per_book: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Hierarchical two-tier retrieval:
        1. Identifies the top macro book catalog matches.
        2. Retrieves the most relevant passage chunks scoped to those specific book IDs.
        """
        matched_books = self.search_books_dense(query, top_k=top_books, filters=filters)
        if not matched_books:
            return []

        book_ids = [b.book_id for b in matched_books]
        query_vec = self.provider.embed_text(query)

        query_sql = """
        SELECT 
            c.id as chunk_id,
            c.book_id,
            b.clean_title,
            b.author,
            b.domain,
            b.subject,
            b.source_path,
            c.page_number,
            c.section_title,
            c.content,
            1.0 - (c.embedding <=> %s::vector) AS similarity
        FROM book_chunks c
        JOIN books b ON b.id = c.book_id
        WHERE c.book_id = ANY(%s)
        ORDER BY c.embedding <=> %s::vector ASC
        LIMIT %s;
        """

        results: List[SearchResult] = []
        with get_db_connection(self.pg_config) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                limit = top_books * chunks_per_book
                cur.execute(query_sql, [query_vec, book_ids, query_vec, limit])
                for row in cur.fetchall():
                    sim = float(row["similarity"]) if row["similarity"] is not None else 0.0
                    results.append(
                        SearchResult(
                            chunk_id=row["chunk_id"],
                            book_id=row["book_id"],
                            clean_title=row["clean_title"],
                            author=row["author"],
                            domain=row["domain"],
                            subject=row["subject"],
                            source_path=row["source_path"],
                            page_number=row["page_number"],
                            section_title=row["section_title"],
                            content=row["content"],
                            score=sim,
                            vector_score=sim,
                        )
                    )

        return results


def build_rag_context(
    results: Sequence[SearchResult],
    max_char_limit: int = 8000,
) -> str:
    """
    Formats retrieved search results into a clean, cited context block for LLM prompt ingestion.
    """
    if not results:
        return "No relevant context passages found in database."

    blocks: List[str] = []
    current_chars = 0

    for idx, res in enumerate(results, start=1):
        header = f"--- [Passage {idx}] {res.citation} ---"
        body = res.content.strip()
        entry = f"{header}\n{body}\n"

        if current_chars + len(entry) > max_char_limit and blocks:
            break

        blocks.append(entry)
        current_chars += len(entry)

    return "\n".join(blocks)


def synthesize_rag_prompt(
    query: str,
    results: Sequence[SearchResult],
    system_instruction: Optional[str] = None,
) -> str:
    """
    Constructs a complete ready-to-use RAG prompt containing citations and user question.
    """
    context_str = build_rag_context(results)
    sys_inst = system_instruction or (
        "You are an expert technical AI research assistant. Answer the user's question accurately "
        "and concisely using only the retrieved library passages below. Always cite your source "
        "book titles and page numbers when stating facts."
    )

    prompt = f"""{sys_inst}

=== RETRIEVED LIBRARY CONTEXT ===
{context_str}
=================================

User Question: {query}

Answer:"""
    return prompt


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Vector + Keyword Search & RAG Retrieval Platform."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Search query or conceptual question",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["hybrid", "vector", "keyword", "hierarchical", "books"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain (e.g. 'Computer Science')",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Filter by subject (e.g. 'Artificial Intelligence')",
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Filter by author name",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Display formatted RAG context citations block",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Display full synthesized RAG prompt for LLM consumption",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Embedding provider (fastembed, ollama, openai)",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    filters: Dict[str, Any] = {}
    if parsed.domain:
        filters["domain"] = parsed.domain
    if parsed.subject:
        filters["subject"] = parsed.subject
    if parsed.author:
        filters["author"] = parsed.author

    try:
        provider = get_embedding_provider(provider_type=parsed.provider)
        engine = RAGSearchEngine(embedding_provider=provider)

        query = parsed.query

        if parsed.mode == "books":
            book_results = engine.search_books_dense(query, top_k=parsed.top_k, filters=filters)
            print(f"\nFound {len(book_results)} book catalog matches for '{query}':\n")
            for idx, b in enumerate(book_results, start=1):
                print(f"[{idx}] 《{b.clean_title}》 (Score: {b.score:.4f})")
                if b.author:
                    print(f"    Author: {b.author}")
                if b.domain or b.subject:
                    print(f"    Taxonomy: {b.domain} > {b.subject}")
                if b.summary:
                    print(f"    Summary: {b.summary[:120]}...")
                print()
            return 0

        # Passage search modes
        results: List[SearchResult] = []
        if parsed.mode == "hybrid":
            results = engine.search_hybrid(query, top_k=parsed.top_k, filters=filters)
        elif parsed.mode == "vector":
            results = engine.search_chunks_dense(query, top_k=parsed.top_k, filters=filters)
        elif parsed.mode == "keyword":
            results = engine.search_chunks_sparse(query, top_k=parsed.top_k, filters=filters)
        elif parsed.mode == "hierarchical":
            results = engine.search_hierarchical(query, top_books=3, chunks_per_book=parsed.top_k, filters=filters)

        if not results:
            print(f"\nNo passages found matching query '{query}'.")
            return 0

        if parsed.synthesize:
            print("\n" + "=" * 60)
            print("                SYNTHESIZED RAG PROMPT")
            print("=" * 60)
            print(synthesize_rag_prompt(query, results))
            print("=" * 60)
            return 0

        if parsed.context:
            print("\n" + "=" * 60)
            print("                 FORMATTED RAG CONTEXT")
            print("=" * 60)
            print(build_rag_context(results))
            print("=" * 60)
            return 0

        # Default: Print formatted result list
        print(f"\nRetrieved {len(results)} passage(s) for query '{query}' [{parsed.mode.upper()} SEARCH]:\n")
        for idx, res in enumerate(results, start=1):
            print(f"[{idx}] {res.citation} (Score: {res.score:.4f})")
            preview = res.content.replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            print(f"    Excerpt: \"{preview}\"")
            print()

        return 0

    except psycopg.OperationalError as e:
        print(f"\n[!] PostgreSQL Connection Error: {e}", file=sys.stderr)
        print("    Ensure PostgreSQL is running and seeded with data.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[!] Search Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
