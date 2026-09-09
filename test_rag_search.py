"""
Unit tests for Hybrid Search, RRF Ranking, and RAG Prompt Synthesis.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from embedding_provider import BaseEmbeddingProvider
from pg_vector_db import PGConfig
from rag_search import (
    BookSearchResult,
    RAGSearchEngine,
    SearchResult,
    build_rag_context,
    compute_rrf_score,
    parse_args,
    synthesize_rag_prompt,
)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__(model_name="mock-provider", dimension=dimension)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class TestRAGSearch(unittest.TestCase):
    def test_compute_rrf_score(self) -> None:
        score_rank1 = compute_rrf_score(rank=1, k=60, weight=1.0)
        self.assertAlmostEqual(score_rank1, 1.0 / 61.0)

        score_rank2 = compute_rrf_score(rank=2, k=60, weight=1.0)
        self.assertAlmostEqual(score_rank2, 1.0 / 62.0)
        self.assertTrue(score_rank1 > score_rank2)

    def test_search_result_citation(self) -> None:
        res = SearchResult(
            chunk_id=1,
            book_id=10,
            clean_title="The Art of Electronics",
            author="Paul Horowitz",
            domain="Engineering",
            subject="Electronics",
            page_number=42,
            section_title="Bipolar Transistors",
            content="Transistors operate as current-controlled amplifiers.",
            score=0.95,
        )
        citation = res.citation
        self.assertIn("《The Art of Electronics》", citation)
        self.assertIn("by Paul Horowitz", citation)
        self.assertIn("Section: Bipolar Transistors", citation)
        self.assertIn("p. 42", citation)

    def test_build_rag_context_and_synthesize_prompt(self) -> None:
        results = [
            SearchResult(
                chunk_id=1,
                book_id=10,
                clean_title="Deep Learning",
                author="Ian Goodfellow",
                domain="Computer Science",
                subject="AI",
                page_number=105,
                section_title="Convolutional Networks",
                content="Convolution leverages parameter sharing and translation invariance.",
                score=0.92,
            )
        ]

        context = build_rag_context(results)
        self.assertIn("[Passage 1]", context)
        self.assertIn("《Deep Learning》", context)
        self.assertIn("Convolution leverages parameter sharing", context)

        prompt = synthesize_rag_prompt("How does convolution work?", results)
        self.assertIn("User Question: How does convolution work?", prompt)
        self.assertIn("=== RETRIEVED LIBRARY CONTEXT ===", prompt)
        self.assertIn("Convolution leverages parameter sharing", prompt)

    @patch("rag_search.get_db_connection")
    def test_search_books_dense(self, mock_get_conn: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "book_id": 1,
                "clean_title": "Reinforcement Learning: An Introduction",
                "subtitle": "Second Edition",
                "author": "Richard Sutton",
                "domain": "Computer Science",
                "subject": "Artificial Intelligence",
                "summary": "Foundational text on RL.",
                "similarity": 0.88,
            }
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        engine = RAGSearchEngine(
            pg_config=PGConfig(),
            embedding_provider=MockEmbeddingProvider(),
        )

        books = engine.search_books_dense("reinforcement learning algorithms", top_k=1)
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0].clean_title, "Reinforcement Learning: An Introduction")
        self.assertAlmostEqual(books[0].score, 0.88)

    @patch("rag_search.get_db_connection")
    def test_search_hybrid_rrf_merging(self, mock_get_conn: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Call 1: dense search returns chunk 101 (sim 0.9) and chunk 102 (sim 0.85)
        # Call 2: sparse search returns chunk 102 (rank 0.8) and chunk 103 (rank 0.7)
        mock_cursor.fetchall.side_effect = [
            [
                {
                    "chunk_id": 101,
                    "book_id": 1,
                    "clean_title": "Book A",
                    "author": "Author A",
                    "domain": "CS",
                    "subject": "AI",
                    "source_path": None,
                    "page_number": 1,
                    "section_title": "Sec 1",
                    "content": "Dense chunk 101 content",
                    "similarity": 0.9,
                },
                {
                    "chunk_id": 102,
                    "book_id": 1,
                    "clean_title": "Book A",
                    "author": "Author A",
                    "domain": "CS",
                    "subject": "AI",
                    "source_path": None,
                    "page_number": 2,
                    "section_title": "Sec 2",
                    "content": "Dense/Sparse chunk 102 content",
                    "similarity": 0.85,
                },
            ],
            [
                {
                    "chunk_id": 102,
                    "book_id": 1,
                    "clean_title": "Book A",
                    "author": "Author A",
                    "domain": "CS",
                    "subject": "AI",
                    "source_path": None,
                    "page_number": 2,
                    "section_title": "Sec 2",
                    "content": "Dense/Sparse chunk 102 content",
                    "rank": 0.8,
                },
                {
                    "chunk_id": 103,
                    "book_id": 2,
                    "clean_title": "Book B",
                    "author": "Author B",
                    "domain": "CS",
                    "subject": "ML",
                    "source_path": None,
                    "page_number": 10,
                    "section_title": "Sec 3",
                    "content": "Sparse only chunk 103 content",
                    "rank": 0.7,
                },
            ],
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        engine = RAGSearchEngine(
            pg_config=PGConfig(),
            embedding_provider=MockEmbeddingProvider(),
        )

        results = engine.search_hybrid("deep neural network", top_k=3)
        self.assertEqual(len(results), 3)

        # Chunk 102 was present in BOTH dense (rank 2) and sparse (rank 1), so it should rank first by RRF
        self.assertEqual(results[0].chunk_id, 102)
        self.assertIsNotNone(results[0].vector_score)
        self.assertIsNotNone(results[0].text_score)

    def test_parse_args(self) -> None:
        args = parse_args(["--query", "How do transformers work?", "--mode", "hybrid", "--top-k", "10", "--synthesize"])
        self.assertEqual(args.query, "How do transformers work?")
        self.assertEqual(args.mode, "hybrid")
        self.assertEqual(args.top_k, 10)
        self.assertTrue(args.synthesize)


if __name__ == "__main__":
    unittest.main()
