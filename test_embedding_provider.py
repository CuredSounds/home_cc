"""
Unit tests for Multi-Backend Embedding Engine and Caching Layer.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from embedding_provider import (
    BaseEmbeddingProvider,
    CachedEmbeddingProvider,
    EmbeddingCache,
    FastEmbedProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
    l2_normalize,
)


class TestEmbeddingProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "test_cache.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_l2_normalize(self) -> None:
        vec = [3.0, 4.0]
        norm_vec = l2_normalize(vec)
        self.assertAlmostEqual(norm_vec[0], 0.6)
        self.assertAlmostEqual(norm_vec[1], 0.8)
        norm = math.sqrt(sum(x * x for x in norm_vec))
        self.assertAlmostEqual(norm, 1.0)

        # Zero vector safety
        zero_vec = [0.0, 0.0]
        self.assertEqual(l2_normalize(zero_vec), [0.0, 0.0])

    def test_fastembed_provider_local(self) -> None:
        provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")
        self.assertEqual(provider.get_dimension(), 384)

        texts = ["Machine learning with vector embeddings.", "PostgreSQL pgvector search."]
        embeddings = provider.embed_batch(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 384)
        self.assertEqual(len(embeddings[1]), 384)

        # Test single text helper
        single_vec = provider.embed_text("A single book title.")
        self.assertEqual(len(single_vec), 384)
        norm = math.sqrt(sum(x * x for x in single_vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    @patch("embedding_provider.urllib.request.urlopen")
    def test_ollama_provider_batch_and_fallback(self, mock_urlopen: MagicMock) -> None:
        # Mock batch response from /api/embed
        dummy_embeddings = [[0.1] * 768, [0.2] * 768]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embeddings": dummy_embeddings}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        self.assertEqual(provider.get_dimension(), 768)

        results = provider.embed_batch(["First text", "Second text"])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 768)

    @patch("embedding_provider.urllib.request.urlopen")
    def test_openai_provider_batch(self, mock_urlopen: MagicMock) -> None:
        # Mock OpenAI /embeddings response
        dummy_data = [
            {"index": 0, "embedding": [0.05] * 1536},
            {"index": 1, "embedding": [0.08] * 1536},
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": dummy_data}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="sk-test-mock-key",
        )
        self.assertEqual(provider.get_dimension(), 1536)

        results = provider.embed_batch(["OpenAI query 1", "OpenAI query 2"])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 1536)

    def test_embedding_cache(self) -> None:
        cache = EmbeddingCache(cache_file=self.cache_file)
        self.assertEqual(len(cache), 0)

        vec = [0.1, 0.2, 0.3]
        cache.set("model-a", "hello world", vec)
        self.assertEqual(cache.get("model-a", "hello world"), vec)
        self.assertIsNone(cache.get("model-b", "hello world"))
        self.assertIsNone(cache.get("model-a", "different text"))

        cache.save()
        self.assertTrue(self.cache_file.is_file())

        # Reload cache from disk
        new_cache = EmbeddingCache(cache_file=self.cache_file)
        self.assertEqual(len(new_cache), 1)
        self.assertEqual(new_cache.get("model-a", "hello world"), vec)

    def test_cached_embedding_provider(self) -> None:
        mock_provider = MagicMock(spec=BaseEmbeddingProvider)
        mock_provider.model_name = "mock-model"
        mock_provider.get_dimension.return_value = 3
        mock_provider.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        cache = EmbeddingCache(cache_file=self.cache_file)
        cached_provider = CachedEmbeddingProvider(provider=mock_provider, cache=cache)

        # First call: cache miss, queries mock_provider
        res1 = cached_provider.embed_batch(["text 1", "text 2"])
        self.assertEqual(len(res1), 2)
        self.assertEqual(mock_provider.embed_batch.call_count, 1)

        # Second call with same texts: cache hit, mock_provider is NOT called again
        res2 = cached_provider.embed_batch(["text 1", "text 2"])
        self.assertEqual(res1, res2)
        self.assertEqual(mock_provider.embed_batch.call_count, 1)

        # Partial hit call: "text 1" is cached, "text 3" is missing
        mock_provider.embed_batch.return_value = [[0.7, 0.8, 0.9]]
        res3 = cached_provider.embed_batch(["text 1", "text 3"])
        self.assertEqual(len(res3), 2)
        self.assertEqual(mock_provider.embed_batch.call_count, 2)
        mock_provider.embed_batch.assert_called_with(["text 3"], batch_size=32)

    def test_factory_get_embedding_provider(self) -> None:
        fast_prov = get_embedding_provider("fastembed", use_cache=False)
        self.assertIsInstance(fast_prov, FastEmbedProvider)

        ollama_prov = get_embedding_provider("ollama", use_cache=False)
        self.assertIsInstance(ollama_prov, OllamaEmbeddingProvider)

        openai_prov = get_embedding_provider("openai", api_key="sk-test", use_cache=False)
        self.assertIsInstance(openai_prov, OpenAIEmbeddingProvider)

        cached_prov = get_embedding_provider("fastembed", use_cache=True, cache_file=self.cache_file)
        self.assertIsInstance(cached_prov, CachedEmbeddingProvider)

        with self.assertRaises(ValueError):
            get_embedding_provider("invalid_provider")


if __name__ == "__main__":
    unittest.main()
