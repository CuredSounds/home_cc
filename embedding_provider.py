"""
Multi-Backend Embedding Engine for Semantic Vector Search & RAG.

Provides a unified interface supporting:
1. Local FastEmbed (ONNX runtime, default, 0-cost, 384-dim BAAI/bge-small-en-v1.5)
2. Local Ollama (REST API, nomic-embed-text / bge-m3)
3. Cloud APIs (OpenAI text-embedding-3-small, Azure OpenAI)
4. Persistent Caching Layer for embedding deduplication and rate-limit mitigation.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_CACHE_FILE = Path(".embedding_cache.json")


def l2_normalize(vector: Sequence[float]) -> List[float]:
    """Applies L2 unit normalization to a float vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm < 1e-12:
        return list(vector)
    return [float(x / norm) for x in vector]


class BaseEmbeddingProvider(abc.ABC):
    """Abstract base class for all embedding providers."""

    def __init__(self, model_name: str, dimension: Optional[int] = None) -> None:
        self.model_name = model_name
        self._dimension = dimension

    @abc.abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates embedding vectors for a batch of input texts."""
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for a single string."""
        results = self.embed_batch([text], batch_size=1)
        if not results:
            raise ValueError(f"Provider {self.__class__.__name__} failed to produce an embedding.")
        return results[0]

    def get_dimension(self) -> int:
        """Returns the vector dimensionality of this embedding model."""
        if self._dimension is not None:
            return self._dimension
        # Auto-discover dimension with a dummy embedding
        test_vec = self.embed_text("test")
        self._dimension = len(test_vec)
        return self._dimension


class FastEmbedProvider(BaseEmbeddingProvider):
    """
    Local ONNX-based embedding provider using FastEmbed.
    Runs locally on CPU/GPU with no external server or API keys required.
    """

    KNOWN_DIMENSIONS: Dict[str, int] = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-small-en": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "nomic-ai/nomic-embed-text-v1.5": 768,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        dimension: Optional[int] = None,
        threads: Optional[int] = None,
    ) -> None:
        dim = dimension or self.KNOWN_DIMENSIONS.get(model_name, 384)
        super().__init__(model_name=model_name, dimension=dim)
        self.threads = threads
        self._model = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name, threads=self.threads)
            except ImportError as e:
                raise ImportError(
                    "FastEmbed is not installed. Please install with `pip install fastembed`."
                ) from e
        return self._model

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        # FastEmbed yields numpy arrays
        raw_embeddings = list(model.embed(texts, batch_size=batch_size))
        vectors: List[List[float]] = []
        for vec in raw_embeddings:
            v_list = [float(x) for x in vec]
            vectors.append(l2_normalize(v_list))
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local or remote Ollama REST API embedding provider.
    Connects to Ollama instance (default: http://localhost:11434).
    """

    KNOWN_DIMENSIONS: Dict[str, int] = {
        "nomic-embed-text": 768,
        "bge-m3": 1024,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        dim = dimension or self.KNOWN_DIMENSIONS.get(model_name)
        super().__init__(model_name=model_name, dimension=dim)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "home_cc-rag/1.0"},
            method="POST",
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = resp.read().decode("utf-8")
                    return json.loads(resp_data)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))

        raise RuntimeError(
            f"Failed to connect to Ollama at {url} after {self.max_retries} attempts: {last_err}"
        )

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        # Attempt batch /api/embed first (Ollama >= 0.1.34)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                res = self._post_request("/api/embed", {"model": self.model_name, "input": chunk})
                if "embeddings" in res and isinstance(res["embeddings"], list):
                    for emb in res["embeddings"]:
                        results.append(l2_normalize([float(x) for x in emb]))
                    continue
            except Exception:
                # Fallback to single /api/embeddings for older Ollama versions
                pass

            # Single fallback
            for text in chunk:
                res = self._post_request("/api/embeddings", {"model": self.model_name, "prompt": text})
                if "embedding" in res:
                    results.append(l2_normalize([float(x) for x in res["embedding"]]))
                else:
                    raise ValueError(f"Unexpected response format from Ollama: {res}")

        if self._dimension is None and results:
            self._dimension = len(results[0])

        return results


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI and OpenAI-compatible Cloud API embedding provider (Azure OpenAI, LiteLLM, vLLM).
    """

    KNOWN_DIMENSIONS: Dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        dim = dimension or self.KNOWN_DIMENSIONS.get(model_name, 1536)
        super().__init__(model_name=model_name, dimension=dim)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _post_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key and "api.openai.com" in self.base_url:
            raise ValueError(
                "OPENAI_API_KEY environment variable or api_key parameter is required for OpenAI provider."
            )

        url = f"{self.base_url}/embeddings"
        payload = {"input": texts, "model": self.model_name}
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "home_cc-rag/1.0",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = resp.read().decode("utf-8")
                    parsed = json.loads(resp_data)
                    # OpenAI data array ordered by index
                    data_arr = sorted(parsed.get("data", []), key=lambda x: x.get("index", 0))
                    return [l2_normalize([float(x) for x in item["embedding"]]) for item in data_arr]
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))

        raise RuntimeError(f"OpenAI embedding request failed: {last_err}")

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            embeddings = self._post_embeddings(chunk)
            results.extend(embeddings)

        if self._dimension is None and results:
            self._dimension = len(results[0])

        return results


class EmbeddingCache:
    """
    Persistent key-value cache for embedding vectors to avoid duplicate API calls and computation.
    """

    def __init__(self, cache_file: Union[str, Path] = DEFAULT_CACHE_FILE) -> None:
        self.cache_file = Path(cache_file)
        self._memory_cache: Dict[str, List[float]] = {}
        self._dirty = False
        self._load()

    def _hash_key(self, model_name: str, text: str) -> str:
        content = f"{model_name}:{text}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _load(self) -> None:
        if self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._memory_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache {self.cache_file}: {e}")
                self._memory_cache = {}

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.cache_file.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache, f)
            temp_path.replace(self.cache_file)
            self._dirty = False
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def get(self, model_name: str, text: str) -> Optional[List[float]]:
        key = self._hash_key(model_name, text)
        return self._memory_cache.get(key)

    def set(self, model_name: str, text: str, vector: List[float]) -> None:
        key = self._hash_key(model_name, text)
        self._memory_cache[key] = vector
        self._dirty = True

    def __len__(self) -> int:
        return len(self._memory_cache)


class CachedEmbeddingProvider(BaseEmbeddingProvider):
    """
    Wraps any BaseEmbeddingProvider with persistent caching.
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        cache: Optional[EmbeddingCache] = None,
        auto_save: bool = True,
    ) -> None:
        super().__init__(model_name=provider.model_name, dimension=provider.get_dimension())
        self.provider = provider
        self.cache = cache if cache is not None else EmbeddingCache()
        self.auto_save = auto_save

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        # Check cache
        for idx, text in enumerate(texts):
            cached_vec = self.cache.get(self.model_name, text)
            if cached_vec is not None:
                results[idx] = cached_vec
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        # Query provider for cache misses
        if missing_texts:
            new_vectors = self.provider.embed_batch(missing_texts, batch_size=batch_size)
            for idx, text, vec in zip(missing_indices, missing_texts, new_vectors):
                results[idx] = vec
                self.cache.set(self.model_name, text, vec)

            if self.auto_save:
                self.cache.save()

        return [r for r in results if r is not None]


def get_embedding_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    cache_file: Optional[Union[str, Path]] = None,
    dimension: Optional[int] = None,
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """
    Factory function instantiating the requested embedding backend.
    Reads configuration from EMBEDDING_PROVIDER environment variable if not specified.
    """
    ptype = (provider_type or os.getenv("EMBEDDING_PROVIDER", "fastembed")).lower().strip()

    provider: BaseEmbeddingProvider

    if ptype in ("fastembed", "local", "onnx"):
        m_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_FASTEMBED_MODEL)
        provider = FastEmbedProvider(model_name=m_name, dimension=dimension, **kwargs)
    elif ptype == "ollama":
        m_name = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        provider = OllamaEmbeddingProvider(model_name=m_name, dimension=dimension, **kwargs)
    elif ptype in ("openai", "azure", "cloud"):
        m_name = model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        provider = OpenAIEmbeddingProvider(model_name=m_name, dimension=dimension, **kwargs)
    else:
        raise ValueError(
            f"Unknown embedding provider '{provider_type}'. Supported: 'fastembed', 'ollama', 'openai'."
        )

    if use_cache:
        c = EmbeddingCache(cache_file=cache_file or DEFAULT_CACHE_FILE)
        return CachedEmbeddingProvider(provider=provider, cache=c)

    return provider
