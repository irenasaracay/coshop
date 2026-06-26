"""
VectorSearch API Client

Used by the local `query/vector_search.py` QueryFunction to delegate retrieval
to a shared server that holds the embedding model on GPU.

Also supports embedding operations (encode/encode_query) as a drop-in replacement
for EmbeddingAPIClient.

Env var:
  VECTOR_SEARCH_API_URL=http://localhost:8001
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests


def _safe_json(response: requests.Response, context: str = "response") -> Dict[str, Any]:
    """Parse response as JSON or raise a clear error with status and body snippet."""
    try:
        return response.json()
    except json.JSONDecodeError as e:
        body = (response.text or "").strip()
        snippet = (body[:500] + "…") if len(body) > 500 else body or "(empty body)"
        raise ValueError(
            f"Vector search API returned invalid JSON ({context}). "
            f"Status={response.status_code}, body={snippet!r}. "
            f"Check VECTOR_SEARCH_API_URL points to the vector search server (not vLLM). "
            f"Original error: {e}"
        ) from e


class VectorSearchAPIClient:
    """HTTP client for the CoShop vector-search API server.

    Wraps all endpoints of :mod:`coshop.tools.vector_search_server` and also
    implements the ``encode`` / ``encode_query`` interface so it can be used as
    a drop-in replacement for any embedding model wrapper.

    The client pings ``/health`` during ``__init__`` to fail fast if the server
    is unreachable.

    Attributes:
        api_url: Base URL of the running server (trailing slash stripped).
        timeout_s: Request timeout in seconds (default 120).
        model_name: Embedding model name reported by the server ``/health``
            endpoint; useful for cache-key generation.
    """

    def __init__(self, api_url: Optional[str] = None, timeout_s: float = 120.0):
        """
        Args:
            api_url: Base URL of the vector search server
                (e.g. ``"http://localhost:3004"``).  Falls back to the
                ``VECTOR_SEARCH_API_URL`` environment variable if not provided.
            timeout_s: Per-request HTTP timeout in seconds.  Defaults to ``120.0``.

        Raises:
            ValueError: If neither ``api_url`` nor ``VECTOR_SEARCH_API_URL`` is set.
            requests.HTTPError: If the health check fails.
        """
        if api_url is None:
            api_url = os.environ.get("VECTOR_SEARCH_API_URL")
        if not api_url:
            raise ValueError(
                "VectorSearchAPIClient requires api_url or VECTOR_SEARCH_API_URL"
            )
        self.api_url = api_url.rstrip("/")
        self.timeout_s = timeout_s

        # Fail fast if server is unavailable.
        health_data = self.health()
        # Store model_name for compatibility with EmbeddingAPIClient interface
        self.model_name = health_data.get("model_name")

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.api_url}/health", timeout=10.0)
        r.raise_for_status()
        return _safe_json(r, "health")

    def vector_search(
        self,
        *,
        dataset: str,
        q: str,
        m: Optional[int],
        version: Optional[str] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        corrupt_representations: bool = False,
        corruption_seed: Optional[int] = None,
        allowed_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Any], List[Optional[float]], Optional[List[str]]]:
        """Run a nearest-neighbour vector search against the server.

        Args:
            dataset: Dataset name registered on the server (e.g. ``"hm"``).
            q: Natural-language query string.  An empty string returns items in
                catalog order.
            m: Maximum number of results to return.  ``None`` returns all
                matching items.
            dataset_kwargs: Optional extra configuration forwarded to the server.
            threshold: Minimum cosine similarity score; items below are dropped.
            corrupt_representations: When ``True``, the server intentionally
                corrupts item text embeddings before searching.  Used for
                robustness experiments.
            corruption_seed: Random seed for reproducible corruption.  Only
                meaningful when ``corrupt_representations=True``.
            allowed_ids: Whitelist of item IDs (as strings) to restrict results
                to.  Items not in this list are excluded from the search.

        Returns:
            A ``(ids, similarities, texts)`` tuple where ``ids`` and
            ``similarities`` are parallel lists and ``texts`` is a list of item
            text strings or ``None`` if the server did not return text.
        """
        payload: Dict[str, Any] = {"dataset": dataset, "q": q, "m": m}
        if version is not None:
            payload["version"] = version
        if dataset_kwargs is not None:
            payload["dataset_kwargs"] = dataset_kwargs
        if threshold is not None:
            payload["threshold"] = threshold
        payload["corrupt_representations"] = corrupt_representations
        if corruption_seed is not None:
            payload["corruption_seed"] = corruption_seed
        if allowed_ids is not None:
            payload["allowed_ids"] = [str(x) for x in allowed_ids]

        r = requests.post(
            f"{self.api_url}/vector_search", json=payload, timeout=self.timeout_s
        )
        r.raise_for_status()
        data = _safe_json(r, "vector_search")
        return (
            data.get("ids", []),
            data.get("similarities", []),
            data.get("texts"),
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode texts into embeddings (for documents).
        
        Args:
            texts: Text(s) to encode
            show_progress_bar: Whether to show progress bar (ignored for API)
            **kwargs: Additional arguments (ignored for API client)
            
        Returns:
            numpy array of embeddings
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        payload = {
            "texts": texts,
            "show_progress_bar": show_progress_bar,
        }
        
        r = requests.post(
            f"{self.api_url}/encode", json=payload, timeout=self.timeout_s
        )
        r.raise_for_status()
        data = _safe_json(r, "encode")
        embeddings = np.array(data["embeddings"])
        
        # Handle shape: if we got a single embedding, return it as 1D
        # Otherwise return as 2D array
        if len(texts) == 1 and embeddings.shape[0] == 1:
            return embeddings[0]
        return embeddings

    def encode_query(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode queries into embeddings (for queries).
        
        Args:
            texts: Text(s) to encode as queries
            show_progress_bar: Whether to show progress bar (ignored for API)
            **kwargs: Additional arguments (ignored for API client)
            
        Returns:
            numpy array of embeddings
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        payload = {
            "texts": texts,
            "show_progress_bar": show_progress_bar,
        }
        
        r = requests.post(
            f"{self.api_url}/encode_query", json=payload, timeout=self.timeout_s
        )
        r.raise_for_status()
        data = _safe_json(r, "encode_query")
        embeddings = np.array(data["embeddings"])
        
        # Handle shape: if we got a single embedding, return it as 1D
        # Otherwise return as 2D array
        if len(texts) == 1 and embeddings.shape[0] == 1:
            return embeddings[0]
        return embeddings

    def cosine_distance(
        self,
        *,
        dataset: str,
        query_id: List[str],
        reference_ids: List[str],
        version: Optional[str] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Compute minimum cosine distance between query_ids and any reference_id.
        Supports batching for efficient processing of many query_ids.
        
        Args:
            dataset: Dataset name
            query_id: List of query item IDs
            reference_ids: List of reference item IDs
            dataset_kwargs: Optional dataset configuration
            
        Returns:
            List of minimum cosine distances (one per query_id)
        """
        payload: Dict[str, Any] = {
            "dataset": dataset,
            "query_id": query_id,
            "reference_ids": reference_ids,
        }
        if version is not None:
            payload["version"] = version
        if dataset_kwargs is not None:
            payload["dataset_kwargs"] = dataset_kwargs

        r = requests.post(
            f"{self.api_url}/cosine_distance", json=payload, timeout=self.timeout_s
        )
        r.raise_for_status()
        data = _safe_json(r, "cosine_distance")
        return [float(d) for d in data["distances"]]

    def cosine_percentile(
        self,
        *,
        dataset: str,
        query_id: List[str],
        reference_ids: List[str],
        version: Optional[str] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Compute percentile of query_ids' score compared to all items in the catalog.
        Supports batching for efficient processing of many query_ids.
        
        Score for each item v is: min_{x* in reference_ids} cosine_distance(v, x*)
        Returns the percentile of score(query_id) compared to scores of all catalog items.
        
        Args:
            dataset: Dataset name
            query_id: List of query item IDs
            reference_ids: List of reference item IDs
            dataset_kwargs: Optional dataset configuration
            
        Returns:
            List of percentiles (0-1, one per query_id) of query_ids' scores
        """
        payload: Dict[str, Any] = {
            "dataset": dataset,
            "query_id": query_id,
            "reference_ids": reference_ids,
        }
        if version is not None:
            payload["version"] = version
        if dataset_kwargs is not None:
            payload["dataset_kwargs"] = dataset_kwargs

        r = requests.post(
            f"{self.api_url}/cosine_percentile", json=payload, timeout=self.timeout_s
        )
        r.raise_for_status()
        data = _safe_json(r, "cosine_percentile")
        return [float(p) for p in data["percentiles"]]

