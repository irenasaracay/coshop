"""
CoShop vector search API server (HNSW).

Builds embedding indexes at startup for all CoShop datasets, then serves:
  POST /vector_search      - HNSW nearest-neighbour search
  POST /encode             - Encode documents into embeddings
  POST /encode_query       - Encode queries into embeddings
  POST /cosine_distance    - Exact cosine distance between ids
  POST /cosine_percentile  - Percentile rank of an item's similarity score
  GET  /health             - Health check

Usage (development):
    python -m coshop.tools.vector_search_server \\
        --embedding_api_url http://localhost:8000 \\
        --embedding_model my-embed-model

Usage (production via gunicorn):
    EMBEDDING_API_URL=http://localhost:8000 EMBEDDING_MODEL=my-embed-model gunicorn \\
        --workers 1 --bind 0.0.0.0:3004 \\
        "coshop.tools.vector_search_server:create_app()"

The model id (EMBEDDING_MODEL / --embedding_model) is required and must match the
name the backend serves the model under (e.g. vLLM's --served-model-name, or
"text-embedding-3-small" for OpenAI).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import pandas as pd

try:
    from flask import Flask, jsonify, request
except ImportError as e:
    raise ImportError(
        "The vector search server requires Flask. Install with: pip install flask"
    ) from e

from ..data import get_dataset, SUPPORTED_VERSIONS, DEFAULT_VERSION
from ..utils.model import EmbeddingModelWrapper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_DATASETS: List[str] = ["hm", "movielens", "goodreads"]


def _index_key(dataset_name: str, version: str) -> str:
    return f"{dataset_name}:{version}"
TOP_K_PERCENTILE_ITEMS = 1000

_CACHE_BASE = Path(os.environ.get("COSHOP_CACHE_DIR", Path.home() / ".cache" / "coshop"))
INDEX_CACHE_DIR = _CACHE_BASE / "vector_search_indexes"
EMBEDDING_CACHE_DIR = str(_CACHE_BASE / "embeddings")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VectorIndex:
    """In-memory HNSW vector index for one dataset.

    Built once at server startup by :func:`_build_index` and then used
    read-only across all request threads.

    Attributes:
        df_id: Item IDs in catalog order, shape ``(N,)``.
        df_text: Item text strings in catalog order, length ``N``.
        embeddings: L2-normalised embeddings, shape ``(N, D)``.
        id_to_row: Maps ``str(id)`` to the row index in ``df_id`` /
            ``embeddings`` for fast lookup.
        catalog: Raw catalog DataFrame (used for column-based scoring).
        hnsw_index: hnswlib cosine-space HNSW index over ``embeddings``.
        feature_descriptions: Mapping of column name → human-readable
            description, copied from the dataset's ``true_features`` dict.
    """

    df_id: np.ndarray          # shape (N,)
    df_text: List[str]         # length N
    embeddings: np.ndarray     # shape (N, D), L2-normalised
    id_to_row: Dict[str, int]  # str(id) -> row index
    catalog: pd.DataFrame
    hnsw_index: hnswlib.Index
    feature_descriptions: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level state (written once at startup, read-only after)
# ---------------------------------------------------------------------------

_model: Optional[EmbeddingModelWrapper] = None
_indexes: Dict[str, VectorIndex] = {}

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(event: str, **fields: Any) -> None:
    try:
        print(json.dumps({"event": event, "ts": time.time(), **fields}))
    except Exception:
        pass


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _build_hnsw(embeddings: np.ndarray) -> hnswlib.Index:
    n, d = embeddings.shape
    idx = hnswlib.Index(space="cosine", dim=d)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(embeddings, np.arange(n))
    idx.set_ef(200)
    return idx


def _require_dataset(
    name: str, version: Optional[str] = None
) -> Tuple[Optional[VectorIndex], Optional[Any]]:
    version = version or DEFAULT_VERSION
    key = _index_key(name, version)
    idx = _indexes.get(key)
    if idx is None:
        return None, (
            jsonify(
                {
                    "error": (
                        f"Unknown dataset '{name}' (version='{version}'). "
                        f"Allowed: {sorted(_indexes)}"
                    )
                }
            ),
            400,
        )
    return idx, None


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def _cache_path(dataset_name: str, version: str = DEFAULT_VERSION) -> Path:
    return INDEX_CACHE_DIR / f"{dataset_name}_{version}"


def _manifest_matches(cache_dir: Path, model_name: str, num_items: int) -> bool:
    p = cache_dir / "manifest.json"
    if not p.exists():
        return False
    try:
        m = json.loads(p.read_text())
        return m.get("model_name") == model_name and m.get("num_items") == num_items
    except Exception:
        return False


def _save_index(cache_dir: Path, index: VectorIndex, model_name: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", index.embeddings)
    with open(cache_dir / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "df_id": index.df_id,
                "df_text": index.df_text,
                "id_to_row": index.id_to_row,
                "feature_descriptions": index.feature_descriptions,
            },
            f,
        )
    index.hnsw_index.save_index(str(cache_dir / "hnsw.bin"))
    (cache_dir / "manifest.json").write_text(
        json.dumps({"model_name": model_name, "num_items": len(index.df_id)})
    )
    print(f"[cache] Saved index to {cache_dir}")


def _load_index(cache_dir: Path, catalog: pd.DataFrame, feature_descriptions: dict) -> VectorIndex:
    embeddings = np.load(cache_dir / "embeddings.npy")
    with open(cache_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    n, d = embeddings.shape
    hnsw_index = hnswlib.Index(space="cosine", dim=d)
    hnsw_index.load_index(str(cache_dir / "hnsw.bin"), max_elements=n)
    hnsw_index.set_ef(200)
    print(f"[cache] Loaded index from {cache_dir} ({n} items)")
    return VectorIndex(
        df_id=meta["df_id"],
        df_text=meta["df_text"],
        embeddings=embeddings,
        id_to_row=meta["id_to_row"],
        catalog=catalog,
        hnsw_index=hnsw_index,
        feature_descriptions=meta.get("feature_descriptions", feature_descriptions),
    )


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def _build_index(dataset_name: str, version: str = DEFAULT_VERSION) -> VectorIndex:
    """Build (or load from cache) the HNSW index for a single dataset.

    If a cached index exists on disk with a matching model name and item count
    (verified via ``manifest.json``), it is loaded directly.  Otherwise the
    catalog is embedded with the global ``_model`` and a new index is built and
    saved to ``INDEX_CACHE_DIR / f"{dataset_name}_{version}"``.

    Args:
        dataset_name: One of the names in ``ALLOWED_DATASETS``
            (e.g. ``"hm"``, ``"movielens"``, ``"goodreads"``).
        version: Dataset version to build the index for (defaults to
            ``DEFAULT_VERSION``).

    Returns:
        A fully initialised :class:`VectorIndex`.
    """
    assert _model is not None
    t0 = time.perf_counter()
    dataset = get_dataset(dataset_name, dev=False, version=version)
    catalog = dataset.catalog
    feature_descriptions = dict(getattr(dataset, "true_features", None) or {})
    model_name = getattr(_model, "model_name", "unknown")

    cache_dir = _cache_path(dataset_name, version)
    if _manifest_matches(cache_dir, model_name, len(catalog)):
        index = _load_index(cache_dir, catalog, feature_descriptions)
        _log("index_loaded_from_cache", dataset=dataset_name, version=version,
             latency_ms=int((time.perf_counter() - t0) * 1000))
        return index

    print(f"[startup] Building index for '{dataset_name}' (version={version}, {len(catalog)} items) ...")
    ids = catalog.index.to_numpy()
    # Retrieval is always done on the unrestricted representation. The display
    # representation (restricted or not) is enforced at the vector-search level by
    # re-rendering returned ids via output_df (see DFQueryFunction.__call__).
    texts = [
        dataset.representation_unrestricted.row_to_str(row)
        for _, row in catalog.iterrows()
    ]
    emb = _model.encode(texts, show_progress_bar=True)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    emb = _l2_normalize(emb)
    id_to_row = {str(pid): int(i) for i, pid in enumerate(ids)}
    hnsw_index = _build_hnsw(emb)

    index = VectorIndex(
        df_id=ids,
        df_text=texts,
        embeddings=emb,
        id_to_row=id_to_row,
        catalog=catalog,
        hnsw_index=hnsw_index,
        feature_descriptions=feature_descriptions,
    )
    _save_index(cache_dir, index, model_name)
    _log("index_built", dataset=dataset_name, version=version, num_items=len(ids),
         latency_ms=int((time.perf_counter() - t0) * 1000))
    return index


def build_all_indexes() -> None:
    for name in ALLOWED_DATASETS:
        for ver in SUPPORTED_VERSIONS:
            _indexes[_index_key(name, ver)] = _build_index(name, ver)
    print(f"[startup] All indexes ready: {sorted(_indexes)}")


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _lookup_rows(
    index: VectorIndex, ids: List[Any]
) -> Tuple[Optional[List[int]], Optional[Any]]:
    rows, missing = [], []
    for id_ in ids:
        row = index.id_to_row.get(str(id_))
        if row is None:
            missing.append(str(id_))
        else:
            rows.append(row)
    if missing:
        return None, (jsonify({"error": f"IDs not found in dataset: {missing}"}), 404)
    return rows, None


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if _model is None:
        return jsonify({"status": "not_ready", "message": "model not initialized"}), 503
    return jsonify({
        "status": "ready",
        "model_name": getattr(_model, "model_name", None),
        "datasets": sorted(_indexes.keys()),
    })


@app.route("/encode", methods=["POST"])
def encode():
    if _model is None:
        return jsonify({"error": "model not initialized"}), 503
    t0 = time.perf_counter()
    texts = (request.get_json() or {}).get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    if isinstance(texts, str):
        texts = [texts]
    emb = _model.encode(texts, show_progress_bar=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    _log("encode", num_texts=len(texts), latency_ms=int((time.perf_counter() - t0) * 1000))
    return jsonify({"embeddings": emb.tolist(), "shape": list(emb.shape)})


@app.route("/encode_query", methods=["POST"])
def encode_query():
    if _model is None:
        return jsonify({"error": "model not initialized"}), 503
    t0 = time.perf_counter()
    texts = (request.get_json() or {}).get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    if isinstance(texts, str):
        texts = [texts]
    emb = _model.encode_query(texts, show_progress_bar=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    _log("encode_query", num_texts=len(texts), latency_ms=int((time.perf_counter() - t0) * 1000))
    return jsonify({"embeddings": emb.tolist(), "shape": list(emb.shape)})


@app.route("/cosine_distance", methods=["POST"])
def cosine_distance():
    if _model is None:
        return jsonify({"error": "model not initialized"}), 503
    t0 = time.perf_counter()
    payload = request.get_json(force=True) or {}

    index, err = _require_dataset(payload.get("dataset", ""), payload.get("version"))
    if err:
        return err

    query_ids = payload.get("query_id")
    reference_ids = payload.get("reference_ids")
    for name, val in [("query_id", query_ids), ("reference_ids", reference_ids)]:
        if not isinstance(val, list) or len(val) == 0:
            return jsonify({"error": f"{name} must be a non-empty list"}), 400

    ref_rows, err = _lookup_rows(index, reference_ids)
    if err:
        return err
    q_rows, err = _lookup_rows(index, query_ids)
    if err:
        return err

    distances = (1.0 - np.dot(index.embeddings[q_rows], index.embeddings[ref_rows].T)).min(axis=1).tolist()
    _log("cosine_distance", dataset=payload.get("dataset"),
         latency_ms=int((time.perf_counter() - t0) * 1000))
    return jsonify({"distances": distances})


@app.route("/cosine_percentile", methods=["POST"])
def cosine_percentile():
    if _model is None:
        return jsonify({"error": "model not initialized"}), 503
    t0 = time.perf_counter()
    payload = request.get_json(force=True) or {}

    index, err = _require_dataset(payload.get("dataset", ""), payload.get("version"))
    if err:
        return err

    query_ids = payload.get("query_id")
    reference_ids = payload.get("reference_ids")
    for name, val in [("query_id", query_ids), ("reference_ids", reference_ids)]:
        if not isinstance(val, list) or len(val) == 0:
            return jsonify({"error": f"{name} must be a non-empty list"}), 400

    ref_rows, err = _lookup_rows(index, reference_ids)
    if err:
        return err
    q_rows, err = _lookup_rows(index, query_ids)
    if err:
        return err

    ref_embs = index.embeddings[ref_rows]
    catalog_scores = (1.0 - np.dot(index.embeddings, ref_embs.T)).min(axis=1)
    n = len(catalog_scores)
    top_k = min(TOP_K_PERCENTILE_ITEMS, n)
    top_indices = set(np.argpartition(catalog_scores, top_k - 1)[:top_k].tolist())
    top_scores = catalog_scores[list(top_indices)]

    percentiles = []
    for q_row in q_rows:
        if q_row not in top_indices:
            percentiles.append(0.0)
        else:
            percentiles.append(float(np.mean(top_scores >= catalog_scores[q_row])))

    _log("cosine_percentile", dataset=payload.get("dataset"),
         latency_ms=int((time.perf_counter() - t0) * 1000))
    return jsonify({"percentiles": percentiles})


@app.route("/vector_search", methods=["POST"])
def vector_search():
    if _model is None:
        return jsonify({"error": "model not initialized"}), 503
    t0 = time.perf_counter()
    payload = request.get_json(force=True) or {}

    dataset_name = payload.get("dataset")
    version = payload.get("version")
    q = payload.get("q")
    m = payload.get("m")
    threshold = payload.get("threshold")
    allowed_ids = payload.get("allowed_ids")

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        return jsonify({"error": "dataset must be a non-empty string"}), 400
    if not isinstance(q, str):
        return jsonify({"error": "q must be a string"}), 400
    if m is not None and not isinstance(m, int):
        return jsonify({"error": "m must be an int or null"}), 400
    if allowed_ids is not None and not isinstance(allowed_ids, list):
        return jsonify({"error": "allowed_ids must be a list or null"}), 400

    index, err = _require_dataset(dataset_name, version)
    if err:
        return err

    allowed_set = {str(x) for x in allowed_ids} if allowed_ids is not None else None

    # Empty query: return items in catalog order
    if not q.strip():
        ids_out, texts_out, seen = [], [], set()
        for i in range(len(index.df_id)):
            id_str = str(index.df_id[i])
            if allowed_set is not None and id_str not in allowed_set:
                continue
            if id_str in seen:
                continue
            seen.add(id_str)
            ids_out.append(index.df_id[i])
            texts_out.append(index.df_text[i])
            if m is not None and len(ids_out) >= m:
                break
        return jsonify({"dataset": dataset_name, "m": m, "ids": ids_out,
                        "texts": texts_out, "similarities": [None] * len(ids_out)})

    # HNSW search
    q_emb = _l2_normalize(
        _model.encode_query([q.lower().strip()], show_progress_bar=False).reshape(1, -1)
    )
    k = min((m or 1000) * 3, len(index.df_id))
    labels, distances = index.hnsw_index.knn_query(q_emb, k=k)
    order = labels[0]
    sims = 1.0 - distances[0]

    if threshold is not None:
        mask = sims >= float(threshold)
        order, sims = order[mask], sims[mask]
    if allowed_set is not None:
        keep = np.array([str(index.df_id[i]) in allowed_set for i in order])
        order, sims = order[keep], sims[keep]

    ids_out, texts_out, sims_out, seen = [], [], [], set()
    for idx, sim in zip(order, sims):
        id_str = str(index.df_id[idx])
        if id_str in seen:
            continue
        seen.add(id_str)
        ids_out.append(index.df_id[idx])
        texts_out.append(index.df_text[idx])
        sims_out.append(float(sim))
        if m is not None and len(ids_out) >= m:
            break

    _log("vector_search", dataset=dataset_name, num_results=len(ids_out),
         latency_ms=int((time.perf_counter() - t0) * 1000))
    return jsonify({"dataset": dataset_name, "m": m, "ids": ids_out,
                    "texts": texts_out, "similarities": sims_out})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _init_model(embedding_api_url: str, embedding_model: str) -> None:
    global _model
    print(
        f"Loading embedding model '{embedding_model}' via API: {embedding_api_url}"
    )
    _model = EmbeddingModelWrapper(
        model_name=embedding_model,
        cache_dir=EMBEDDING_CACHE_DIR,
        use_cache=True,
        max_cache_size_mb=float("inf"),
        max_cache_files=None,
        batch_size=128,
        embedding_api_url=embedding_api_url,
    )
    print("Model ready.")


def create_app(
    embedding_api_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Flask:
    """WSGI app factory for gunicorn.

    The embedding API URL and model name are read from the EMBEDDING_API_URL and
    EMBEDDING_MODEL env vars when called without arguments (standard gunicorn
    factory invocation). Both are required.
    """
    url = embedding_api_url or os.environ.get("EMBEDDING_API_URL")
    if not url:
        raise RuntimeError(
            "Embedding API URL is required. "
            "Pass embedding_api_url= or set the EMBEDDING_API_URL environment variable."
        )
    model = embedding_model or os.environ.get("EMBEDDING_MODEL")
    if not model:
        raise RuntimeError(
            "Embedding model id is required. Pass embedding_model= or set the "
            "EMBEDDING_MODEL environment variable to the name the backend serves "
            "the model under (e.g. vLLM's --served-model-name)."
        )
    _init_model(url, model)
    build_all_indexes()
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="CoShop vector search API server")
    parser.add_argument(
        "--embedding_api_url",
        required=True,
        help="Base URL of the OpenAI-compatible embeddings API (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--embedding_model",
        default=os.environ.get("EMBEDDING_MODEL"),
        required=os.environ.get("EMBEDDING_MODEL") is None,
        help=(
            "Model id sent to the embeddings API. Must match the name the backend "
            "serves the model under (e.g. vLLM's --served-model-name). "
            "Falls back to the EMBEDDING_MODEL environment variable."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3004)
    args = parser.parse_args()

    _init_model(args.embedding_api_url, args.embedding_model)
    build_all_indexes()
    print(f"Starting vector search server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
