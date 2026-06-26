"""
Disk-based caches with optional LRU eviction.

Used by:
- ScoringCache: evaluation scoring in evaluation/metrics.py
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ScoringCache:
    """
    Disk-based cache for evaluation scoring.

    Key is the tuple used in evaluation/metrics.py:
    (catalog_ids_tuple, xstar_hash, compute_column_ustar, compute_embedding_ustar,
    "ustar", dataset_version).
    Value is the all_items_with_scores DataFrame
    (columns: id, column_ustar, embedding_ustar, ustar, em).
    """

    def __init__(
        self,
        cache_dir: str = ".cache/scoring",
        enable_lru: bool = False,
        max_size: Optional[int] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_lru = enable_lru
        self.max_size = max_size
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata (LRU tracking) from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"access_times": {}, "size": 0}
        return {"access_times": {}, "size": 0}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception:
            pass

    def _get_cache_key_str(self, key_tuple: Tuple[Any, ...]) -> str:
        """Generate a stable cache key string from the metrics key tuple."""
        return hashlib.sha256(pickle.dumps(key_tuple)).hexdigest()

    def _get_cache_path(self, cache_key_str: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key_str}.pkl"

    def get(self, key_tuple: Tuple[Any, ...]) -> Optional[Any]:
        """
        Retrieve a cached value.

        Returns:
            Cached all_items_with_scores DataFrame (columns: id, column_ustar, embedding_ustar, ustar, em),
            or None if not found.
        """
        cache_key_str = self._get_cache_key_str(key_tuple)
        cache_path = self._get_cache_path(cache_key_str)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                value = pickle.load(f)

            if self.enable_lru:
                self.metadata["access_times"][cache_key_str] = time.time()
                self._save_metadata()

            return value
        except Exception:
            return None

    def put(self, key_tuple: Tuple[Any, ...], value: Any):
        """Store a value in the cache."""
        cache_key_str = self._get_cache_key_str(key_tuple)
        cache_path = self._get_cache_path(cache_key_str)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)

            if self.enable_lru:
                current_time = time.time()
                self.metadata["access_times"][cache_key_str] = current_time

                if self.max_size is not None:
                    access_times = self.metadata["access_times"]
                    if len(access_times) > self.max_size:
                        sorted_keys = sorted(access_times.items(), key=lambda x: x[1])
                        num_to_remove = len(access_times) - self.max_size
                        for old_key, _ in sorted_keys[:num_to_remove]:
                            old_path = self._get_cache_path(old_key)
                            if old_path.exists():
                                old_path.unlink()
                            del access_times[old_key]

                self._save_metadata()
        except Exception:
            pass
