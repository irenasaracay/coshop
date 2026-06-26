"""Utility functions for scoring catalog items against ground-truth targets.

A :class:`UtilityFunction` maps a list of item IDs to scalar scores in the
range 0–100, where 100 is a perfect match with the ground-truth target
(``xstar``) and 0 is no match.  Scores are used for evaluation (NDCG) and
optionally for re-ranking candidate lists.

Available implementations:

* :class:`ExactMatchUtilityFunction` — 100 iff item ID is in xstar, else 0.
* :class:`ColumnMatchingUtilityFunction` — percentage of columns that match
  xstar, with pluggable per-column match functions.
* :class:`ServerCosinePercentileUtilityFunction` — cosine percentile via the
  vector search API server.
* :class:`ColumnMatchWithServerCosinePercentileUtilityFunction` — weighted
  combination of column matching and server cosine percentile.

The ``MATCH_FUNCTIONS`` dict maps function name strings to callables and
supports the following per-column override keys: ``"exact_match"``,
``"jaccard_similarity"``, ``"contains_all"``, ``"greater_than_or_equal_to"``,
``"less_than_or_equal_to"``.
"""

from typing import Dict, List, Tuple, Any, Union, Optional
import numpy as np
import pandas as pd
import re
from ..utils.misc import check_na, parse_set
from ..utils.misc import explode_df
import os


def _get_vector_search_api_url(api_url: Optional[str] = None) -> str:
    url = api_url or os.environ.get("VECTOR_SEARCH_API_URL")
    if not url:
        raise RuntimeError(
            "Vector search API URL not configured. "
            "Pass api_url= at construction or set the VECTOR_SEARCH_API_URL environment variable."
        )
    return url


class UtilityFunction:
    """Abstract base class for item utility scoring functions.

    A UtilityFunction assigns a scalar score in [0, 100] to each catalog item
    reflecting how well the item matches the user's latent preferences (xstar).

    Subclasses must implement :meth:`__call__`.
    """

    def __call__(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """Score a list of catalog items.

        Args:
            item_ids: List of item ID strings to score.  Items not present in
                the catalog receive a score of ``0.0``.
            return_metadata: When ``True``, return a tuple
                ``(scores, metadata_list)`` where ``metadata_list[i]`` is a
                dict containing debugging information for ``item_ids[i]``
                (e.g. matched columns, percentile, reason for zero score).

        Returns:
            A list of scores in [0, 100], or a ``(scores, metadata)`` tuple
            when ``return_metadata=True``.
        """
        raise NotImplementedError("Subclasses must implement this method")


class ExactMatchUtilityFunction(UtilityFunction):
    """Utility function based on exact item-ID match with ground-truth targets.

    Assigns 100.0 to any item whose ID is in ``xstar``, and 0.0 to everything
    else.  Items not present in the catalog also receive 0.0.
    """

    def __init__(self, xstar: pd.DataFrame, catalog: pd.DataFrame):
        """Initialise an ExactMatchUtilityFunction.

        Args:
            xstar: DataFrame of ground-truth items (index = item IDs).
            catalog: Full catalog DataFrame (index = item IDs).
        """
        self.xstar_df = xstar
        self.catalog = catalog
        self._xstar_ids = {str(i) for i in self.xstar_df.index.astype(str).tolist()}

    def __call__(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Score a list of items using this utility function.

        Args:
            item_ids: List of item IDs to score
            return_metadata: If True, return tuple of (scores, metadata)

        Returns:
            List of scores (0-100) or tuple of (scores, metadata)
        """
        assert isinstance(item_ids, list), "item_ids must be a list of strings"

        if return_metadata:
            scores = []
            metadata = []
            for item_id in item_ids:
                score, item_metadata = self.score_single_item(item_id, True)
                scores.append(score)
                metadata.append(item_metadata)
            return scores, metadata
        else:
            return [self.score_single_item(item_id, False) for item_id in item_ids]

    def score_single_item(
        self, item_id: str, return_metadata: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Score an item by ID using this utility function.
        If the item_id is not in the catalog, returns 0.
        """
        # Check if item_id is in catalog
        if item_id not in self.catalog.index:
            if return_metadata:
                return 0.0, {"warning": f"Item ID not in catalog: {item_id}"}
            else:
                return 0.0

        # Check if item_id matches any xstar row
        if item_id in self._xstar_ids:
            if return_metadata:
                return 100.0, {"disliked_features": []}
            else:
                return 100.0

        if return_metadata:
            return 0.0, {"disliked_features": self.xstar_df.columns.tolist()}
        else:
            return 0.0


def _jaccard_match(x: Any, y: Any) -> float:
    """Jaccard similarity; returns 1.0 when both sets are empty (no division by zero)."""
    a, b = parse_set(x), parse_set(y)
    union = a | b
    if len(union) == 0:
        return 1.0  # both empty -> treat as match
    return len(a & b) / len(union)


def _contains_all_match(item_val: Any, required_val: Any) -> int:
    """Hard constraint: item must contain all required set elements (required ⊆ item). Returns 1 or 0."""
    required = parse_set(required_val)
    if not required:
        return 1  # no requirement -> match
    item_set = parse_set(item_val)
    return 1 if required.issubset(item_set) else 0


def _parse_numeric_value(v: Any) -> Optional[float]:
    """Best-effort parse of numeric-like values such as '>= 3.5' or '1,000'."""
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _parse_decade_start(token: str) -> Optional[int]:
    """Parse a single decade token like '80s' or '2000s' into a start year."""
    token = token.strip().lower()
    m4 = re.fullmatch(r"(\d{4})s", token)
    if m4:
        return int(m4.group(1))
    m2 = re.fullmatch(r"(\d{2})s", token)
    if m2:
        return 1900 + int(m2.group(1))
    return None


def _parse_decade_block_bounds(v: Any) -> Optional[tuple[int, int]]:
    """
    Parse decade labels into a 20-year window [start, end].
    Supports both single labels ('80s') and composed labels ('70s or 80s').
    """
    if not isinstance(v, str):
        return None
    tokens = re.findall(r"\b(?:\d{2}|\d{4})s\b", v.strip().lower())
    starts = [_parse_decade_start(t) for t in tokens]
    starts = [s for s in starts if s is not None]
    if not starts:
        return None
    if len(starts) == 1:
        # For a lone decade label, keep an overlapping 20-year interpretation.
        return starts[0] - 10, starts[0] + 9
    low = min(starts)
    high = max(starts)
    return low, high + 9


def _safe_float_cmp_ge(x: Any, y: Any) -> int:
    """Compare x >= y numerically; both NA -> 1, one NA -> 0, else compare."""
    if check_na(x) and check_na(y):
        return 1
    if check_na(x) or check_na(y):
        return 0
    x_dec = _parse_decade_block_bounds(x)
    y_dec = _parse_decade_block_bounds(y)
    if x_dec is not None and y_dec is not None:
        return int(not (x_dec[1] < y_dec[0] or y_dec[1] < x_dec[0]))
    if y_dec is not None:
        x_num = _parse_numeric_value(x)
        if x_num is None:
            return 0
        return int(y_dec[0] <= x_num <= y_dec[1])
    if x_dec is not None:
        y_num = _parse_numeric_value(y)
        if y_num is None:
            return 0
        return int(x_dec[0] >= y_num)
    x_num = _parse_numeric_value(x)
    y_num = _parse_numeric_value(y)
    if x_num is None or y_num is None:
        return 0
    return int(x_num >= y_num)


def _safe_float_cmp_le(x: Any, y: Any) -> int:
    """Compare x <= y numerically; both NA -> 1, one NA -> 0, else compare."""
    if check_na(x) and check_na(y):
        return 1
    if check_na(x) or check_na(y):
        return 0
    x_dec = _parse_decade_block_bounds(x)
    y_dec = _parse_decade_block_bounds(y)
    if x_dec is not None and y_dec is not None:
        return int(not (x_dec[1] < y_dec[0] or y_dec[1] < x_dec[0]))
    if y_dec is not None:
        x_num = _parse_numeric_value(x)
        if x_num is None:
            return 0
        return int(y_dec[0] <= x_num <= y_dec[1])
    if x_dec is not None:
        y_num = _parse_numeric_value(y)
        if y_num is None:
            return 0
        return int(x_dec[1] <= y_num)
    x_num = _parse_numeric_value(x)
    y_num = _parse_numeric_value(y)
    if x_num is None or y_num is None:
        return 0
    return int(x_num <= y_num)


MATCH_FUNCTIONS = {
    "exact_match": lambda x, y: int(str(x) == str(y)),
    "jaccard_similarity": _jaccard_match,
    "contains_all": _contains_all_match,
    "greater_than_or_equal_to": _safe_float_cmp_ge,
    "less_than_or_equal_to": _safe_float_cmp_le,
}


class ColumnMatchingUtilityFunction(UtilityFunction):
    """Utility function based on the percentage of matching columns with xstar.

    For each candidate item, computes the fraction of ``cols_to_compare``
    columns whose value matches the corresponding value in ``xstar``.  The
    fraction is multiplied by 100 to produce a score in [0, 100].

    When ``xstar`` has multiple rows (multiple acceptable ground-truth items),
    the score is the maximum over all xstar rows.

    Per-column match logic can be customised via ``special_match_functions``.
    Supported function names (keys of ``MATCH_FUNCTIONS``):

    * ``"exact_match"`` (default) — ``str(item) == str(xstar)``.
    * ``"jaccard_similarity"`` — Jaccard similarity of parsed sets.
    * ``"contains_all"`` — item set must be a superset of xstar set.
    * ``"greater_than_or_equal_to"`` — numeric ``item >= xstar``.
    * ``"less_than_or_equal_to"`` — numeric ``item <= xstar``.
    """

    def __init__(
        self,
        xstar: pd.DataFrame,
        catalog: pd.DataFrame,
        cols_to_compare: List[str] = None,
        cols_to_explode: List[str] = None,
        special_match_functions: Dict[str, str] = {},
    ):
        """Initialise a ColumnMatchingUtilityFunction.

        Args:
            xstar: DataFrame of ground-truth items (index = item IDs).
            catalog: Full catalog DataFrame (index = item IDs).
            cols_to_compare: Columns to include in the match computation.
                Columns missing from ``xstar`` are silently ignored.  Defaults
                to all columns of ``xstar``.
            cols_to_explode: Columns to explode (comma-separated string →
                multiple rows) before comparison.  Defaults to ``[]``.
            special_match_functions: Mapping from column name to match function
                name string.  Columns not listed use ``"exact_match"``.
        """
        self.xstar_df = xstar
        self.catalog = catalog
        self._xstar_ids = {str(i) for i in self.xstar_df.index.astype(str).tolist()}
        if cols_to_compare is None:
            cols_to_compare = self.xstar_df.columns.tolist()
        self.cols_to_compare = [
            c
            for c in (
                cols_to_compare
                if cols_to_compare is not None
                else xstar.columns.tolist()
            )
            if c in xstar.columns
        ]
        self.cols_to_explode = cols_to_explode if cols_to_explode is not None else []
        self.match_functions = {
            col: MATCH_FUNCTIONS[special_match_functions.get(col, "exact_match")]
            for col in cols_to_compare
        }


    def __call__(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Score a list of items using this utility function.
        """
        assert isinstance(item_ids, list), "item_ids must be a list of strings"

        if return_metadata:
            scores = []
            metadata = []
            for item_id in item_ids:
                score, item_metadata = self.score_single_item(item_id, True)
                scores.append(score)
                metadata.append(item_metadata)
            return scores, metadata
        else:
            return [self.score_single_item(item_id, False) for item_id in item_ids]

    def score_single_item(
        self, item_id: str, return_metadata: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Score an item by ID using this utility function.
        If the item_id is not in the catalog, returns 0.

        Per-column match logic follows ``special_match_functions`` (defaulting to
        exact match); see :class:`ColumnMatchingUtilityFunction`.
        """
        # Check if item_id is in catalog
        if item_id not in self.catalog.index:
            if return_metadata:
                return 0.0, {"warning": f"Item ID not in catalog: {item_id}"}
            else:
                return 0.0

        item_row = self.catalog.loc[item_id]
        item_row_exploded = pd.DataFrame([item_row.copy()])
        for c in self.cols_to_explode:
            item_row_exploded = explode_df(item_row_exploded, c)
        item_row_exploded = item_row_exploded.iloc[0]

        # For multiple targets, compute score against each and take the best match.
        best_score: float = -1.0
        best_disliked: List[str] = []
        best_matches = 0.0
        best_total = 0.0

        for _, xrow in self.xstar_df.iterrows():
            matches = 0.0
            total = 0.0
            disliked_features: List[str] = []
            for feature in self.cols_to_compare:  # loop through xstar features
                if feature in item_row_exploded:
                    # If xstar is na, skip in the calculation
                    if check_na(xrow[feature]):
                        continue

                    total += 1.0
                    # otherwise, if item is na, it's not a match
                    if check_na(item_row_exploded[feature]):
                        continue
                    
                    # If both have values, compare the values
                    match_function = self.match_functions.get(
                        feature, "exact_match"
                    )
                    m = match_function(item_row_exploded[feature], xrow[feature])
                    matches += m
                    if m == 0:
                        disliked_features.append(feature)
                else:
                    disliked_features.append(feature)

            score = float(matches) / float(total) * 100.0
            if score > best_score:
                best_score = score
                best_disliked = disliked_features
                best_total = total
                best_matches = matches

        best_score = max(0.0, best_score)
        if return_metadata:
            return best_score, {"disliked_features": best_disliked, "num_matches": best_matches, "total_cols": best_total}
        return best_score


class ServerCosinePercentileUtilityFunction(UtilityFunction):
    """Utility function that scores items via cosine percentile from the vector search server.

    For each candidate item, queries the vector search API to obtain the
    cosine-similarity percentile of that item relative to the ground-truth
    xstar items.  The percentile (0–1) is scaled to a score in [0, 100].

    xstar items always receive a score of 100 without an API call.  Items not
    found in the catalog receive 0.  Requires the vector search server to be
    running and reachable (set ``VECTOR_SEARCH_API_URL`` env var or pass
    ``api_url``).
    """

    def __init__(
        self,
        xstar: List[str],
        dataset_name: str,
        catalog: Optional[pd.DataFrame] = None,
        api_url: Optional[str] = None,
        version: Optional[str] = None,
    ):
        """Initialise a ServerCosinePercentileUtilityFunction.

        Args:
            xstar: List of ground-truth item ID strings used as reference IDs
                for the cosine percentile computation.
            dataset_name: Dataset identifier forwarded to the vector search
                API (e.g. ``"hm"``).
            catalog: Optional catalog DataFrame for item-ID validation.  When
                provided, unknown IDs are short-circuited to score 0 without
                an API call.
            api_url: Base URL of the vector search server.  Falls back to the
                ``VECTOR_SEARCH_API_URL`` environment variable.
            version: Dataset version (e.g. ``"v1"`` or ``"v2"``) forwarded to
                the vector search API so the server selects the matching index.
        """
        self.xstar = xstar
        self.dataset_name = dataset_name
        self.catalog = catalog
        self.version = version
        self._xstar_ids = {str(i) for i in xstar}
        self._api_url = api_url
        self.api_client = None

        try:
            from ..tools.vector_search_api_client import VectorSearchAPIClient

            self.api_client = VectorSearchAPIClient(api_url=_get_vector_search_api_url(api_url))
        except Exception as e:
            print(
                "[info] Optional smooth utility (u*) disabled: the vector-search "
                "server is not configured/reachable, so the embedding-based "
                "smooth relevance score won't be computed. This is an optional "
                "auxiliary signal — core evaluation (exact-match relevance, "
                "recall, etc.) is unaffected and will run normally. To enable it, "
                "set the VECTOR_SEARCH_API_URL environment variable to a running "
                f"vector-search server. (details: {e})"
            )

    def __call__(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Score a list of items using cosine percentile.
        Batches API calls for efficiency.

        Args:
            item_ids: List of item IDs to score
            return_metadata: If True, return tuple of (scores, metadata)

        Returns:
            List of scores (0-100) or tuple of (scores, metadata)
        """
        assert isinstance(item_ids, list), "item_ids must be a list of strings"

        if self.api_client is None:
            raise RuntimeError(
                "Smooth utility (u*) was requested but the vector-search API "
                "client is not initialized. This optional embedding-based score "
                "requires a running vector-search server; set the "
                "VECTOR_SEARCH_API_URL environment variable to enable it."
            )

        if len(item_ids) == 0:
            return ([], []) if return_metadata else []

        # Separate items into exact matches, invalid items, and items to query
        exact_match_indices = []
        invalid_indices = []
        query_indices = []
        query_item_ids = []

        for idx, item_id in enumerate(item_ids):
            item_id_str = str(item_id)
            if item_id_str in self._xstar_ids:
                exact_match_indices.append(idx)
            elif self.catalog is not None and item_id not in self.catalog.index:
                invalid_indices.append(idx)
            else:
                query_indices.append(idx)
                query_item_ids.append(item_id_str)

        # Initialize scores and metadata
        scores = [0.0] * len(item_ids)
        metadata_list = [{}] * len(item_ids) if return_metadata else None

        # Set exact match scores
        for idx in exact_match_indices:
            scores[idx] = 100.0
            if return_metadata:
                metadata_list[idx] = {"percentile": 1.0}

        # Set invalid item scores
        for idx in invalid_indices:
            scores[idx] = 0.0
            if return_metadata:
                metadata_list[idx] = {
                    "warning": f"Item ID not in catalog: {item_ids[idx]}"
                }

        # Batch query remaining items
        if query_item_ids:
            try:
                # Call the API with batched query_ids
                percentiles = self.api_client.cosine_percentile(
                    dataset=self.dataset_name,
                    query_id=query_item_ids,
                    reference_ids=[str(x) for x in self.xstar],
                    version=self.version,
                )

                # Convert percentiles (0-1) to scores (0-100) and assign to results
                for i, idx in enumerate(query_indices):
                    percentile = percentiles[i]
                    score = float(percentile * 100.0)
                    scores[idx] = score
                    if return_metadata:
                        metadata_list[idx] = {"percentile": percentile}

            except Exception as e:
                # If API call fails, set scores to 0 for query items
                for idx in query_indices:
                    scores[idx] = 0.0
                    if return_metadata:
                        metadata_list[idx] = {
                            "error": str(e),
                            "warning": "API call failed",
                        }

        if return_metadata:
            return scores, metadata_list
        else:
            return scores


class ColumnMatchWithServerCosinePercentileUtilityFunction(UtilityFunction):
    """Combined utility function: column matching + server cosine percentile.

    Treats the server cosine-percentile score as one extra synthetic "column"
    and averages it with the column-matching score.  Formally, if
    ``k = len(cols_to_compare)``:

    .. code-block:: none

        combined = ((k * col_match_score/100) + cos_score/100) / (k + 1) * 100

    This is the default ``ustar`` used by :class:`~coshop.data.dataset.Specification`
    when the vector search server is reachable.
    """

    def __init__(
        self,
        column_matching_uf: ColumnMatchingUtilityFunction,
        server_cosine_uf: ServerCosinePercentileUtilityFunction,
    ):
        """Initialise the combined utility function.

        Args:
            column_matching_uf: A fully-configured
                :class:`ColumnMatchingUtilityFunction` instance.
            server_cosine_uf: A fully-configured
                :class:`ServerCosinePercentileUtilityFunction` instance.
        """
        self.column_matching_uf = column_matching_uf
        self.server_cosine_uf = server_cosine_uf

    def score_columns_only(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Delegate to the wrapped ColumnMatchingUtilityFunction.
        """
        return self.column_matching_uf(item_ids, return_metadata)

    def score_server_cosine_only(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Delegate to the wrapped ServerCosinePercentileUtilityFunction.
        """
        return self.server_cosine_uf(item_ids, return_metadata)

    def __call__(
        self, item_ids: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        assert isinstance(item_ids, list), "item_ids must be a list of strings"

        if len(item_ids) == 0:
            return ([], []) if return_metadata else []

        if return_metadata:
            col_scores, col_meta = self.column_matching_uf(item_ids, True)
            cos_scores, cos_meta = self.server_cosine_uf(item_ids, True)
        else:
            col_scores = self.column_matching_uf(item_ids, False)
            cos_scores = self.server_cosine_uf(item_ids, False)
            col_meta = cos_meta = None

        # Number of "real" columns participating in the column matcher.
        k = len(getattr(self.column_matching_uf, "cols_to_compare", []))
        if k < 0:
            k = 0

        combined_scores: List[float] = []
        metadata_list: List[Dict[str, Any]] = [] if return_metadata else []

        for i, _ in enumerate(item_ids):
            col_score = float(col_scores[i]) if i < len(col_scores) else 0.0
            cos_score = float(cos_scores[i]) if i < len(cos_scores) else 0.0

            # Map both to [0, 1]
            m_cols = max(0.0, min(1.0, col_score / 100.0))
            m_cos = max(0.0, min(1.0, cos_score / 100.0))

            denom = float(k + 1) if k + 1 > 0 else 1.0
            combined = ((k * m_cols) + m_cos) / denom * 100.0
            combined = max(0.0, min(100.0, combined))
            combined_scores.append(combined)

            if return_metadata:
                meta: Dict[str, Any] = {
                    "column_match_score": col_score,
                    "server_cosine_percentile_score": cos_score,
                    "combined_score": combined,
                    "num_columns": k,
                }
                # Attach original metadata where available
                if col_meta is not None and i < len(col_meta):
                    meta["column_match_metadata"] = col_meta[i]
                if cos_meta is not None and i < len(cos_meta):
                    meta["server_cosine_percentile_metadata"] = cos_meta[i]
                metadata_list.append(meta)

        return (combined_scores, metadata_list) if return_metadata else combined_scores

    def score_single_item(
        self, item_id: str, return_metadata: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Score an item by ID using cosine percentile.
        If the item_id is not in the catalog, returns 0.

        Note: For efficiency, prefer using __call__ with a list of item_ids.
        """
        scores, metadata = self.__call__([item_id], return_metadata=True)
        if return_metadata:
            return scores[0], metadata[0]
        else:
            return scores[0]
