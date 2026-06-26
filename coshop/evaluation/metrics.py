"""Evaluation metrics for the coshop benchmark.

Given a ranked list of item IDs produced by a policy, these functions compute
utility-based and exact-match metrics against the ground-truth xstar.

Main exports:
    compute_ndcg_at_k: Standard NDCG@k over a relevance score list.
    compute_evaluation_metrics: Full evaluation pipeline returning an
        :class:`~coshop.utils.custom_types.Evaluation` dataclass.
"""

from typing import List, Optional
from ..utils.custom_types import (
    Evaluation,
    SeenEvaluation,
    PredictedEvaluation,
    ItemEvaluationInfo,
)
from ..data.utility import ExactMatchUtilityFunction
from ..utils.caches import ScoringCache
import pandas as pd
import numpy as np
import math
import hashlib


def _get_all_items_with_scores(
    *,
    spec,
    catalog: pd.DataFrame,
    compute_column_ustar: bool = True,
    compute_embedding_ustar: bool = False,
    scoring_cache: Optional["ScoringCache"] = None,
) -> pd.DataFrame:
    """Score every item in the catalog against the spec's utility functions.

    Results are cached on disk via ``scoring_cache`` keyed by a hash of the
    catalog IDs and xstar rows.  On a cache hit the DataFrame is returned
    immediately without re-scoring.

    Args:
        spec: :class:`~coshop.data.dataset.Specification` instance providing
            ``column_ustar``, ``embedding_ustar``, ``ustar``, and
            ``xstar_series``.
        catalog: Full catalog DataFrame (index = item IDs).
        compute_column_ustar: Whether to compute column-matching utility scores.
            Defaults to ``True``.
        compute_embedding_ustar: Whether to compute embedding-based utility
            scores.  Defaults to ``False`` (requires vector search server).
        scoring_cache: Optional :class:`~coshop.utils.caches.ScoringCache`
            for persistent caching of scoring results.

    Returns:
        A DataFrame with columns ``["id", "column_ustar", "embedding_ustar",
        "ustar", "em"]``, one row per catalog item.
    """
    all_catalog_ids = list(catalog.index)
    catalog_ids_tuple = tuple(all_catalog_ids)
    xstar_series = getattr(spec, "xstar_series", None)
    if xstar_series is not None:
        try:
            xstar_hash = hashlib.md5(
                pd.util.hash_pandas_object(xstar_series).values.tobytes()
            ).hexdigest()
        except Exception:
            xstar_hash = str(hash(str(xstar_series)))
    else:
        xstar_hash = None
    # catalog_ids_tuple already uniquely identifies the catalog (ids differ
    # across datasets), so dataset_name is redundant and omitted. The version is
    # kept since the same catalog ids can be scored under different config
    # versions. Key layout matches the research codebase.
    dataset_version = getattr(spec, "version", None)
    cache_key = (
        catalog_ids_tuple,
        xstar_hash,
        compute_column_ustar,
        compute_embedding_ustar,
        "ustar",
        dataset_version,
    )

    if scoring_cache is not None:
        cached = scoring_cache.get(cache_key)
        if cached is not None:
            return cached

        # Cache miss: look for "similar" cache entries that already exist on disk.
        try:
            similar_keys = []
            alt_flag_combos = [
                (compute_column_ustar, not compute_embedding_ustar),
                (not compute_column_ustar, compute_embedding_ustar),
                (not compute_column_ustar, not compute_embedding_ustar),
            ]
            for alt_col, alt_emb in alt_flag_combos:
                alt_key = (catalog_ids_tuple, xstar_hash, alt_col, alt_emb, "ustar", dataset_version)
                if alt_key == cache_key:
                    continue
                alt_hash = scoring_cache._get_cache_key_str(alt_key)
                alt_path = scoring_cache._get_cache_path(alt_hash)
                if alt_path.exists():
                    similar_keys.append(
                        {
                            "n_catalog": len(catalog_ids_tuple),
                            "xstar_hash": xstar_hash,
                            "compute_column_ustar": alt_col,
                            "compute_embedding_ustar": alt_emb,
                            "file": alt_path.name,
                        }
                    )

            num_entries = len(list(scoring_cache.cache_dir.glob("*.pkl")))
            print(
                "[ScoringCache] MISS "
                f"(n_catalog={len(catalog_ids_tuple)}, "
                f"xstar_hash={xstar_hash}, "
                f"col={compute_column_ustar}, emb={compute_embedding_ustar}); "
                f"{num_entries} existing entries in {scoring_cache.cache_dir}."
            )
            if similar_keys:
                print("[ScoringCache] Found similar existing entries with different flags:")
                for sk in similar_keys:
                    print(
                        "  - "
                        f"xstar_hash={sk['xstar_hash']}, "
                        f"col={sk['compute_column_ustar']}, "
                        f"emb={sk['compute_embedding_ustar']}, "
                        f"file={sk['file']}"
                    )
        except Exception:
            pass

    print("Computing scores for all items in catalog...this may take a while...")
    column_ustar = getattr(spec, "column_ustar", None) if compute_column_ustar else None
    embedding_ustar = getattr(spec, "embedding_ustar", None) if compute_embedding_ustar else None
    # Unified ustar (may be a combined utility such as ColumnMatchWithServerCosinePercentileUtilityFunction)
    ustar_fn = getattr(spec, "ustar", None)
    if compute_column_ustar and column_ustar is not None:
        all_column_ustar_scores = column_ustar(all_catalog_ids)
    else:
        all_column_ustar_scores = [0.0] * len(all_catalog_ids)
    # Embedding ustar requires the vector search server. If it is unavailable,
    # degrade gracefully (NaN -> serialized as null) instead of aborting the
    # whole evaluation so that column-based metrics (including recall) are still
    # produced. NaN is used (rather than 0.0) so the resulting embedding metrics
    # are clearly "not computed" rather than a misleading zero.
    embedding_ustar_available = True
    if compute_embedding_ustar and embedding_ustar is not None:
        try:
            all_embedding_ustar_scores = embedding_ustar(all_catalog_ids)
        except RuntimeError as e:
            print(
                f"Warning: embedding_ustar unavailable, leaving embedding "
                f"metrics null: {e}"
            )
            all_embedding_ustar_scores = [float("nan")] * len(all_catalog_ids)
            embedding_ustar_available = False
    else:
        all_embedding_ustar_scores = [0.0] * len(all_catalog_ids)

    # The unified ustar may also depend on the vector search server; degrade the
    # same way and fall back to column_ustar when the server is unreachable.
    if ustar_fn is not None and embedding_ustar_available:
        try:
            all_ustar_scores = ustar_fn(all_catalog_ids)
        except RuntimeError as e:
            print(
                f"Warning: unified ustar unavailable, falling back to "
                f"column_ustar: {e}"
            )
            all_ustar_scores = all_column_ustar_scores
    elif ustar_fn is not None and not embedding_ustar_available:
        # Server is down; skip the (server-backed) unified ustar entirely.
        all_ustar_scores = all_column_ustar_scores
    else:
        # If no unified ustar is defined, default to column_ustar when available,
        # otherwise fall back to embedding_ustar.
        if compute_column_ustar and column_ustar is not None:
            all_ustar_scores = all_column_ustar_scores
        elif compute_embedding_ustar and embedding_ustar is not None:
            all_ustar_scores = all_embedding_ustar_scores
        else:
            all_ustar_scores = [0.0] * len(all_catalog_ids)
    if xstar_series is not None:
        all_em_scores = ExactMatchUtilityFunction(xstar_series, catalog)(all_catalog_ids)
    else:
        all_em_scores = [0] * len(all_catalog_ids)
    all_items_with_scores = pd.DataFrame(
        {
            "id": all_catalog_ids,
            "column_ustar": all_column_ustar_scores,
            "embedding_ustar": all_embedding_ustar_scores,
            "ustar": all_ustar_scores,
            "em": all_em_scores,
        }
    )
    if scoring_cache is not None:
        scoring_cache.put(cache_key, all_items_with_scores)
    return all_items_with_scores


def compute_ndcg_at_k(relevance_scores: list, k: int) -> float:
    """Compute Normalised Discounted Cumulative Gain at k (NDCG@k).

    Positions are 1-indexed; the discount at position ``i`` is
    ``1 / log2(i + 1)``.  The ideal DCG is computed from the same
    ``relevance_scores`` list sorted in descending order (not from a separate
    oracle list).

    Args:
        relevance_scores: Relevance scores in the *predicted* ranking order
            (e.g. ``ustar`` values for items ranked by the policy, with
            remaining catalog items appended at the end).
        k: Number of top positions to consider.

    Returns:
        NDCG@k as a float in [0, 1].  Returns ``0.0`` when the list is empty,
        ``k`` is zero, or the ideal DCG is zero.
    """
    if len(relevance_scores) == 0 or k == 0:
        return 0.0

    # Take top k items
    topk_scores = relevance_scores[:k]

    # Compute DCG@k: sum of (relevance / log2(position + 1))
    dcg = sum(score / np.log2(i + 2) for i, score in enumerate(topk_scores))

    # Compute IDCG@k: DCG of ideal ranking (sorted in descending order)
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))

    # Normalize: NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def _nan_fields_to_none(obj) -> None:
    """In-place: replace any float NaN attribute on a dataclass instance with None.

    Used so metrics that could not be computed (e.g. embedding ustar when the
    vector search server is down) serialize as JSON null instead of NaN.
    """
    from dataclasses import fields, is_dataclass

    if not is_dataclass(obj):
        return
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, float) and math.isnan(v):
            setattr(obj, f.name, None)


def compute_evaluation_metrics(
    *,
    spec,
    ranked_item_ids: List[str],
    catalog: pd.DataFrame,
    execution_max_per_retrieval: Optional[int],
    execution_max_queries: Optional[int] = None,
    execution_global_max: Optional[int] = None,
    k: int,
    seen_item_ids: Optional[List[str]] = None,
    compute_column_ustar: bool = True,
    compute_embedding_ustar: bool = True,
    cache_dir: Optional[str] = None,
) -> Evaluation:
    """Compute evaluation metrics for one elicitation episode.

    Scores all catalog items once (with optional caching), then computes
    **seen** metrics (items the policy retrieved during the conversation) and
    **predicted** metrics (the final ranked list returned by the policy).

    Predicted metrics are computed at both ``@1`` and ``@k``:

    * ``max_ustar_at_k`` / ``avg_ustar_at_k`` — max / mean combined utility
      score in the top-k.
    * ``ndcg_ustar_at_k`` — NDCG@k using combined utility as relevance.
    * ``recall_at_k`` — 1 if any xstar item appears in the top-k, else 0.

    Args:
        spec: :class:`~coshop.data.dataset.Specification` providing the
            utility functions and xstar reference.
        ranked_item_ids: Ordered list of item IDs predicted by the policy
            (position 0 = most-preferred).  May contain duplicates; they are
            deduplicated while preserving order.
        catalog: Full catalog DataFrame (index = item IDs).
        execution_max_per_retrieval: Maximum items returned per retrieval call
            (stored in the :class:`~coshop.utils.custom_types.Evaluation`
            metadata, formerly called ``m``).
        execution_max_queries: Maximum number of retrieval queries allowed
            during the episode (stored in metadata).
        execution_global_max: Maximum total items across all queries (stored
            in metadata).
        k: Number of top positions used for ``@k`` metrics.
        seen_item_ids: Item IDs that the policy retrieved at any point during
            the conversation (not just the final ranked list).  Defaults to
            ``[]``.
        compute_column_ustar: Whether to compute column-matching utility
            scores.  Defaults to ``True``.
        compute_embedding_ustar: Whether to compute embedding-based utility
            scores (requires vector search server).  Defaults to ``True``.
        cache_dir: Optional directory for the
            :class:`~coshop.utils.caches.ScoringCache`.  ``None`` disables
            caching.

    Returns:
        An :class:`~coshop.utils.custom_types.Evaluation` dataclass
        containing both seen and predicted sub-evaluations.
    """
    assert catalog is not None
    print("Computing evaluation metrics...")
    scoring_cache = ScoringCache(cache_dir=cache_dir) if cache_dir else None

    # Prepare seen_item_ids (default to empty list if not provided)
    if seen_item_ids is None:
        seen_item_ids = []

    # Deduplicate ranked_item_ids while preserving order
    seen = set()
    deduplicated_ranked_ids = []
    for item_id in ranked_item_ids:
        if item_id not in seen:
            seen.add(item_id)
            deduplicated_ranked_ids.append(item_id)

    # Deduplicate seen_item_ids
    seen_set = set(seen_item_ids)
    deduplicated_seen_ids = list(seen_set)

    # Get top k valid IDs (that exist in catalog) for predicted items
    predicted_ids = []
    for item_id in deduplicated_ranked_ids:
        if item_id in catalog.index:
            predicted_ids.append(item_id)
            # Stop once we have k valid items
            if len(predicted_ids) >= k:
                break

    # Filter seen_item_ids to only include those in catalog
    seen_ids_in_catalog = [
        item_id for item_id in deduplicated_seen_ids if item_id in catalog.index
    ]

    # Get all catalog items with scores (uses scoring cache)
    all_items_with_scores = _get_all_items_with_scores(
        spec=spec,
        catalog=catalog,
        compute_column_ustar=compute_column_ustar,
        compute_embedding_ustar=compute_embedding_ustar,
        scoring_cache=scoring_cache,
    )

    # ========== COMPUTE SEEN METRICS ==========
    seen_item_infos = []
    if len(seen_ids_in_catalog) > 0:
        seen_items_df = all_items_with_scores[
            all_items_with_scores["id"].isin(seen_ids_in_catalog)
        ].copy()
        for _, row in seen_items_df.iterrows():
            seen_item_infos.append(
                ItemEvaluationInfo(
                    id=row["id"],
                    column_ustar=float(row["column_ustar"]),
                    embedding_ustar=float(row["embedding_ustar"]),
                    ustar=float(row["ustar"]),
                    em=bool(row["em"] > 0),
                )
            )

        max_column_ustar = float(seen_items_df["column_ustar"].max())
        avg_column_ustar = float(seen_items_df["column_ustar"].mean())
        max_embedding_ustar = float(seen_items_df["embedding_ustar"].max())
        avg_embedding_ustar = float(seen_items_df["embedding_ustar"].mean())
        max_ustar = float(seen_items_df["ustar"].max())
        avg_ustar = float(seen_items_df["ustar"].mean())
        recall_at_seen = float((seen_items_df["em"] > 0).any())
    else:
        max_column_ustar = None
        avg_column_ustar = None
        max_embedding_ustar = None
        avg_embedding_ustar = None
        max_ustar = None
        avg_ustar = None
        recall_at_seen = None

    seen_eval = SeenEvaluation(
        seen_ids=seen_item_infos,
        max_column_ustar=max_column_ustar,
        avg_column_ustar=avg_column_ustar,
        max_embedding_ustar=max_embedding_ustar,
        avg_embedding_ustar=avg_embedding_ustar,
        max_ustar=max_ustar,
        avg_ustar=avg_ustar,
        recall_at_seen=recall_at_seen,
    )

    # ========== COMPUTE PREDICTED METRICS ==========
    predicted_item_infos = []
    if len(predicted_ids) > 0:
        predicted_items_df = all_items_with_scores[
            all_items_with_scores["id"].isin(predicted_ids)
        ].copy()
        # Sort predicted_items_df to match the order in predicted_ids
        predicted_items_df = (
            predicted_items_df.set_index("id").loc[predicted_ids].reset_index()
        )

        for _, row in predicted_items_df.iterrows():
            predicted_item_infos.append(
                ItemEvaluationInfo(
                    id=row["id"],
                    column_ustar=float(row["column_ustar"]),
                    embedding_ustar=float(row["embedding_ustar"]),
                    ustar=float(row["ustar"]),
                    em=bool(row["em"] > 0),
                )
            )

        # Get top k predicted items
        topk_predicted = predicted_items_df.head(k)

        if len(topk_predicted) > 0:
            max_column_ustar_at_k = float(topk_predicted["column_ustar"].max())
            avg_column_ustar_at_k = float(topk_predicted["column_ustar"].mean())
            max_embedding_ustar_at_k = float(topk_predicted["embedding_ustar"].max())
            avg_embedding_ustar_at_k = float(topk_predicted["embedding_ustar"].mean())
            max_ustar_at_k = float(topk_predicted["ustar"].max())
            avg_ustar_at_k = float(topk_predicted["ustar"].mean())
            recall_at_k = float((topk_predicted["em"] > 0).any())
        else:
            max_column_ustar_at_k = 0
            avg_column_ustar_at_k = 0
            max_embedding_ustar_at_k = 0
            avg_embedding_ustar_at_k = 0
            max_ustar_at_k = 0
            avg_ustar_at_k = 0
            recall_at_k = 0

        # Get top 1 predicted item (at_1 metrics)
        top1_predicted = predicted_items_df.head(1)
        if len(top1_predicted) > 0:
            column_ustar_at_1 = float(top1_predicted["column_ustar"].iloc[0])
            embedding_ustar_at_1 = float(top1_predicted["embedding_ustar"].iloc[0])
            ustar_at_1 = float(top1_predicted["ustar"].iloc[0])
            recall_at_1 = float((top1_predicted["em"] > 0).any())
        else:
            column_ustar_at_1 = None
            embedding_ustar_at_1 = None
            ustar_at_1 = None
            recall_at_1 = None

        # Compute NDCG metrics
        # Put predicted items at the top: create a list with predicted items first, then rest
        predicted_ids_set = set(predicted_ids)
        predicted_items_ordered = all_items_with_scores[
            all_items_with_scores["id"].isin(predicted_ids_set)
        ].copy()
        predicted_items_ordered = (
            predicted_items_ordered.set_index("id").loc[predicted_ids].reset_index()
        )
        rest_items = all_items_with_scores[
            ~all_items_with_scores["id"].isin(predicted_ids_set)
        ]
        items_with_scores_ordered = pd.concat(
            [predicted_items_ordered, rest_items], ignore_index=True
        )

        # NDCG for column_ustar
        ndcg_column_ustar_at_k = compute_ndcg_at_k(
            items_with_scores_ordered["column_ustar"].tolist(), k
        )

        # NDCG for embedding_ustar
        ndcg_embedding_ustar_at_k = compute_ndcg_at_k(
            items_with_scores_ordered["embedding_ustar"].tolist(), k
        )

        # NDCG for unified ustar
        ndcg_ustar_at_k = compute_ndcg_at_k(
            items_with_scores_ordered["ustar"].tolist(), k
        )

        # Binary NDCG
        ndcg_binary_at_k = compute_ndcg_at_k(
            items_with_scores_ordered["em"].tolist(), k
        )
    else:
        max_column_ustar_at_k = None
        avg_column_ustar_at_k = None
        max_embedding_ustar_at_k = None
        avg_embedding_ustar_at_k = None
        max_ustar_at_k = None
        avg_ustar_at_k = None
        column_ustar_at_1 = None
        embedding_ustar_at_1 = None
        ustar_at_1 = None
        recall_at_k = None
        recall_at_1 = None
        ndcg_column_ustar_at_k = None
        ndcg_embedding_ustar_at_k = None
        ndcg_ustar_at_k = None
        ndcg_binary_at_k = None

    predicted_eval = PredictedEvaluation(
        predicted_ids=predicted_item_infos,
        max_column_ustar_at_k=max_column_ustar_at_k,
        avg_column_ustar_at_k=avg_column_ustar_at_k,
        max_embedding_ustar_at_k=max_embedding_ustar_at_k,
        avg_embedding_ustar_at_k=avg_embedding_ustar_at_k,
        max_ustar_at_k=max_ustar_at_k,
        avg_ustar_at_k=avg_ustar_at_k,
        column_ustar_at_1=column_ustar_at_1,
        embedding_ustar_at_1=embedding_ustar_at_1,
        ustar_at_1=ustar_at_1,
        ndcg_column_ustar_at_k=ndcg_column_ustar_at_k,
        ndcg_embedding_ustar_at_k=ndcg_embedding_ustar_at_k,
        ndcg_ustar_at_k=ndcg_ustar_at_k,
        ndcg_binary_at_k=ndcg_binary_at_k,
        recall_at_k=recall_at_k,
        recall_at_1=recall_at_1,
    )

    # Normalize NaN scalar fields to None so they serialize as JSON null. NaNs
    # arise when a score source (e.g. embedding ustar) was unavailable.
    _nan_fields_to_none(seen_eval)
    _nan_fields_to_none(predicted_eval)
    for _info in seen_eval.seen_ids:
        _nan_fields_to_none(_info)
    for _info in predicted_eval.predicted_ids:
        _nan_fields_to_none(_info)

    return Evaluation(
        seen=seen_eval,
        predicted=predicted_eval,
        execution_max_per_retrieval=execution_max_per_retrieval,
        execution_max_queries=execution_max_queries,
        execution_global_max=execution_global_max,
        k=k,
    )
