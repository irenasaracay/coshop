"""
History search tool for policies that can search the user's purchase/rating/review history.

Uses the existing vector_search API with allowed_ids to restrict results to items
in the user's history, then augments each result with the historical_data value
(user-specific info: ratings, reviews, "I purchased this", etc.).
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool, Tool
import json


HISTORY_SEARCH_TOOL_DESCRIPTION = (
    "Search the user's past purchase, rating, or review history. "
    "The first argument, query, is a natural language string describing what kinds of items you're looking for. "
    "Pass an empty query string to browse or list history items without semantic filtering — "
    "this will return up to max_items items from the user's history in no particular order. "
    "The second argument, max_items, is the maximum number of items from the user's history to return. "
    "The optional third argument, filters, is a dict mapping catalog column names to lists of allowed values. "
    "Only items whose column value exactly matches one of the listed values are returned. "
    "Use 'NA' as a value to match rows where that column is missing/null. "
    "Values are compared as strings. "
    "IMPORTANT: filters only support exact equality — no negation, inequality, or partial matching. "
    "Results will include a description of the item, the item ID, and any user-specific data (e.g. their rating)."
)

# Sentinel strings the model may use to request null/missing rows.
_NA_SENTINELS = {"na", "nan", "none", "null", "n/a", ""}


def get_history_search_tool(
    historical_data: Dict[str, str],
    dataset_name: str,
    version: Optional[str] = None,
    api_url: Optional[str] = None,
    max_items_limit: Optional[int] = None,
    threshold: Optional[float] = None,
    catalog: Optional[pd.DataFrame] = None,
    filterable_features: Optional[List[str]] = None,
) -> Tool:
    """
    Get a tool that allows the policy to search the user's purchase/rating/review history.

    Args:
        historical_data: Dict mapping item IDs to user-specific strings (e.g., ratings, reviews).
        dataset_name: Dataset name for the vector_search API.
        api_url: Vector search API URL. If None, uses VECTOR_SEARCH_API_URL env var.
        max_items_limit: If provided, caps max_items per call.
        threshold: Optional similarity threshold for vector search.
        catalog: Optional DataFrame of catalog items (used for column filtering).
            Ideally the historical subset only (spec.historical_df).
        filterable_features: Optional list of column names the agent may filter on.
            If None, the filters argument of the tool is accepted but ignored.

    Returns:
        A LangChain Tool for search_user_purchase_history(query, max_items, filters).
    """
    api_url = api_url or os.environ.get("VECTOR_SEARCH_API_URL")
    if not api_url:
        raise ValueError(
            "get_history_search_tool requires api_url or VECTOR_SEARCH_API_URL"
        )

    from .vector_search_api_client import VectorSearchAPIClient

    client = VectorSearchAPIClient(api_url=api_url)
    allowed_ids_base = list(historical_data.keys())

    # Pre-index the historical subset of the catalog for fast column lookups.
    hist_catalog: Optional[pd.DataFrame] = None
    if catalog is not None:
        hist_ids_in_catalog = [i for i in allowed_ids_base if i in catalog.index]
        hist_catalog = catalog.loc[hist_ids_in_catalog] if hist_ids_in_catalog else catalog.iloc[0:0]

    @tool(description=HISTORY_SEARCH_TOOL_DESCRIPTION)
    def search_user_purchase_history(
        query: str,
        max_items: int,
        filters: Optional[Dict[str, List[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the user's past purchase, rating, or review history."""
        if max_items is None:
            raise ValueError("max_items cannot be None")
        if max_items_limit is not None and max_items > max_items_limit:
            raise ValueError(f"max_items cannot be greater than {max_items_limit}")

        # Compute allowed_ids from hard filters if provided.
        allowed_ids = allowed_ids_base
        if filters and hist_catalog is not None and filterable_features is not None:
            invalid = [c for c in filters if c not in filterable_features]
            if invalid:
                raise ValueError(
                    f"Column(s) {invalid} are not filterable. "
                    f"Filterable columns: {filterable_features}"
                )
            mask = pd.Series(True, index=hist_catalog.index)
            for col, values in filters.items():
                if col not in hist_catalog.columns:
                    raise ValueError(f"Column '{col}' not found in catalog.")
                str_values = {str(v) for v in values}
                wants_na = bool(
                    str_values & _NA_SENTINELS
                    or {s.lower() for s in str_values} & _NA_SENTINELS
                )
                col_mask = hist_catalog[col].astype(str).isin(str_values)
                if wants_na:
                    col_mask |= hist_catalog[col].isna()
                mask &= col_mask
            allowed_ids = hist_catalog.index[mask].astype(str).tolist()

        ids, sims, texts = client.vector_search(
            dataset=dataset_name,
            version=version,
            q=query or "",
            m=max_items,
            allowed_ids=allowed_ids,
            threshold=threshold,
            corrupt_representations=False,
        )

        if not ids:
            return []

        results = []
        for i, item_id in enumerate(ids):
            item_id_str = str(item_id)
            text = texts[i] if texts and i < len(texts) else ""
            sim = sims[i] if sims and i < len(sims) else None

            user_interaction = historical_data.get(item_id_str, "")
            if user_interaction:
                augmented_text = (
                    f"User's past interaction: {user_interaction}\n\n{text}"
                )
            else:
                augmented_text = text

            result: Dict[str, Any] = {"id": item_id_str, "text": augmented_text}
            if sim is not None:
                result["similarity"] = sim
            results.append(result)

        return json.dumps(results)

    return search_user_purchase_history
