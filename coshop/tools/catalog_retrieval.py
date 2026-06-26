"""Catalog retrieval tools for shopping-assistant policies.

Provides two LangChain tool factories:

- :func:`get_hard_filter_retrieval_tool` — exposes a ``search_web(query, max_items, filters)``
  tool that accepts optional column-level hard filters before running semantic search.
- :func:`get_retrieval_tool` — simpler ``search_web(query, max_items)`` variant with no
  explicit filter argument.

Both tools enforce optional per-call and global item-count budgets at runtime.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .retrieval.retrieval import get_retrieval_fn


def get_hard_filter_retrieval_tool(
    retrieval_name: str,
    catalog: pd.DataFrame,
    filterable_features: List[str],
    max_items_limit: Optional[int] = None,
    execution_max_queries: Optional[int] = None,
    execution_global_max: Optional[int] = None,
    show_col_names: bool = False,
    **retrieval_fn_kwargs
):
    """Return a LangChain tool that searches the catalog with optional hard column filters.

    Unlike :func:`get_retrieval_tool`, the returned ``search_web`` tool accepts an
    optional ``filters`` argument so the agent can pin exact column values before
    the semantic/BM25 search runs.  Filters are applied by computing an
    ``allowed_ids`` set and passing it to the underlying retrieval function.

    The inner ``search_web`` tool signature is::

        search_web(query: str, max_items: int, filters: Optional[Dict[str, List[Any]]] = None)

    **Filter semantics:**

    - Keys are catalog column names (must be in ``filterable_features``).
    - Values are lists of *exact* allowed values (string-coerced before comparison).
    - Include ``"NA"`` in the list to match rows where the column is null/missing.
    - No negation, inequality, or partial matching is supported.

    Args:
        retrieval_name: Retrieval backend to use.  Forwarded to
            :func:`~coshop.tools.retrieval.retrieval.get_retrieval_fn` along with
            ``catalog``, ``eval_expression_columns=filterable_features``, and any
            extra ``**retrieval_fn_kwargs``.
        catalog: Full catalog DataFrame.  Used to derive ``allowed_ids`` from
            ``filters`` and to build per-column example values when
            ``show_col_names=True``.
        filterable_features: Column names the agent may filter on.  Attempting to
            filter on any other column raises ``ValueError`` at call time.
        max_items_limit: If set, the tool raises ``ValueError`` when the agent
            requests more than this many items in a single call.
        execution_max_queries: If set, the tool raises ``RuntimeError`` after this
            many calls.
        execution_global_max: If set, the tool raises ``RuntimeError`` once the
            cumulative ``max_items`` requested across all calls exceeds this limit.
        show_col_names: When ``True``, the tool description lists each filterable
            column with up to 10 example values.  When ``False`` (default), a
            shorter generic description is shown.
        **retrieval_fn_kwargs: Additional keyword arguments forwarded to
            :func:`~coshop.tools.retrieval.retrieval.get_retrieval_fn`.

    Returns:
        A LangChain ``Tool`` wrapping ``search_web``.
    """
    # Set up retrieval fn
    retrieval_function = get_retrieval_fn(
        retrieval_name,
        catalog=catalog,
        eval_expression_columns=filterable_features,
        **retrieval_fn_kwargs,
    )

    # Set up retrieval tool
    from langchain_core.tools import tool

    call_count = [0]
    total_items_retrieved = [0]

    # Sentinel strings the model may use to request null/missing rows.
    _NA_SENTINELS = {"na", "nan", "none", "null", "n/a", ""}

    if show_col_names:
        # Build per-column descriptions: name + up to 10 unique non-null values,
        # plus "NA" appended when any nulls exist in the column.
        column_lines = []
        for col in filterable_features:
            if col in catalog.columns:
                unique_vals = catalog[col].dropna().unique().tolist()[:10]
                if catalog[col].isna().any():
                    unique_vals.append("NA")
                column_lines.append(f"  - {col}: {unique_vals}")

        columns_block = "\n".join(column_lines) if column_lines else "  (none)"

        base_description = (
            "Search the catalog for products using a natural-language query, with optional "
            "hard filters that pin exact values for specific catalog columns.\n\n"
            "Arguments:\n"
            "  query (str): Natural-language search query. An empty string returns the full catalog.\n"
            "  max_items (int): Maximum number of items to return.\n"
            "  filters (dict, optional): Maps column names to a list of allowed values. "
            "Only items whose column value exactly matches one of the listed values are returned. "
            "This can help denoise your results. "
            "Values are compared as strings, so '1.0', 'True', and 'false' will be compared "
            "against the string representation of each cell. "
            "To match rows where a column is missing/null, include 'NA' in the value list. "
            "IMPORTANT: there is no support for negation (no NOT), inequality (no <, <=, >, >=), "
            "or partial matching — only exact equality.\n\n"
            f"Filterable columns (column_name: [up to 10 example values]):\n{columns_block}"
        )
    else:
        base_description = (
            "Search the web for products. The first argument is a string used to search the web. "
            "The second argument is the maximum number of items to return for this query. "
            "The more specific the query, the more likely you are to find a particular product. "
            "If the query is too specific, you may get no results. Note that you may have to "
            "obtain multiple documents about the same item to fully understand it: you can match "
            "items based on their ids or names.\n"
            "An empty query will return the entire web. The search query should be in natural language.\n"
            "Optionally, supply a 'filters' dict mapping column names to a list of EXACT allowed "
            "values to restrict results to items matching those column values. "
            "This can help denoise your results. "
            "Use 'NA' as a value to match rows where that column is missing/null. "
            "IMPORTANT: there is no support for negation (no NOT) or inequality (no <, <=, >, >=)."
        )

    budget_info_parts = []
    if max_items_limit is not None:
        budget_info_parts.append(f"Each query can return at most {max_items_limit} items")
    if execution_global_max is not None:
        budget_info_parts.append(
            f"Total budget of {execution_global_max} items across all queries"
        )
    if execution_max_queries is not None:
        budget_info_parts.append(f"Maximum {execution_max_queries} query call(s)")

    if budget_info_parts:
        budget_info = "\n\nBudget constraints: " + "; ".join(budget_info_parts) + "."
        full_description = base_description + budget_info
    else:
        full_description = base_description

    @tool(description=full_description)
    def search_web(
        query: str,
        max_items: int,
        filters: Optional[Dict[str, List[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # Enforce call-count budget
        if execution_max_queries is not None:
            if call_count[0] >= execution_max_queries:
                raise RuntimeError(
                    f"Query limit exceeded. This tool can only be called "
                    f"{execution_max_queries} time(s)."
                )
            call_count[0] += 1

        assert max_items is not None, "max_items cannot be None"

        if max_items_limit is not None and max_items > max_items_limit:
            raise ValueError(f"max_items cannot be greater than {max_items_limit}")

        if execution_global_max is not None:
            remaining = execution_global_max - total_items_retrieved[0]
            if max_items > remaining:
                raise RuntimeError(
                    f"Item limit exceeded. Requested {max_items} items, but only "
                    f"{remaining} items remaining out of {execution_global_max} total."
                )
            total_items_retrieved[0] += max_items

        # Compute allowed_ids from hard filters
        allowed_ids: Optional[List[str]] = None
        if filters:
            invalid = [c for c in filters if c not in filterable_features]
            if invalid:
                raise ValueError(
                    f"Column(s) {invalid} are not filterable. "
                    + (f"Filterable columns: {filterable_features}" if show_col_names else "")
                )

            mask = pd.Series(True, index=catalog.index)
            for col, values in filters.items():
                if col not in catalog.columns:
                    raise ValueError(
                        f"Column '{col}' not found in catalog. "
                        f"Filterable columns: {filterable_features}"
                    )
                # Coerce everything to str to forgive type mismatches
                # (e.g. LM passes "True" for a bool column, or "3.5" for a float column)
                str_values = {str(v) for v in values}

                # Check whether any supplied value is an NA sentinel
                # (case-insensitive: "NA", "NaN", "None", "null", "n/a", "")
                wants_na = bool(str_values & _NA_SENTINELS or
                                {s.lower() for s in str_values} & _NA_SENTINELS)

                col_mask = catalog[col].astype(str).isin(str_values)
                if wants_na:
                    col_mask |= catalog[col].isna()
                mask &= col_mask

            allowed_ids = catalog.index[mask].astype(str).tolist()

        # Delegate to the query function, passing allowed_ids if supported
        results_df = retrieval_function(query, max_items, allowed_ids=allowed_ids)
        return results_df.to_dict(orient="records")

    return search_web

def get_retrieval_tool(
    retrieval_name: str,
    max_items_limit: Optional[int] = None,
    execution_max_queries: Optional[int] = None,
    execution_global_max: Optional[int] = None,
    **retrieval_fn_kwargs,
):
    """Return a simple LangChain tool that searches the catalog without explicit filters.

    The returned ``search_web`` tool signature is::

        search_web(query: str, max_items: int)

    Use :func:`get_hard_filter_retrieval_tool` instead if the agent should be able
    to restrict results to exact column values.

    Args:
        retrieval_name: Retrieval backend to use.  Forwarded to
            :func:`~coshop.tools.retrieval.retrieval.get_retrieval_fn` along with
            any extra ``**retrieval_fn_kwargs``.
        max_items_limit: If set, raises ``ValueError`` when the agent requests more
            than this many items in a single call.
        execution_max_queries: If set, raises ``RuntimeError`` after this many calls.
        execution_global_max: If set, raises ``RuntimeError`` once cumulative
            ``max_items`` requested exceeds this limit.
        **retrieval_fn_kwargs: Additional keyword arguments forwarded to
            :func:`~coshop.tools.retrieval.retrieval.get_retrieval_fn`.

    Returns:
        A LangChain ``Tool`` wrapping ``search_web``.
    """

    # Set up retrieval fn
    retrieval_function = get_retrieval_fn(
        retrieval_name,
        **retrieval_fn_kwargs,
    )

    # Set up tool
    from langchain_core.tools import tool

    # Use lists to track state (mutable for closure)
    call_count = [0]
    total_items_retrieved = [0]

    # Build description with budget information
    base_description = "Search the web for products. The first argument is a string used to search the web. The second argument is the maximum number of items to return for this query. The more specific the query, the more likely you are to find a particular product. If the query is too specific, you may get no results. Note that you may have to obtain multiple documents about the same item to fully understand it: you can match items based on their ids or names.\nAn empty query will return the entire web. The search query should be in natural language."

    budget_info_parts = []
    if max_items_limit is not None:
        budget_info_parts.append(
            f"Each query can return at most {max_items_limit} items"
        )
    if execution_global_max is not None:
        budget_info_parts.append(
            f"Total budget of {execution_global_max} items across all queries"
        )
    if execution_max_queries is not None:
        budget_info_parts.append(f"Maximum {execution_max_queries} query call(s)")

    if budget_info_parts:
        budget_info = " Budget constraints: " + "; ".join(budget_info_parts) + "."
        full_description = base_description + budget_info
    else:
        full_description = base_description

    @tool(description=full_description)
    def search_web(query: str, max_items: int) -> List[Dict[str, Any]]:
        # Check query limit before processing
        if execution_max_queries is not None:
            if call_count[0] >= execution_max_queries:
                raise RuntimeError(
                    f"Query limit exceeded. This tool can only be called {execution_max_queries} time(s)."
                )
            call_count[0] += 1

        assert max_items is not None, "max_items cannot be None"
        # Enforce max_items_limit if provided
        if max_items_limit is not None and max_items > max_items_limit:
            raise ValueError(f"max_items cannot be greater than {max_items_limit}")

        # Check execution_global_max limit before processing
        if execution_global_max is not None:
            remaining_budget = execution_global_max - total_items_retrieved[0]
            if max_items > remaining_budget:
                raise RuntimeError(
                    f"Item limit exceeded. Requested {max_items} items, but only {remaining_budget} items remaining out of {execution_global_max} total."
                )
            # Track the requested max_items (not the actual items returned)
            total_items_retrieved[0] += max_items

        # Execute the query
        results = retrieval_function(query, max_items).to_dict(orient="records")

        return results

    return search_web

