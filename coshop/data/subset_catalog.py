"""Utility for deterministically subsetting a catalog DataFrame.

When running the benchmark with large catalogs it can be useful to work with
a smaller subset for faster development iteration.  :func:`subset_catalog_df`
guarantees that all ground-truth target items (``xstar``) and, depending on
the chosen strategy, some historical items are always retained.

Usage::

    from coshop.data.subset_catalog import subset_catalog_df

    new_catalog = subset_catalog_df(
        catalog_df,
        fraction=0.01,
        xstar_ids={0: ["id1", "id2"], 1: ["id3"]},
        historical_data_ids={0: ["id4", "id5"], 1: ["id6"]},
    )
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Literal

import pandas as pd


def _compute_capped_historical_required(
    xstar_ids: set[str],
    per_user_hist: dict[int, set[str]],
    hist_candidates: set[str],
    max_per_user: int,
) -> set[str]:
    """Compute a minimal required item set that covers xstars and partial history.

    Greedily selects historical items so that each user's history contributes
    at most ``max_per_user`` items to the required set, preferring items that
    appear in the most users' histories (set-cover heuristic).

    Args:
        xstar_ids: Ground-truth target item IDs (always included).
        per_user_hist: Mapping from user index to their historical item IDs
            (already filtered to items present in the catalog).
        hist_candidates: Universe of historical item IDs eligible for
            inclusion.
        max_per_user: Maximum number of historical items retained per user.

    Returns:
        A set of item IDs that includes all ``xstar_ids`` and greedily covers
        up to ``max_per_user`` historical items per user.
    """
    required = set(xstar_ids)
    need_per_user = {ix: min(max_per_user, len(h)) for ix, h in per_user_hist.items()}
    covered_per_user = {ix: h & required for ix, h in per_user_hist.items()}

    while True:
        remaining = {
            ix: need_per_user[ix] - len(covered_per_user[ix])
            for ix in per_user_hist
        }
        if all(r <= 0 for r in remaining.values()):
            break
        best_item = None
        best_help = -1
        for item in hist_candidates:
            if item in required:
                continue
            help_count = sum(
                1 for ix in per_user_hist
                if remaining[ix] > 0 and item in per_user_hist[ix]
            )
            if help_count > best_help:
                best_help = help_count
                best_item = item
        if best_item is None or best_help == 0:
            break
        required.add(best_item)
        for ix in per_user_hist:
            if best_item in per_user_hist[ix]:
                covered_per_user[ix].add(best_item)
    return required


def subset_catalog_df(
    catalog: pd.DataFrame,
    fraction: float,
    xstar_ids: dict[int, List[str]],
    historical_data_ids: dict[int, List[str]],
    strategy: Literal["xstar", "xstar+historical", "xstar+prefer_historical"] = "xstar+prefer_historical",
) -> pd.DataFrame:
    """Deterministically subset a catalog DataFrame while preserving key items.

    Retains approximately ``fraction`` of the catalog, but always keeps:

    * ``"xstar"`` — all ground-truth target items.
    * ``"xstar+historical"`` — all target items **and** all historical items.
    * ``"xstar+prefer_historical"`` *(default)* — all target items plus a
      greedily-selected subset of historical items (up to 3 per user), chosen
      to maximise coverage across users with minimal catalog bloat.

    Remaining slots up to ``fraction * len(catalog)`` are filled with items
    from the catalog sorted by index (deterministic).

    Args:
        catalog: Full catalog DataFrame whose index contains item IDs (will be
            cast to ``str``).
        fraction: Target fraction of the full catalog to retain, e.g. ``0.01``
            for 1 %.
        xstar_ids: Mapping from spec index to list of ground-truth item IDs
            for that spec.
        historical_data_ids: Mapping from spec index to list of historical item
            IDs for that spec's user.
        strategy: Item retention strategy; one of ``"xstar"``,
            ``"xstar+historical"``, or ``"xstar+prefer_historical"``.

    Returns:
        A subset of ``catalog`` with ``str``-typed index, containing all
        required items plus deterministic (index-sorted) filler up to the
        target size.

    Raises:
        ValueError: If the required items alone exceed the target size by more
            than 20 items (i.e. the requested ``fraction`` is too small to be
            feasible).
        ValueError: If ``strategy`` is not one of the three supported values.
    """
    catalog_index = catalog.index.astype(str)
    catalog_index_set = set(catalog_index)

    target_size = max(1, int(len(catalog) * fraction))

    all_xstar: set[str] = set()
    per_user_hist: dict[int, set[str]] = {}
    all_historical: set[str] = set()

    for ix, xs in xstar_ids.items():
        all_xstar.update(str(x) for x in xs)

    for ix, hist in historical_data_ids.items():
        hist_in_catalog = {str(h) for h in hist if str(h) in catalog_index_set}
        per_user_hist[ix] = hist_in_catalog
        all_historical.update(hist_in_catalog)

    xstar_in_catalog = all_xstar & catalog_index_set

    if strategy == "xstar":
        required_ids = xstar_in_catalog
    elif strategy == "xstar+historical":
        required_ids = xstar_in_catalog | all_historical
    elif strategy == "xstar+prefer_historical":
        required_ids = _compute_capped_historical_required(
            xstar_in_catalog,
            per_user_hist,
            all_historical,
            max_per_user=3,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    required_in_catalog = required_ids & catalog_index_set

    if len(required_in_catalog) > target_size + 20:
        raise ValueError(
            f"Cannot attain catalog fraction {fraction}: "
            f"need {len(required_in_catalog)} items "
            f"(for strategy '{strategy}'), "
            f"but target size is {target_size}"
        )

    if len(required_in_catalog) < target_size:
        extra_pool = sorted(catalog_index_set - required_in_catalog)
        n_extra = target_size - len(required_in_catalog)
        selected_ids = required_in_catalog | set(extra_pool[:n_extra])
    else:
        selected_ids = required_in_catalog

    selected_index = [i for i in catalog.index if str(i) in selected_ids]
    new_catalog = catalog.loc[selected_index].copy()
    new_catalog.index = new_catalog.index.astype(str)
    return new_catalog


def subset_catalog_from_transactions(
    catalog: pd.DataFrame,
    fraction: float,
    transactions_dir: str,
    *,
    test_set_size: int,
    user_suffix: str,
    history_extractor: Callable[[dict], List[str]],
    strategy: Literal[
        "xstar", "xstar+historical", "xstar+prefer_historical"
    ] = "xstar+prefer_historical",
) -> pd.DataFrame:
    """Subset a catalog using xstar/historical IDs read from transaction files.

    Convenience wrapper around :func:`subset_catalog_df` that gathers the
    required IDs directly from a dataset's transactions directory, so callers
    don't have to reimplement the file globbing/parsing.

    For each spec index in the test split (indices ``[0, test_set_size)``), it
    reads ``{idx}_items.txt`` (first line, comma-separated) for the xstar IDs
    and ``{idx}{user_suffix}`` (JSON) for the historical IDs (via
    ``history_extractor``).

    Args:
        catalog: Full catalog DataFrame whose index contains item IDs.
        fraction: Target fraction of the full catalog to retain.
        transactions_dir: Directory containing ``{idx}_items.txt`` and
            ``{idx}{user_suffix}`` files.
        test_set_size: Number of held-out test specs (split boundary).
        user_suffix: Filename suffix for the per-user JSON file, e.g.
            ``"_customer.json"`` or ``"_user.json"``.
        history_extractor: Callable mapping a parsed user-info dict to a list
            of historical item ID strings.
        strategy: Item retention strategy forwarded to
            :func:`subset_catalog_df`.

    Returns:
        The subset catalog DataFrame from :func:`subset_catalog_df`.
    """
    idxs = list(range(test_set_size))

    xstar_ids: dict[int, List[str]] = {}
    historical_data_ids: dict[int, List[str]] = {}
    for ix in idxs:
        items_path = os.path.join(transactions_dir, f"{ix}_items.txt")
        user_path = os.path.join(transactions_dir, f"{ix}{user_suffix}")
        if os.path.exists(items_path):
            with open(items_path) as f:
                lines = f.read().splitlines()
            xstar_ids[ix] = (
                [s.strip() for s in lines[0].split(",") if s.strip()] if lines else []
            )
        if os.path.exists(user_path):
            with open(user_path) as f:
                info = json.load(f)
            historical_data_ids[ix] = history_extractor(info)

    return subset_catalog_df(
        catalog, fraction, xstar_ids, historical_data_ids, strategy
    )
