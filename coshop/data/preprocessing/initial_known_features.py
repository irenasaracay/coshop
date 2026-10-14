"""
Initial known features (S1) by incremental search recall.

For each user, start from the always-included search feature (product type for
H&M, genres for MovieLens/Goodreads) and repeatedly:

1. ask an LM (LOTUS sem_map) to turn the known attributes of x* into one
   search query,
2. run the query through vector search and check whether x* is in the top K,
3. if not, add one random search feature (from the user's SEC split) that has
   a value on x*.

The saved set is the last one *before* x* became retrievable, so every row has
``recall == 0`` by construction. Requires a gpt-oss-120b vLLM server and the
vector-search server. The SEC split must exist for every user first
(:mod:`~coshop.data.preprocessing.compute_sec_split`).

Usage:
    python -m coshop.data.preprocessing.initial_known_features --dataset hm \\
        --vllm-url http://localhost:3005 --vector-search-url http://localhost:3004
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ...tools.retrieval.retrieval import get_retrieval_fn
from ...utils.lotus import configure_lotus, sem_map_with_retries
from ...utils.misc import check_na
from ..dataset import OPEN_TO_ANYTHING
from ._common import (
    add_common_args,
    load_dataset,
    output_path,
    resolve_datasets,
    todo_indexes,
    user_indexes,
)

MODEL_NAME = "openai/gpt-oss-120b"
K = 250
# Extra attempts for rows whose query came back empty (LM error or empty output).
QUERY_RETRIES = 3

# Features always included in the known set at bootstrap.
ALWAYS_INCLUDED_FEATURES = {
    "hm": ["product_type_name"],
    "movielens": ["genres"],
    "goodreads": ["genres"],
}

# LOTUS instruction, per dataset.
INSTRUCTION_TEMPLATES = {
    "hm": (
        "You are helping to find a clothing item in a catalog. "
        "Given the following known attributes of the item the user wants, output a single search query string "
        "that could be used to search the catalog (e.g. for a search box). "
        "Output ONLY the search query and nothing else, or a JSON object with a single key \"search_query\" and the query as value.\n\n"
        "{lm_input}"
    ),
    "movielens": (
        "You are helping to find a movie in a catalog. "
        "Given the following known attributes of the movie the user wants, output a single search query string "
        "that could be used to search the catalog (e.g. for a search box). "
        "Output ONLY the search query and nothing else, or a JSON object with a single key \"search_query\" and the query as value.\n\n"
        "{lm_input}"
    ),
    "goodreads": (
        "You are helping to find a book in a catalog. "
        "Given the following known attributes of the book the user wants, output a single search query string "
        "that could be used to search the catalog (e.g. for a search box). "
        "Output ONLY the search query and nothing else, or a JSON object with a single key \"search_query\" and the query as value.\n\n"
        "{lm_input}"
    ),
}

CSV_COLUMNS = ["user_idx", "n_features_used", "query", "feature_names", "recall"]


def compute_recall_at_k(
    retrieved_ids: List[str], xstar_ids: List[str], k: int
) -> int:
    """Return 1 if any xstar is in top k retrieved, else 0."""
    if not retrieved_ids or not xstar_ids:
        return 0
    retrieved_set = set(str(i) for i in retrieved_ids[:k])
    xstar_set = set(str(i) for i in xstar_ids)
    return 1 if retrieved_set.intersection(xstar_set) else 0


def build_search_only_pool(
    target_row: pd.Series,
    search_column_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Build the list of search features available for this target item.

    Returns:
        List of feature dicts: {"type": "column", "name": col, "value": val}.
        Only columns with a non-missing, non-OPEN_TO_ANYTHING value are included.
    """
    pool: List[Dict[str, Any]] = []
    for col in search_column_names:
        if col not in target_row.index:
            continue
        val = target_row[col]
        # Skip missing/empty values (including lists with any NA) and simulator placeholder ("open to anything")
        if check_na(val) or (
            isinstance(val, (str, bytes)) and not str(val).strip()
        ):
            continue
        if (
            isinstance(val, (str, bytes))
            and (
                str(val).strip() == OPEN_TO_ANYTHING
                or str(val).strip().lower() == "open to anything"
            )
        ):
            continue
        pool.append({"type": "column", "name": col, "value": val})
    return pool


def feature_names_from_known(known_features: List[Dict[str, Any]]) -> List[str]:
    """Return list of feature names for the known set: column name for columns, 'tag:<value>' for tags."""
    names = []
    for f in known_features:
        if f["type"] == "column":
            names.append(f["name"])
        else:
            names.append(f"tag:{f['value']}")
    return names


def format_known_features_for_lm(known_features: List[Dict[str, Any]]) -> str:
    """Format known features as text for LM input (e.g. 'Known attributes: col: val; tag: t')."""
    parts = []
    for f in known_features:
        if f["type"] == "column":
            parts.append(f"{f['name']}: {f['value']}")
        else:
            parts.append(f"tag: {f['value']}")
    return "Known attributes: " + "; ".join(parts) if parts else "Known attributes: (none)"


def parse_search_query_from_lm(raw: Any) -> str:
    """
    Extract search query string from LOTUS sem_map _map output.
    Handles raw query text or JSON like {"search_query": "..."}.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Try JSON first
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict) and "search_query" in obj:
                return str(obj["search_query"]).strip()
        except Exception:
            pass
    return s


def _feature_key(f: Dict[str, Any]) -> Tuple[str, str]:
    """Unique key for a feature for set/deduplication."""
    if f["type"] == "column":
        return ("column", f"{f['name']}\0{f['value']}")
    return ("tag", f["value"])


def add_random_feature(
    known_features: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Return one feature from pool not already in known_features, or None."""
    known_keys = {_feature_key(f) for f in known_features}
    candidates = [f for f in pool if _feature_key(f) not in known_keys]
    if not candidates:
        return None
    return rng.choice(candidates)


def _result_row(user_idx: int, known: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    return {
        "user_idx": user_idx,
        "n_features_used": len(known),
        "query": query,
        "feature_names": feature_names_from_known(known),
        # By construction, the saved set itself has not yet recalled.
        "recall": 0,
    }


def _generate_queries(
    lm_inputs: List[str], instruction_template: str, chunk_size: int
) -> List[str]:
    """One search query per LM input, in chunks; "" where the LM call failed."""
    queries: List[str] = []
    for start in range(0, len(lm_inputs), chunk_size):
        chunk = pd.DataFrame({"lm_input": lm_inputs[start : start + chunk_size]})
        try:
            lm_out = sem_map_with_retries(chunk, instruction_template)
            raws = lm_out["_map"].tolist() if "_map" in lm_out.columns else [None] * len(chunk)
        except Exception as e:
            print(f"[incremental] sem_map failed for {len(chunk)} rows: {e}")
            raws = [None] * len(chunk)
        queries.extend(parse_search_query_from_lm(raw) for raw in raws)
    return queries


def run_incremental_recall_loop(
    users_df: pd.DataFrame,
    query_function: Callable[[Optional[str], int], pd.DataFrame],
    instruction_template: str,
    chunk_size: int = 50,
    max_iter: int = 20,
    k: int = K,
    seed: int = 0,
    retrieval_workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run the incremental recall loop: sem_map to get query, run query, compute recall, drop recalled, add feature.

    LOTUS must already be configured. Each user draws features from its own
    ``random.Random(f"{seed}:{user_idx}")``, so a user's result does not depend
    on which other users are in the batch.

    If a user's query can't be generated (after retries) or retrieval fails,
    that round is redone in the next iteration instead of being counted as a
    non-recalling round, so failures never add a feature.

    Args:
        users_df: DataFrame with columns user_idx, xstar_ids, known_features, pool.
            known_features and pool are lists of feature dicts (format from build_search_only_pool).
        query_function: Callable(q, m) -> DataFrame with "id" column
        instruction_template: LOTUS instruction string; must contain "{lm_input}".
        chunk_size: Rows per LOTUS sem_map batch.
        max_iter: Max iterations (add-feature steps) per user.
        k: Recall at k.
        seed: Base random seed for adding features.
        retrieval_workers: Number of retrieval calls to run concurrently in each
            round. Calls are independent, so this changes speed, not results.

    Returns:
        List of result dicts: user_idx, n_features_used, query, feature_names, recall (always 0).
    """
    rngs = {uid: random.Random(f"{seed}:{uid}") for uid in users_df["user_idx"]}
    results: List[Dict[str, Any]] = []
    df = users_df.copy()

    # Track the previous round (features + query) so we can save the set *before* recall.
    df["prev_known_features"] = [[] for _ in range(len(df))]
    df["prev_query"] = ["" for _ in range(len(df))]

    iteration = 0
    while len(df) > 0 and iteration < max_iter:
        iteration += 1
        print(f"[incremental] iteration={iteration}, active_users={len(df)}")
        df["lm_input"] = df["known_features"].apply(format_known_features_for_lm)

        # Generate one query per row. Rows whose LM call failed or came back empty
        # are retried here, so a failed call is not mistaken for a non-recalling query.
        queries = _generate_queries(df["lm_input"].tolist(), instruction_template, chunk_size)
        for _ in range(QUERY_RETRIES):
            empty = [i for i, q in enumerate(queries) if not q]
            if not empty:
                break
            print(f"[incremental] retrying {len(empty)} empty queries")
            retried = _generate_queries(
                [df["lm_input"].iloc[i] for i in empty], instruction_template, chunk_size
            )
            for i, q in zip(empty, retried):
                queries[i] = q
        df["_query"] = queries

        # Compute recall for each row; None if there is no query or retrieval failed.
        def recall_for(row: pd.Series) -> Optional[int]:
            if not row["_query"]:
                return None
            try:
                result_df = query_function(row["_query"], k)
            except Exception as e:
                print(f"[incremental] retrieval failed for user {row['user_idx']}: {e}")
                return None
            retrieved = result_df["id"].astype(str).tolist() if "id" in result_df.columns else []
            return compute_recall_at_k(retrieved, row["xstar_ids"], k)

        rows = [row for _, row in df.iterrows()]
        if retrieval_workers > 1:
            with ThreadPoolExecutor(max_workers=retrieval_workers) as pool:
                recalls = list(pool.map(recall_for, rows))
        else:
            recalls = [recall_for(row) for row in rows]
        df["_failed"] = [r is None for r in recalls]
        df["_recall"] = [0 if r is None else r for r in recalls]
        if df["_failed"].any():
            print(f"[incremental] {int(df['_failed'].sum())} users failed this round; redoing it next iteration")

        # Record and drop recalled rows: save the round *before* recall (prev_known, prev_query).
        for _, row in df[df["_recall"] == 1].iterrows():
            prev_known = row["prev_known_features"] or row["known_features"] or []
            prev_query_str = row["prev_query"] or row["_query"] or ""
            results.append(_result_row(row["user_idx"], prev_known, prev_query_str))
        df = df[df["_recall"] != 1].copy()
        if len(df) == 0:
            break

        # Add one random feature to each remaining row; drop rows with no pool left.
        # Failed rows are carried over unchanged, so they redo this round.
        rows_updated: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            if row["_failed"]:
                rows_updated.append({
                    "user_idx": row["user_idx"],
                    "xstar_ids": row["xstar_ids"],
                    "known_features": row["known_features"],
                    "pool": row["pool"],
                    "prev_known_features": row["prev_known_features"],
                    "prev_query": row["prev_query"],
                })
                continue
            # This non-recalling round becomes the new "previous" round.
            prev_known = list(row["known_features"])
            prev_query_str = row["_query"]
            extra = add_random_feature(
                row["known_features"], row["pool"], rngs[row["user_idx"]]
            )
            if extra is None:
                # No more features to add; record this last non-recalling round.
                results.append(_result_row(row["user_idx"], prev_known, prev_query_str))
            else:
                rows_updated.append({
                    "user_idx": row["user_idx"],
                    "xstar_ids": row["xstar_ids"],
                    "known_features": row["known_features"] + [extra],
                    "pool": row["pool"],
                    "prev_known_features": prev_known,
                    "prev_query": prev_query_str,
                })

        # Exhausted rows were recorded above; keep only rows that got a new feature
        # so the max_iter block below doesn't record them a second time.
        df = pd.DataFrame(rows_updated)
        if df.empty:
            break

    # Any remaining rows (max_iter reached) get recorded with recall 0 using the last non-recalling round.
    for _, row in df.iterrows():
        prev_known = row.get("prev_known_features") or row.get("known_features") or []
        prev_query_str = row.get("prev_query") or row.get("_query") or ""
        results.append(_result_row(row["user_idx"], prev_known, prev_query_str))

    return results


def build_user_row(spec: Any, always_included: List[str]) -> Optional[Dict[str, Any]]:
    """Initial loop state for one user, or None if it has no search features."""
    target_row = spec.xstar_simulator_view.iloc[0]
    pool = build_search_only_pool(target_row, spec.sec_split["search"])
    if not pool:
        return None
    known: List[Dict[str, Any]] = []
    seen = set()
    for f in pool:
        key = (f["type"], f.get("name", f.get("value")))
        if f["type"] == "column" and f["name"] in always_included and key not in seen:
            seen.add(key)
            known.append(f)
    return {
        "user_idx": int(spec.index),
        "xstar_ids": spec.xstar,
        "known_features": known,
        "pool": pool,
    }


def read_csv_rows(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.read_csv(path)


def write_csv_rows(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.sort_values("user_idx")[CSV_COLUMNS].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate initial known features (S1) by incremental search recall."
    )
    add_common_args(parser)
    parser.add_argument("--vllm-url", required=True, help="gpt-oss-120b vLLM base URL.")
    parser.add_argument(
        "--vector-search-url", required=True, help="Vector search API server URL."
    )
    parser.add_argument("--max-iter", type=int, default=20, help="Max features to add per user.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Rows per LOTUS sem_map batch.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for sampling features.")
    parser.add_argument(
        "--retrieval-workers",
        type=int,
        default=16,
        help="Concurrent retrieval (vector search + LLM prefilter) calls per round (default 16).",
    )
    parser.add_argument(
        "--users-per-batch",
        type=int,
        default=100,
        help="Users run together before results are written to disk (default 100).",
    )
    args = parser.parse_args()

    configure_lotus(
        MODEL_NAME,
        {"model_provider": "vllm", "vllm_api_url": args.vllm_url, "temperature": 0.0},
        max_tokens=4096,
    )

    for name in resolve_datasets(args):
        ds = load_dataset(name, args)
        out = output_path(ds, "initial_known_features_extended", args)
        existing_df = read_csv_rows(out)
        existing = {str(i): None for i in existing_df["user_idx"]}
        todo = todo_indexes(user_indexes(ds, args), existing, args, out)
        if not todo:
            continue

        retrieval_fn = get_retrieval_fn(
            "VectorSearch",
            catalog=ds.catalog,
            representation=ds.representation,
            retrieval_representation=ds.representation_unrestricted,
            dataset_name=name,
            version=ds.version,
            vector_search_api_url=args.vector_search_url,
            prefilter=True,
            eval_expression_model_name=MODEL_NAME,
            model_provider="vllm",
            vllm_api_url=args.vllm_url,
        )

        for start in range(0, len(todo), args.users_per_batch):
            batch = todo[start : start + args.users_per_batch]
            rows = []
            for ix in batch:
                row = build_user_row(ds[ix], ALWAYS_INCLUDED_FEATURES[name])
                if row is None:
                    print(f"{name}_{ix}: no search features with a value on x*; skipped")
                    continue
                rows.append(row)
            if not rows:
                continue
            print(f"{name}: users {batch[0]}-{batch[-1]} ({len(rows)} with a feature pool), k={K}")
            results = run_incremental_recall_loop(
                pd.DataFrame(rows),
                retrieval_fn,
                INSTRUCTION_TEMPLATES[name],
                chunk_size=args.chunk_size,
                max_iter=args.max_iter,
                k=K,
                seed=args.seed,
                retrieval_workers=args.retrieval_workers,
            )
            new_df = pd.DataFrame(results)
            existing_df = existing_df[~existing_df["user_idx"].isin(new_df["user_idx"])]
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            write_csv_rows(out, existing_df)
            print(f"Saved {len(new_df)} users to {out}")


if __name__ == "__main__":
    main()
