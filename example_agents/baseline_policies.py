"""
Baseline policies for evaluation: Random and Popularity baselines.
These policies don't actually engage in conversation but provide baseline rankings.
"""

from typing import Optional, Tuple, Any, Dict, List, Union
import numpy as np
import pandas as pd
from .policy import InteractionPolicy
from coshop.data.utility import UtilityFunction


class RandomUtilityFunction(UtilityFunction):
    """Utility function that returns random scores for baseline evaluation."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        # Store scores for items we've seen (for consistency)
        self._scores: Dict[str, float] = {}

    def __call__(
        self, items: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Return random scores for the items.

        Args:
            items: List of item strings to score
            return_metadata: If True, return tuple of (scores, metadata)

        Returns:
            List of scores (0-100) or tuple of (scores, metadata)
        """
        assert isinstance(items, list), "Items must be a list of strings"

        scores = []
        metadata_list = []

        for item in items:
            if item not in self._scores:
                # Generate random score in 0-100 range
                self._scores[item] = self.rng.random() * 100.0
            score = self._scores[item]
            scores.append(score)

            if return_metadata:
                metadata_list.append({"random_score": score})

        if return_metadata:
            return scores, metadata_list
        else:
            return scores


class PopularityUtilityFunction(UtilityFunction):
    """Utility function that scores items based on popularity"""

    def __init__(
        self, popularity_df: pd.DataFrame, catalog: pd.DataFrame, representation
    ):
        """
        Initialize popularity utility function.

        Args:
            popularity_df: DataFrame indexed by 'id' with a 'popularity' column
            catalog: Catalog DataFrame with items
            representation: Representation instance for converting text to row
                This is used to find the ID of the item and then look it up in the popularity_df
        """
        self.popularity_df = popularity_df
        self.catalog = catalog
        self.representation = representation
        assert "popularity" in popularity_df.columns, (
            "popularity_df must have a 'popularity' column"
        )
        # Cache max popularity for normalization (0-100 range)
        self._max_popularity = float(popularity_df["popularity"].max())
        if self._max_popularity == 0:
            self._max_popularity = 1.0  # Avoid division by zero

    def __call__(
        self, items: List[str], return_metadata: bool = False
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Return popularity scores for the items.

        Args:
            items: List of item strings to score
            return_metadata: If True, return tuple of (scores, metadata)

        Returns:
            List of scores (0-100) or tuple of (scores, metadata)
        """
        assert isinstance(items, list), "Items must be a list of strings"

        scores = []
        metadata_list = []

        for item in items:
            score = 0.0
            raw_popularity = 0.0
            item_id = None

            # Extract id from the text representation
            item_id = self.representation.str_to_id(item)
            if item_id is None:
                if return_metadata:
                    metadata_list.append(
                        {
                            "warning": f"Invalid item text: {item}",
                            "reason": "invalid_item_text",
                        }
                    )
                continue

            if item_id in self.popularity_df.index:
                raw_popularity = float(self.popularity_df.loc[item_id, "popularity"])
                # Normalize to 0-100 range
                if self._max_popularity is not None and self._max_popularity > 0:
                    score = (raw_popularity / self._max_popularity) * 100.0
                else:
                    score = raw_popularity

            scores.append(score)

            if return_metadata:
                metadata_list.append(
                    {
                        "item_id": item_id,
                        "raw_popularity": raw_popularity,
                        "normalized_score": score,
                    }
                )

        if return_metadata:
            return scores, metadata_list
        else:
            return scores


class RandomBaselinePolicy(InteractionPolicy):
    """Baseline policy that returns random rankings."""

    def __init__(
        self,
        *args,
        spec: Optional[Any] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(spec=spec, *args, **kwargs)
        self.seed = seed
        self.baseline_queries = spec.baseline_queries
        self._utility_function = RandomUtilityFunction(seed=seed)

    def _generate_message(
        self, user_response: Optional[str] = None
    ) -> Tuple[str, int, float, bool]:
        """
        Generate a message. For baseline, we just end immediately.

        Returns:
            Tuple[str, int, float, bool]: (message, token_cost, runtime_cost, wants_to_end_conversation)
        """
        # End conversation immediately
        return "Baseline: Random ranking", 0, 0.0, True

    def get_final_predictions(
        self,
        k: int,
        retrieval_function=None,
        execution_max_per_retrieval: Optional[int] = None,
        execution_max_queries: Optional[int] = None,
        execution_global_max: Optional[int] = None,
        max_retries: int = 3,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Get final predictions: query items and rank using random utility function.

        Returns:
            Tuple[List[str], Dict[str, Any]]: (Top k item IDs in ranked order, metadata)
        """
        if retrieval_function is None:
            raise ValueError("retrieval_function must be provided")

        # If max_execution_items is set but m_test and max_queries are not, split evenly
        if execution_global_max is not None:
            if execution_max_queries is None and execution_max_per_retrieval is None:
                execution_max_per_retrieval = execution_global_max // len(
                    self.baseline_queries
                )
            elif execution_max_queries is not None:
                execution_max_per_retrieval = execution_global_max // execution_max_queries
            elif execution_max_per_retrieval is not None:
                execution_max_queries = execution_global_max // execution_max_per_retrieval

        # Limit baseline queries to max_queries if provided
        queries_to_use = self.baseline_queries
        if execution_max_queries is not None:
            queries_to_use = queries_to_use[:execution_max_queries]

        # Query items using baseline queries and take union
        all_results = []
        for baseline_query in queries_to_use:
            if not baseline_query or not isinstance(baseline_query, str):
                continue
            try:
                query_results = retrieval_function(baseline_query, execution_max_per_retrieval)
                if len(query_results) > 0:
                    all_results.append(query_results)
            except ValueError:
                # Skip this query if ValueError is raised (e.g., max_items limit exceeded)
                continue

        if len(all_results) == 0:
            return [], {"search_queries": queries_to_use, "seen_ids": []}

        # Union all results by concatenating and dropping duplicates by id
        query_results = pd.concat(all_results, ignore_index=True)
        query_results = query_results.drop_duplicates(subset=["id"], keep="first")

        if len(query_results) == 0:
            return [], {"search_queries": queries_to_use, "seen_ids": []}

        # Score items using utility function
        item_texts = query_results["text"].tolist()
        scores = self._utility_function(item_texts)

        # Create list of (item_id, score) tuples and sort by score
        item_ids = query_results["id"].tolist()
        scored_items = list(zip(item_ids, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Return top k item IDs
        top_k_ids = [item_id for item_id, _ in scored_items[:k]]
        # seen_ids = all items retrieved (not just top k)
        seen_ids = item_ids
        return top_k_ids, {"search_queries": queries_to_use, "seen_ids": seen_ids}


class PopularityBaselinePolicy(InteractionPolicy):
    """Baseline policy that returns popularity-based rankings."""

    def __init__(
        self,
        *args,
        spec: Optional[Any] = None,
        popularity_df: Optional[pd.DataFrame] = None,
        catalog: Optional[pd.DataFrame] = None,
        representation=None,
        **kwargs,
    ):
        super().__init__(spec=spec, *args, **kwargs)
        self.popularity_df = popularity_df
        self.catalog = catalog
        self.representation = representation
        self.baseline_queries = spec.baseline_queries
        if (
            popularity_df is not None
            and catalog is not None
            and representation is not None
        ):
            self._utility_function = PopularityUtilityFunction(
                popularity_df=popularity_df,
                catalog=catalog,
                representation=representation,
            )
        else:
            # Fallback to random if data not available
            self._utility_function = RandomUtilityFunction()

    def _generate_message(
        self, user_response: Optional[str] = None
    ) -> Tuple[str, int, float, bool]:
        """
        Generate a message. For baseline, we just end immediately.

        Returns:
            Tuple[str, int, float, bool]: (message, token_cost, runtime_cost, wants_to_end_conversation)
        """
        # End conversation immediately
        return "Baseline: Popularity-based ranking", 0, 0.0, True

    def get_final_predictions(
        self,
        k: int,
        retrieval_function=None,
        execution_max_per_retrieval: Optional[int] = None,
        execution_max_queries: Optional[int] = None,
        execution_global_max: Optional[int] = None,
        max_retries: int = 3,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Get final predictions: query items and rank using popularity utility function.

        Returns:
            Tuple[List[str], Dict[str, Any]]: (Top k item IDs in ranked order, metadata)
        """
        if retrieval_function is None:
            raise ValueError("retrieval_function must be provided")

        # If max_execution_items is set but m_test and max_queries are not, split evenly
        if execution_global_max is not None:
            if execution_max_queries is None and execution_max_per_retrieval is None:
                execution_max_per_retrieval = execution_global_max // len(
                    self.baseline_queries
                )
            elif execution_max_queries is not None:
                execution_max_per_retrieval = execution_global_max // execution_max_queries
            elif execution_max_per_retrieval is not None:
                execution_max_queries = execution_global_max // execution_max_per_retrieval

        # Limit baseline queries to max_queries if provided
        queries_to_use = self.baseline_queries
        if execution_max_queries is not None:
            queries_to_use = queries_to_use[:execution_max_queries]

        # Query items using baseline queries and take union
        all_results = []
        for baseline_query in queries_to_use:
            if not baseline_query or not isinstance(baseline_query, str):
                continue
            try:
                query_results = retrieval_function(baseline_query, execution_max_per_retrieval)
                if len(query_results) > 0:
                    all_results.append(query_results)
            except ValueError:
                # Skip this query if ValueError is raised (e.g., max_items limit exceeded)
                continue

        if len(all_results) == 0:
            return [], {"search_queries": queries_to_use, "seen_ids": []}

        # Union all results by concatenating and dropping duplicates by id
        query_results = pd.concat(all_results, ignore_index=True)
        query_results = query_results.drop_duplicates(subset=["id"], keep="first")

        if len(query_results) == 0:
            return [], {"search_queries": queries_to_use, "seen_ids": []}

        # Score items using utility function
        item_texts = query_results["text"].tolist()
        scores = self._utility_function(item_texts)

        # Create list of (item_id, score) tuples and sort by score
        item_ids = query_results["id"].tolist()
        scored_items = list(zip(item_ids, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Return top k item IDs
        top_k_ids = [item_id for item_id, _ in scored_items[:k]]
        # seen_ids = all items retrieved (not just top k)
        seen_ids = item_ids
        return top_k_ids, {"search_queries": queries_to_use, "seen_ids": seen_ids}
