"""Feature revelation tracker for the user simulator.

Manages the SEC (Search / Experience / Credence) feature state for a single
elicitation episode: which features are currently known to the agent, which
are queued for revelation, and how many may be revealed per turn.

Main exports:

- :class:`Feature` — dataclass describing one preference feature.
- :class:`FeatureTracker` — stateful tracker; updated as the conversation progresses.
- :func:`randomly_init_known_features` — helper to draw an initial random known set.
"""

from dataclasses import dataclass
import pandas as pd
from typing import List, Literal, Tuple, Dict, Any
import random

from ...data.dataset import OPEN_TO_ANYTHING
from .message_parser import ClarifyingQuestion, ItemToEval, Explanation
from .feature_utils import (
    normalize_feature_name,
    filter_duplicate_features,
)
from ...utils.misc import print_debug


@dataclass
class Feature:
    column_name: str
    target_values: List[str]
    category: Literal["search", "experience", "credence"]
    known: bool = False


class FeatureTracker:
    def __init__(
        self,
        target_df: pd.DataFrame,
        search_features: List[str],
        experience_features: List[str],
        credence_features: List[str],
        max_features_to_reveal: int = None,
        verbosity: int = 0,
        column_descriptions: Dict[str, str] = None,
        item_name: str = None,
        explanations_can_reveal_non_queued_features: bool = True,
    ):
        """
        Initialize the FeatureTracker with the target dataframe and the list of features.
        If a column is not in the list of search, experience, or credence features, it is considered a hidden feature.

        Args:
            target_df: this should be from ds.simulator_catalog -> subset to xstar ids -> explode tags
            search_features: columns in target_df; this includes columns in the original catalog + tags
            experience_features: columns in target_df; this includes columns in the original catalog + tags
            credence_features: columns in target_df; this includes columns in the original catalog + tags
            max_features_to_reveal: maximum number of features to reveal per item / question
            verbosity: if > 0, print feature lists and settings at initialization
        """
        self._max_features_to_reveal = max_features_to_reveal
        self.credence_queue: List[str] = []  # queue of credence features
        self._column_descriptions = column_descriptions
        self._item_name = item_name
        self._verbosity = verbosity
        self._history: List[Dict[str, Any]] = []
        self._explanations_can_reveal_non_queued_features = (
            explanations_can_reveal_non_queued_features
        )

        # build up the list of features
        self.features = []
        for col_name in target_df.columns:
            if col_name not in (
                search_features + experience_features + credence_features
            ):
                # hidden feature
                continue
            sec_category = (
                "search"
                if col_name in search_features
                else "experience"
                if col_name in experience_features
                else "credence"
            )
            self.features.append(
                Feature(
                    column_name=col_name,
                    target_values=target_df[col_name].tolist(),
                    category=sec_category,
                )
            )

        if verbosity > 0:
            print_debug(
                f"search_features: {search_features}\n"
                f"experience_features: {experience_features}\n"
                f"credence_features: {credence_features}\n"
                f"max_features_to_reveal: {max_features_to_reveal}\n"
                f"total tracked features: {len(self.features)}",
                "FeatureTracker.__init__",
            )

    @property
    def known_features(self) -> List[Feature]:
        return [f for f in self.features if f.known]

    @property
    def unknown_features(self) -> List[Feature]:
        return [f for f in self.features if not f.known]

    @property
    def search_features(self) -> List[Feature]:
        return [f for f in self.features if f.category == "search"]

    @property
    def unknown_search_features(self) -> List[Feature]:
        return [f for f in self.search_features if not f.known]

    @property
    def experience_features(self) -> List[Feature]:
        return [f for f in self.features if f.category == "experience"]

    @property
    def credence_features(self) -> List[Feature]:
        return [f for f in self.features if f.category == "credence"]

    @property
    def reveal_history(self) -> List[Dict[str, Any]]:
        return self._history

    def clear_reveal_history(self) -> None:
        """Clear the feature reveal history (e.g. on simulator reset)."""
        self._history = []

    def clear_credence_queue(self) -> None:
        self.credence_queue = []

    def is_open_to_anything(self, column_name: str) -> bool:
        """True if this known feature's target value is the dataset's NA placeholder 'open to anything'."""
        for f in self.known_features:
            if f.column_name == column_name:
                vals = {
                    str(v).strip().lower()
                    for v in f.target_values
                    if not (v is None or (isinstance(v, float) and pd.isna(v)))
                }
                return vals <= {OPEN_TO_ANYTHING.lower()}
        return False

    def get_value(self, column_name: str) -> str:
        """
        Get the value of a feature.
        """
        for f in self.known_features:
            if f.column_name == column_name:
                return (
                    "Any of the following (indifferent between these, would like nay of them): "
                    + ", ".join([str(v) for v in f.target_values])
                )
        return None

    def reveal_features(
        self,
        column_names: List[str],
        categories: Literal["search", "experience", "credence"],
    ) -> int:
        """
        Reveal a list of features by setting them to known.
        Returns the number of features revealed.
        """
        num_revealed = 0
        for feature in self.unknown_features:
            if categories is not None and feature.category not in categories:
                continue
            if feature.column_name in column_names and self._reveal_feature(
                feature.column_name, source="external_call"
            ):
                num_revealed += 1
        return num_revealed

    def _reveal_feature(self, column_name: str, source: str = "unknown") -> bool:
        """
        Reveal a feature by setting it to known.
        """
        for f in self.unknown_features:
            if f.column_name == column_name:
                if self._verbosity > 0 and source not in ["external_call", "unknown"]:
                    print_debug(
                        f"Revealing feature {column_name} from {source}",
                        "FeatureTracker._reveal_feature",
                    )
                f.known = True
                self._history.append(
                    {
                        "column_name": column_name,
                        "value": f.target_values,
                        "source": source,
                    }
                )
                return True
        return False

    """
    Context for answering questions and giving feedback.
    Returns a string of the form:
    <column_name>: <target_values>
    <column_name>: <target_values>
    ...
    <column_name>: <target_values>
    <column_name>: <target_values>
    ...
    <column_name>: <target_values>
    """

    def get_known_context(
        self, relevant_columns: List[str] = None, drop_na_vals: bool = False
    ) -> str:
        """
        Returns the context for answering questions and giving feedback.
        Filters to known features.
        If relevant_columns is provided, filters to only include known features in the relevant columns.
        """
        known_features = [
            f
            for f in self.known_features
            if relevant_columns is None or f.column_name in relevant_columns
        ]
        key_values = {
            f"{normalize_feature_name(self._column_descriptions.get(f.column_name, f.column_name))}: ": f"{' or '.join([str(v) for v in f.target_values if (not drop_na_vals) or (v != OPEN_TO_ANYTHING)])}"
            for f in known_features
        }
        body = "\n".join(f"{k}{v}" for k, v in key_values.items() if v.strip())
        if body.strip() == "" and not drop_na_vals:
            body = "Open to anything"
        return (
            f"I would like to find a {self._item_name} that matches the following preferences:\n\n"
            + body
        )

    """
    Functions to reveal features.
    """

    def process_question_reveals(
        self, question: ClarifyingQuestion
    ) -> Tuple[int, List[str]]:
        """
        Reveal any features based on the question.
        Returns the number of features revealed and the list of credence features the question pertains to.
        """
        unknown_search_credence_features = [
            f for f in (self.search_features + self.credence_features) if not f.known
        ]
        unknown_search_credence_features_names = [
            f.column_name for f in unknown_search_credence_features
        ]
        if not unknown_search_credence_features:
            return 0, []
        num_revealed = 0
        credence_features_this_question = []
        # Filter the relevant columns for this question to drop redundant,
        # overly granular variants before deciding what to reveal.
        filtered_relevant = filter_duplicate_features(
            [
                c
                for c in question.relevant_columns
                if c in unknown_search_credence_features_names
            ],
            self._column_descriptions or {},
        )
        for feature in unknown_search_credence_features:
            if (
                self._max_features_to_reveal is not None
                and num_revealed >= self._max_features_to_reveal
            ):
                break
            if feature.column_name not in filtered_relevant:
                continue
            if feature.category == "credence":
                credence_features_this_question.append(feature.column_name)
            elif feature.category == "search":
                self._reveal_feature(feature.column_name, source="question")
                num_revealed += 1

        self.credence_queue.extend(credence_features_this_question)

        return num_revealed, credence_features_this_question

    def get_unknown_feature_types(self, relevant_columns: List[str]) -> Dict[str, List[str]]:
        """
        For the given column names, return the unknown features grouped by category.
        Only includes features that are tracked (search/experience/credence) and not yet known.
        """
        result: Dict[str, List[str]] = {"search": [], "experience": [], "credence": []}
        col_set = set(relevant_columns)
        for f in self.features:
            if f.column_name in col_set and not f.known:
                result[f.category].append(f.column_name)
        return result

    def process_feedback_reveals(self, item: ItemToEval) -> Tuple[int, List[str]]:
        """
        Reveal any features based on an item evaluation.
        Returns:
            - the number of features revealed
            - a list of newly revealed experience feature column names for this item
        """
        unknown_search_experience_features = [
            f for f in self.search_features + self.experience_features if not f.known
        ]
        unknown_search_experience_features_names = [
            f.column_name for f in unknown_search_experience_features
        ]
        if not unknown_search_experience_features:
            return 0, []
        num_revealed = 0
        newly_revealed_experience: List[str] = []
        # Filter the item-level relevant columns before revealing anything
        # so we prefer coarser features when both coarse and granular variants
        # have been marked relevant.
        filtered_relevant = filter_duplicate_features(
            [
                c
                for c in item.relevant_columns
                if c in unknown_search_experience_features_names
            ],
            self._column_descriptions or {},
        )

        # Within the filtered relevant set, prioritize features whose target
        # values are NOT open-to-anything before those that are.
        def _is_open_to_anything_feature(feat: Feature) -> bool:
            vals = {
                str(v).strip().lower()
                for v in feat.target_values
                if not (v is None or (isinstance(v, float) and pd.isna(v)))
            }
            if not vals:
                return False
            return vals <= {OPEN_TO_ANYTHING.lower()}

        ordered_unknown = sorted(
            unknown_search_experience_features,
            key=lambda f: _is_open_to_anything_feature(f),
        )

        for feature in ordered_unknown:
            if (
                self._max_features_to_reveal is not None
                and num_revealed >= self._max_features_to_reveal
            ):
                break
            if feature.column_name in filtered_relevant:
                self._reveal_feature(feature.column_name, source="feedback")
                num_revealed += 1
                if feature.category == "experience":
                    newly_revealed_experience.append(feature.column_name)
        return num_revealed, newly_revealed_experience

    def process_credence_reveals(
        self, explanation: Explanation
    ) -> Tuple[int, List[str], List[str]]:
        """
        Reveal any queued credence features based on the assistant message.
        Returns the number of features revealed and the list of credence features the explanation pertains to.
        Returns:
        - number of features revealed
        - list of credence features that we should now answer questions about
        - list of credence features that we should acknowledge but which we don't reveal
        """
        if not self._explanations_can_reveal_non_queued_features:
            if not self.credence_queue:
                return 0, [], []
            if explanation.relevant_column not in self.credence_queue:
                return 0, [], []

        if explanation.target_value_mentioned_or_in_range:
            if explanation.relevant_column in self.credence_queue:
                self.credence_queue.remove(explanation.relevant_column)
            self._reveal_feature(explanation.relevant_column, source="credence")
            return 1, [explanation.relevant_column], []
        elif explanation.relevant_column in self.credence_queue:
            self.credence_queue.remove(explanation.relevant_column)
            return 0, [], [explanation.relevant_column]
        else:
            return 0, [], []


def randomly_init_known_features(
    feature_tracker: FeatureTracker, num_features: int, dropout_rate: float = 0.0
):
    """
    Randomly reveal a subset of the (search) features as known.

    Each unknown search feature is independently kept with probability
    ``1 - dropout_rate``. Only search features are revealed; experience and
    credence features are left unknown. ``num_features`` is currently unused.
    Returns the number of features revealed.
    """
    features_to_reveal = [
        f.column_name
        for f in feature_tracker.unknown_features
        if f.category == "search" and random.random() < 1 - dropout_rate
    ]
    feature_tracker.reveal_features(features_to_reveal, "search")
    return len(features_to_reveal)
