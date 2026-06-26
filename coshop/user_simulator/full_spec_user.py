"""
FullSpecificationUser: ExpertUser with all features known from the start.

The single "user" message is the spec's preference text for the configured
``z_condition`` (z0 / zs / zse / zstar; defaults to zstar). No LLM is used for
responses. Regardless of which z-variant is surfaced, the simulator reveals all
features for that condition and can rank items using the same logic as
ExpertUser (perfect information for ranking).
"""

from __future__ import annotations

from typing import Tuple, Optional

from .expert_user import ExpertUser


class FullSpecificationUser(ExpertUser):
    """
    Oracle user simulator: inherits from ExpertUser with all features for the
    configured z-condition revealed. The first (and only) "user" message is the
    spec's z-variant text selected by ``z_condition`` (z0 / zs / zse / zstar).
    Ranking uses the same rank_items / rank_items_initial_state as ExpertUser,
    with full feature knowledge.
    """

    def __init__(
        self,
        spec,
        dataset,
        z_condition: str = "zstar",
        *args,
        **kwargs,
    ):
        if dataset is None:
            raise ValueError("dataset is required for FullSpecificationUser")

        # Store which z-variant to surface as the first (and only) user message.
        # Ranking logic still uses full feature knowledge.
        self._z_condition = z_condition
        if self._z_condition == "z0":
            self.z_text = spec.z0
        elif self._z_condition == "zs":
            self.z_text = spec.zs
        elif self._z_condition == "zse":
            self.z_text = spec.zse
        elif self._z_condition == "zstar":
            self.z_text = spec.zstar
        else:
            raise ValueError(f"Invalid z-condition: {self._z_condition}")

        super().__init__(
            spec=spec,
            dataset=dataset,
            *args,
            **kwargs,
        )

        # Align initial-state cache with "all known" for rank_items_initial_state
        self._initial_known_feature_names = [
            f.column_name for f in self.feature_tracker.known_features
        ]
        self._initial_z_context = self.feature_tracker.get_known_context()

        # Reveal features according to the requested z-condition so that
        # get_known_context() matches z0 / zs / zse / zstar.
        all_columns = list(self.features_star.columns)
        if self._z_condition == "z0":
            # Only the initial_known_features from the parent (ExpertUser).
            pass
        elif self._z_condition == "zs":
            # Reveal all remaining search features.
            self.feature_tracker.reveal_features(all_columns, categories=["search"])
        elif self._z_condition == "zse":
            # Reveal all remaining search + experience features.
            self.feature_tracker.reveal_features(
                all_columns, categories=["search", "experience"]
            )
        else:
            # Default: reveal all features (full zstar-equivalent).
            self.feature_tracker.reveal_features(
                all_columns, categories=["search", "experience", "credence"]
            )

    def _generate_response(
        self, assistant_msg: Optional[str]
    ) -> Tuple[str, int, float]:
        """
        Return a single-turn preference description z, with no LLM cost.
        The specific z-variant (z0, zs, zse, or zstar) is controlled by the
        z_condition parameter; when not available on the spec, we fall back
        to the full known context (zstar-equivalent).
        """
        z_text = (
            self.z_text
            # + "\n\nWhen searching for items for me, try to incorporate a lot of the information above in your search query (direct quotes are best). Try to be exhaustive in your search and use the full budget."
        )
        return z_text, 0, 0.0
