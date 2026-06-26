"""
Structured item feedback: build (item, feature) rows, run semantic match via LOTUS,
then generate per-item feedback (2 sentences) with optional parallel LOTUS call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lotus
import pandas as pd

from .feature_tracker import FeatureTracker
from .feature_utils import normalize_feature_name
from .message_parser import ItemToEval
from ...utils.lotus import configure_lotus, sem_map_with_retries
from ...utils.misc import print_debug

# Type for catalog row / representation (avoid heavy dataset imports)
Catalog = Any
Representation = Any


# Prompt for generating 2-sentence feedback per item (placeholders filled from row).
FEEDBACK_PROMPT = """You are the SHOPPER (customer) evaluating an item the agent recommended.

My current preferences (what I'm looking for):
{z}

Item ID: {item_id}

Description from the agent:
{raw_description}

=====================

How this item compares to my requirements:
{comparison_section}

Features I care about that the agent didn't clarify for this item:
{missing_section}

New preferences I just realized matter (and how this item fits):
{newly_revealed_section}

=====================

Write exactly 2 sentences of brief, natural feedback on how the item matches or doesn't match your requirements. YOU DO NOT CARE ABOUT ANY FEATURES OTHER THAN THE REQUIREMENTS ABOVE. Keep it conversational, not formal, like a text message. You do NOT need to list every preference that the item satisfies — at most, mention one thing that looks good. Clearly point out the ways the item fails your requirements. If any mismatches are because you just realized you care about something new, say that explicitly (e.g., \"Oh, I actually do care about color and I want it to be blue, so this red one doesn't quite fit\"). For any feature that is missing or unclear, especially newly realized ones, phrase it as an uncertainty statement to the agent (e.g., \"I'm not sure if it has adjustable shoulder straps, which I'd like.\"). Write in the FIRST PERSON as the shopper. Paraphrase, don't quote the preferred values."""


PERFECT_MATCH_FEEDBACK = "So far this item seems to match everything I'm looking for. Is there anything else I should know about it? If not we can move on to the research stage."
ALSO_PERFECT_MATCH_FEEDBACK = "This also looks like it matches everything I'm looking for. Is there anything else I should know about it? If not we can move on to the research stage."


class ItemComparer:
    """
    Builds a structured (item, feature) view, runs semantic match via LOTUS,
    then generates per-item feedback (2 sentences), with an option to batch feedback via LOTUS.

    Note: "perfect" here means that all presented feature columns semantically pass the user's
    requirements for this item. It does NOT mean an exact match to the user's ideal x*,
    which is handled separately in `expert_user.py`.
    """

    def __init__(
        self,
        feature_tracker: FeatureTracker,
        true_features: Dict[str, str],
        model_name: str = "gpt-5-nano",
        model_kwargs: Optional[dict] = None,
        verbosity: int = 0,
        *,
        catalog: Optional[Catalog] = None,
        representation: Optional[Representation] = None,
        hint_missing_features: Union[bool, int] = False,
        max_more_info_requests: int = 1,
        use_actual_item_values: bool = False,
    ):
        self.feature_tracker = feature_tracker
        self._true_features = true_features or {}
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs.setdefault("temperature", 0.0)
        self.verbosity = verbosity
        self._catalog = catalog
        self._representation = representation
        self._hint_missing_features: Union[bool, int] = hint_missing_features
        self._use_actual_item_values: bool = use_actual_item_values
        self._lm = configure_lotus(model_name, self.model_kwargs)
        self._feedback_prompt_history: List[Dict[str, Any]] = []
        # Track how many times we've shown PERFECT_MATCH_FEEDBACK per item_id across messages.
        self._perfect_match_counts: Dict[str, int] = {}
        self._max_more_info_requests = max_more_info_requests

    @property
    def feedback_prompt_history(self) -> List[Dict[str, Any]]:
        """History of (item_id, prompt, response) for each 2-sentence feedback prompt shown to the simulator."""
        return self._feedback_prompt_history

    def clear_feedback_prompt_history(self) -> None:
        """Clear the feedback prompt history (e.g. on simulator reset)."""
        self._feedback_prompt_history = []

    def _get_item_description(self, item: ItemToEval) -> str:
        """
        Build the textual description of an item for feedback prompts.
        When use_actual_item_values is True, we ignore the agent's description.
        """
        if self._use_actual_item_values:
            return ""
        return (
            item.raw_description.strip()
            if item.raw_description
            else "(No description provided.)"
        )

    def _gather_by_item(
        self,
        items_to_evaluate: List[ItemToEval],
        new_reveals_by_item_id: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build per-item comparison data directly from FeatureValue.match fields
        (no LOTUS semantic match step needed — match was already computed in MessageParser).
        """
        known_feature_names = [
            f.column_name for f in self.feature_tracker.known_features
        ]
        target_value_map = {
            f.column_name: " or ".join(str(v) for v in f.target_values)
            for f in self.feature_tracker.known_features
        }
        new_reveals_by_item_id = new_reveals_by_item_id or {}
        per_item: List[Dict[str, Any]] = []
        for item in items_to_evaluate:
            iid = str(item.id)
            comparison_lines: List[str] = []
            mentioned: set = set()
            failed_feature_labels: List[str] = []
            newly_revealed_failed_labels: List[str] = []
            newly_revealed_missing_labels: List[str] = []
            newly_revealed_cols = new_reveals_by_item_id.get(iid, [])
            for fv in item.features:
                if fv.column_name not in known_feature_names:
                    continue
                if fv.match is None:
                    continue
                mentioned.add(fv.column_name)
                label = self._true_features.get(
                    fv.column_name, normalize_feature_name(fv.column_name)
                )
                preferred = target_value_map.get(fv.column_name, "")
                pref_str = f"; preferred value: {preferred}" if preferred else ""
                status = "matches my requirement" if fv.match else "fails to match my requirement"
                comparison_lines.append(f"- feature: {label}{pref_str} ({status})")
                if not fv.match:
                    label_with_pref = f"{label} (I want: {preferred})" if preferred else label
                    failed_feature_labels.append(label_with_pref)
                    if fv.column_name in newly_revealed_cols:
                        newly_revealed_failed_labels.append(label_with_pref)
            missing = []
            for f in known_feature_names:
                if f not in mentioned and not self.feature_tracker.is_open_to_anything(f):
                    lbl = normalize_feature_name(self._true_features.get(f, f))
                    preferred = target_value_map.get(f, "")
                    entry = f"{lbl} (I want: {preferred})" if preferred else lbl
                    missing.append(entry)
            # Newly revealed preferences that are not mentioned at all in the item
            for col in newly_revealed_cols:
                lbl = normalize_feature_name(self._true_features.get(col, col))
                preferred = target_value_map.get(col, "")
                entry = f"{lbl} (I want: {preferred})" if preferred else lbl
                if entry in missing and entry not in newly_revealed_missing_labels:
                    newly_revealed_missing_labels.append(entry)
            all_pass = len(comparison_lines) > 0 and all(
                fv.match
                for fv in item.features
                if fv.column_name in known_feature_names and fv.match is not None
            )
            per_item.append(
                {
                    "item_id": iid,
                    "raw_description": self._get_item_description(item),
                    "comparison_lines": comparison_lines,
                    "missing_features": missing,
                    "all_pass": all_pass,
                    "failed_feature_labels": failed_feature_labels,
                    "newly_revealed_failed_labels": newly_revealed_failed_labels,
                    "newly_revealed_missing_labels": newly_revealed_missing_labels,
                }
            )
        return per_item

    def _missing_to_hint(self, rec: Dict[str, Any]) -> List[str]:
        """Return the list of missing features we will actually show. 0/False → none; int k → first k; True → all."""
        if not self._hint_missing_features:
            return []
        missing = rec.get("missing_features", [])
        if isinstance(self._hint_missing_features, int):
            return missing[: self._hint_missing_features]
        return missing

    def _build_feedback_prompt(self, rec: Dict[str, Any], z: str) -> str:
        """Build the prompt text for one item for the feedback sem_map. z is current known preferences context."""
        hinted = self._missing_to_hint(rec)
        comparison_section = (
            "\n".join(rec["comparison_lines"])
            if rec["comparison_lines"]
            else "None mentioned."
        )
        if hinted:
            missing_section = "\n".join(f"- {m}" for m in hinted)
        else:
            missing_section = "(read what the agent said)"

        newly_failed = rec.get("newly_revealed_failed_labels", [])
        newly_missing = rec.get("newly_revealed_missing_labels", [])
        newly_lines: List[str] = []
        for label in newly_failed:
            newly_lines.append(
                f"- I just realized I care about {label}, and this item doesn't really fit that."
            )
        for label in newly_missing:
            newly_lines.append(
                f"- I just realized I care about {label}, but I can't tell from the description whether this item has it."
            )
        if newly_lines:
            newly_revealed_section = "\n".join(newly_lines)
        else:
            newly_revealed_section = "(none this turn)"

        return FEEDBACK_PROMPT.format(
            z=z,
            item_id=rec["item_id"],
            raw_description=rec["raw_description"],
            comparison_section=comparison_section,
            missing_section=missing_section,
            newly_revealed_section=newly_revealed_section,
        )

    def compute_feedback(
        self,
        items_to_evaluate: List[ItemToEval],
        feedback_fn: Optional[Callable[[str], Tuple[str, int, float]]] = None,
        new_reveals_by_item_id: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Dict[str, str], int, float]:
        """
        Compute structured feedback for each item.

        Args:
            items_to_evaluate: list of ItemToEval from the message parser.
            feedback_fn: optional callable (prompt: str) -> (feedback_text: str, token_cost: int, runtime: float).
                If None, uses LOTUS sem_map to batch-generate feedback.

        Returns:
            (id_to_feedback, total_token_cost, total_runtime_cost).
        """
        id_to_feedback: Dict[str, str] = {}
        total_token_cost = 0
        total_runtime_cost = 0.0

        # Filter to valid items we can describe
        items_to_process: List[ItemToEval] = []
        for item in items_to_evaluate:
            if not item.valid_in_catalog:
                id_to_feedback[item.id] = (
                    "This doesn't seem to actually exist in the catalog."
                )
                continue
            if not (item.raw_description or item.features):
                id_to_feedback[item.id] = "I don't know what features this item has."
                continue
            items_to_process.append(item)

        if not items_to_process:
            return id_to_feedback, total_token_cost, total_runtime_cost

        # 1) Gather per-item comparison data from pre-computed match fields
        per_item = self._gather_by_item(
            items_to_process, new_reveals_by_item_id=new_reveals_by_item_id
        )

        # 4) Per-item feedback: perfect-match hardcode only when nothing is missing to ask about
        perfect_recs: List[Dict[str, Any]] = []
        for r in per_item:
            # Perfect if everything mentioned passed and we have at least one comparison line.
            # When hinting missing features, also require that there are no missing features to show.
            if r["all_pass"] and r["comparison_lines"] and not self._missing_to_hint(r):
                perfect_recs.append(r)

        # Apply PERFECT_MATCH_FEEDBACK up to a capped number of times per item_id.
        # Among multiple perfect items in the same message, the first uses the standard
        # PERFECT_MATCH_FEEDBACK wording and subsequent ones use ALSO_PERFECT_MATCH_FEEDBACK.
        perfect_ids_used: List[str] = []
        first_perfect_in_message = True
        for rec in perfect_recs:
            iid = rec["item_id"]
            count = self._perfect_match_counts.get(iid, 0)
            if count >= self._max_more_info_requests:
                # We've already shown PERFECT_MATCH_FEEDBACK max times for this item; use a short, non-repetitive affirmation.
                feedback_text = "Okay this seems great."
                id_to_feedback[iid] = feedback_text
                perfect_ids_used.append(iid)
                self._feedback_prompt_history.append(
                    {
                        "item_id": iid,
                        "prompt": "(Perfect match; max more-info requests reached; no prompt was shown to the model.)",
                        "response": feedback_text,
                        "missing_features": rec.get("missing_features", []),
                    }
                )
                continue
            self._perfect_match_counts[iid] = count + 1
            feedback_text = (
                PERFECT_MATCH_FEEDBACK
                if first_perfect_in_message
                else ALSO_PERFECT_MATCH_FEEDBACK
            )
            first_perfect_in_message = False
            id_to_feedback[iid] = feedback_text
            perfect_ids_used.append(iid)
            # Log so history has one entry per item (no LLM prompt was shown for perfect match)
            self._feedback_prompt_history.append(
                {
                    "item_id": iid,
                    "prompt": "(Perfect match; no prompt was shown to the model.)",
                    "response": feedback_text,
                    "missing_features": rec.get("missing_features", []),
                }
            )

        need_prompt = [r for r in per_item if r["item_id"] not in perfect_ids_used]
        if not need_prompt:
            return id_to_feedback, total_token_cost, total_runtime_cost

        current_z = self.feature_tracker.get_known_context()

        if feedback_fn is not None:
            for rec in need_prompt:
                prompt = self._build_feedback_prompt(rec, current_z)
                if not prompt:
                    continue
                text, tc, rt = feedback_fn(prompt)
                id_to_feedback[rec["item_id"]] = text
                total_token_cost += tc
                total_runtime_cost += rt
                self._feedback_prompt_history.append(
                    {
                        "item_id": rec["item_id"],
                        "prompt": prompt,
                        "response": text,
                        "missing_features": rec["missing_features"],
                    }
                )
            return id_to_feedback, total_token_cost, total_runtime_cost

        # Batch feedback via LOTUS sem_map
        feedback_df = pd.DataFrame(
            [
                {
                    "item_id": rec["item_id"],
                    "prompt": self._build_feedback_prompt(rec, current_z),
                }
                for rec in need_prompt
            ]
        )
        feedback_df = feedback_df[feedback_df["prompt"].str.len() > 0]
        if feedback_df.empty:
            return id_to_feedback, total_token_cost, total_runtime_cost

        lotus.settings.configure(lm=self._lm)
        prompt_template = "{prompt}"
        try:
            mapped = sem_map_with_retries(
                feedback_df,
                prompt_template,
                validation_fn=lambda x: isinstance(x, str) and len(x.strip()) >= 10,
            )
        except Exception as e:
            if self.verbosity > 0:
                print_debug(
                    f"ItemComparer LOTUS feedback sem_map failed: {e}",
                    "ItemComparer.compute_feedback",
                )
            for rec in need_prompt:
                if rec["item_id"] not in id_to_feedback:
                    id_to_feedback[rec["item_id"]] = ""
            return id_to_feedback, total_token_cost, total_runtime_cost

        for idx, row in mapped.iterrows():
            iid = row.get("item_id", "")
            raw = row.get("_map", "")
            if isinstance(raw, str) and raw.strip():
                id_to_feedback[iid] = raw.strip()
            else:
                id_to_feedback[iid] = (
                    "I need more details about this item to say whether it matches."
                )

        for _, row in feedback_df.iterrows():
            iid = row["item_id"]
            self._feedback_prompt_history.append(
                {
                    "item_id": iid,
                    "prompt": row["prompt"],
                    "response": id_to_feedback.get(iid, ""),
                }
            )

        return id_to_feedback, total_token_cost, total_runtime_cost
