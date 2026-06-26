"""
Parse assistant messages that use structured dialog-action lines (see
``policy.conversational.MSG_FMT_STRUCTURED_DIALOG_ACTIONS``).

Each action is one line: KEYWORD + space + JSON object.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...utils.misc import parse_json, print_debug

from .message_parser import (
    ClarifyingQuestion,
    Explanation,
    MessageParser,
    sem_map_with_retries,
)
import pandas as pd
import lotus

_ASK_QUESTION = "ASK_QUESTION "
_SHOW_ITEM = "SHOW_ITEM_FOR_FEEDBACK "
_ITEM_FOLLOWUP = "ITEM_FOLLOWUP "
_EXPLAIN = "EXPLAIN "


EXPLANATIONS_PARSING_PROMPT = """You are a helper that checks if an explanation covers a target value for a feature.

An EXPLANATION is when the agent dedicates AT LEAST A PARAGRAPH of text to teaching the user about a feature: what it means or what values/options it can take (e.g. "Sustainable items are made with...", "Some lengths include floor-length, ankle-length, calf-length...", "Some patterns include stripes & florals."). Do NOT treat questions as explanations: e.g. "What skirt length are you looking for?" is a question, not an explanation—return no explanation for that. Do not treat short lists inside questions ("What colors (e.g. blue, red) are you looking for?") as explanations. This is not a paragraph-length explanation.

You are given:
1. The assistant message.
2. A description of each column name in the catalog, as well as a corresponding list of target values for that column. Use this to set "target_value_mentioned_or_in_range".

Your task: FOR EACH COLUMN IN THE CATALOG, determine if the agent's explanation mentions or includes the target value for this feature. The explanation may describe a range or list of options; if the target value (or a broader category that includes it) is mentioned, set true. Examples: if the explanation says "Dress patterns include stripes and polka dots" and the target value is "large stripes", set true (stripes is mentioned). If the target value is "chevron" or "solid", set false (not mentioned). If the explanation lists "floor-length, ankle-length, calf-length" and the target value is "calf-length", set true.

Return a comma separated list of the column names which are 
covered by the explanation.

Assistant message:
{explanation_text}

Catalog columns with target values:
{features_descriptions}

Return a comma-separated list that represents a subset of the following columns.
Include a column if its target value is captured in the assistant message.
{all_cols}
"""


class StructuredActionMessageParser(MessageParser):
    """
    Same constructor and ancillary behavior as ``MessageParser``, but ``parse``
    reads ASK_QUESTION / SHOW_ITEM_FOR_FEEDBACK / EXPLAIN lines instead of
    running LOTUS for questions or scanning for <item> / JSON items.
    """

    def __init__(self, *args, require_strict_column_names: bool = True, **kwargs):
        self.require_strict_column_names = require_strict_column_names
        super().__init__(*args, **kwargs)

    def _parse_structured_question(self, line: str) -> Optional[ClarifyingQuestion]:
        """
        Parse an ASK_QUESTION line into a ClarifyingQuestion object.
        """
        s = line.strip()
        if s.startswith(_ASK_QUESTION):
            raw = s[len(_ASK_QUESTION) :].strip()
            try:
                obj = parse_json(raw)
            except Exception:
                return None
            if isinstance(obj, dict):
                return ClarifyingQuestion(
                    question=obj["question"], relevant_columns=obj["relevant_features"]
                )
            return None
        return None

    def _parse_structured_item(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a SHOW_ITEM_FOR_FEEDBACK or ITEM_FOLLOWUP line into a dict with
        item_id and raw_description. Both actions share the same JSON schema.
        """
        s = line.strip()
        prefix = None
        if s.startswith(_SHOW_ITEM):
            prefix = _SHOW_ITEM
        elif s.startswith(_ITEM_FOLLOWUP):
            prefix = _ITEM_FOLLOWUP
        if prefix is None:
            return None
        raw = s[len(prefix):].strip()
        try:
            obj = parse_json(raw)
        except Exception:
            return None
        if isinstance(obj, dict):
            if "item_id" not in obj:
                return None
            ff = obj.get("features_for_feedback", {})
            if isinstance(ff, str):
                coerced = parse_json(ff)
                ff = coerced if isinstance(coerced, (dict, list)) else {}
            if isinstance(ff, dict):
                relevant_columns = list(ff.keys())
            elif isinstance(ff, list):
                relevant_columns = [str(x) for x in ff]
            else:
                ff = {}
                relevant_columns = []
            try:
                raw_description = json.dumps(ff)
            except Exception:
                raw_description = json.dumps(str(ff))
            return {
                "item_id": obj["item_id"],
                "raw_description": raw_description,
                "relevant_columns": relevant_columns,
            }
        return None

    def _parse_structured_explanations(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse an EXPLAIN line into an Explanation object.
        """
        s = line.strip()
        if s.startswith(_EXPLAIN):
            raw = s[len(_EXPLAIN) :].strip()
            try:
                obj = parse_json(raw)
                assert len(obj["relevant_features"])
            except Exception:
                return None
            if isinstance(obj, dict):
                return {
                    "relevant_columns": obj["relevant_features"],
                    "explanation_text": obj["explanation_text"],
                }
            return None
        return None

    def _map_model_feature_names_to_true_feature_names(
        self, model_feature_names: List[str]
    ) -> List[str]:
        if self.require_strict_column_names:
            return [col for col in model_feature_names if col in self._feature_summary]

        lotus.settings.configure(lm=self.lm)

        # Use lotus to try to map each model mentioned feature to a true feature.
        df_for_lotus = pd.DataFrame(model_feature_names, columns=["model_feature_name"])
        df_for_lotus["features_descriptions"] = "\n".join(
            [
                f"{i + 1}. {col}: {self._feature_summary[col]['description']}"
                for i, col in enumerate(self._feature_summary)
            ]
        )
        prompt = """Map the {model_feature_name} to zero, one, or more than one of the true feature names.

        True features:
        {features_descriptions}
        
        Return a comma separated list of the true feature names, or NONE if no mapping is found."""

        def _validation_fn(x: str) -> bool:
            return len(x) > 0

        df_for_lotus = sem_map_with_retries(df_for_lotus, prompt)

        def parse_output(x: str) -> List[str]:
            if x.strip().upper() == "NONE":
                return []
            return [col.strip() for col in x.strip().split(",")]

        out = df_for_lotus["_map"].apply(parse_output)
        nested = out.tolist()
        flat = [item for sublist in nested for item in sublist]
        return list(set(flat))

    #### OVERRIDE MESSAGE PARSER METHODS ####
    def _parse_clarifying_questions(
        self,
        message: str,
        question_columns_to_include: Optional[List[str]] = None,
    ) -> List[ClarifyingQuestion]:
        lines = [line.strip() for line in message.split("\n") if line.strip()]
        clarifying_questions = []
        previously_sole = self._columns_previously_sole_relevant()
        for line in lines:
            clarifying_question = self._parse_structured_question(line)
            if clarifying_question is None:
                continue
            cols = self._map_model_feature_names_to_true_feature_names(
                clarifying_question.relevant_columns
            )
            if question_columns_to_include is not None:
                cols = [c for c in cols if c in question_columns_to_include]
            if previously_sole:
                new_cols = [c for c in cols if c not in previously_sole]
                old_cols = [c for c in cols if c in previously_sole]
                cols = new_cols + old_cols
            if (
                self._max_features_to_reveal is not None
                and len(cols) > self._max_features_to_reveal
            ):
                cols = cols[: self._max_features_to_reveal]
            clarifying_question.relevant_columns = cols
            clarifying_questions.append(clarifying_question)
        return clarifying_questions

    def _build_rows_for_items(self, message: str):
        lines = [line.strip() for line in message.split("\n") if line.strip()]

        current_by_id: Dict[str, Dict[str, Any]] = {}
        for line in lines:
            item = self._parse_structured_item(line)
            if item is None:
                continue
            item_id = item["item_id"]
            if item_id in current_by_id:
                existing = json.loads(current_by_id[item_id]["raw_description"])
                incoming = json.loads(item["raw_description"])
                merged = {**existing, **incoming}
                current_by_id[item_id]["raw_description"] = json.dumps(merged)
                current_by_id[item_id]["relevant_columns"] = list(
                    set(current_by_id[item_id]["relevant_columns"]) | set(item["relevant_columns"])
                )
            else:
                current_by_id[item_id] = dict(item)

        item_descs = {}
        all_rows: List[Dict[str, Any]] = []
        for item_id, item in current_by_id.items():
            if item_id not in self.catalog.index:
                item_descs[item_id] = None
                continue

            raw = item["raw_description"]
            prev = self._item_state.get(item_id)
            if prev is not None and prev.raw_description:
                merged_raw = (
                    prev.raw_description
                    + "\nAdditional updated information:\n"
                    + raw
                )
            else:
                merged_raw = raw

            if self._use_oracle_item_representations:
                oracle = self._get_oracle_description(item_id)
                if oracle is not None:
                    merged_raw = oracle

            item_descs[item_id] = merged_raw
            all_rows.extend(
                self._build_feature_rows(
                    item_id,
                    merged_raw,
                    column_names=self._map_model_feature_names_to_true_feature_names(
                        item["relevant_columns"]
                    ),
                )
            )
        return all_rows, item_descs

    def _parse_explanations(
        self, message: str, explanation_columns_to_include: List[str] = None
    ) -> List[Explanation]:
        """
        Parse credence-style explanations from the message: spans where the agent
        explains what a feature means or what values it can take. Uses LOTUS
        (similar in spirit to CoPrefUser._resolve_credence_explanations
        but extracts the explaining text per column).
        """
        if not message or not message.strip():
            return []
        if not self.target_df.columns.tolist():
            return []
        if explanation_columns_to_include is None:
            explanation_columns_to_include = self._column_names
        if explanation_columns_to_include == []:
            return []

        lotus.settings.configure(lm=self.lm)

        # Restrict the feature/target-value JSONs to the current credence features.
        feature_summary = self._feature_summary if isinstance(self._feature_summary, dict) else {}
        filtered_feature_summary: Dict[str, Any] = {
            col: feature_summary[col]
            for col in feature_summary
            if col in explanation_columns_to_include
        }
        if not filtered_feature_summary:
            return []
        target_values = {
            col: filtered_feature_summary[col]["values"]
            for col in filtered_feature_summary
        }

        rows = []
        for line in message.split("\n"):
            explanation = self._parse_structured_explanations(line)
            if explanation is not None:
                explanation["relevant_columns"] = (
                    self._map_model_feature_names_to_true_feature_names(
                        explanation["relevant_columns"]
                    )
                )
                explanation["relevant_columns"] = [
                    c
                    for c in explanation["relevant_columns"]
                    if c in explanation_columns_to_include
                ]
                if len(explanation["relevant_columns"]) == 0:
                    continue
                rows.append(
                    {
                        "explanation_text": explanation["explanation_text"],
                        "features_descriptions": "\n".join(
                            [
                                f"{i + 1}. COLUMN NAME `{col}` ({filtered_feature_summary[col]['description']}) | LIST OF TARGET VALUES: `{target_values[col]}`"
                                for i, col in enumerate(explanation["relevant_columns"])
                            ]
                        ),
                        "all_cols": explanation["relevant_columns"],
                    }
                )

        df_for_lotus = pd.DataFrame(rows)
        if df_for_lotus.empty:
            return []

        prompt = EXPLANATIONS_PARSING_PROMPT

        try:
            df_for_lotus = sem_map_with_retries(
                df_for_lotus, prompt, validation_fn=lambda x: True
            )
        except Exception as e:
            if self.verbosity > 0:
                print_debug(
                    f"MessageParser._parse_explanations LOTUS sem_map failed: {e}",
                    "MessageParser._parse_explanations",
                )
            return []

        explanations: List[Explanation] = []
        for _, row in df_for_lotus.iterrows():
            in_range_cols = row["_map"].split(",")
            all_cols = row["all_cols"]
            for col in all_cols:
                explanations.append(
                    Explanation(
                        relevant_column=col,
                        target_value_mentioned_or_in_range=col in in_range_cols,
                    )
                )
        return explanations
