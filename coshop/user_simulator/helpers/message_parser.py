"""Natural-language message parser for the user simulator.

Parses the shopping agent's response into structured actions that the simulator
can act on: clarifying questions, item evaluations, and explanations.

Main exports:

- :class:`ClarifyingQuestion` — a question the agent asked the user.
- :class:`ItemToEval` — an item the agent showed that the simulator should evaluate.
- :class:`Explanation` — an explanation/statement from the agent about a feature.
- :class:`MessageParser` — parses a full agent message into the above types.
"""

import unittest
from ...utils.lotus import configure_lotus, sem_map_with_retries, sem_filter_with_retries
from ...utils.misc import parse_json, print_debug
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import re
import pandas as pd
import lotus

from .feature_utils import parse_feature_match_with_lotus
from ...data.dataset import OPEN_TO_ANYTHING


@dataclass
class FeatureValue:
    column_name: str
    # True = matches user's requirement, False = does not match, None = not mentioned
    match: Optional[bool]


@dataclass
class ItemToEval:
    id: str
    valid_in_catalog: bool
    raw_description: str
    features: List[FeatureValue]
    relevant_columns: List[str]

    def __repr__(self) -> str:
        # Avoid printing potentially large per-feature details; show a compact summary instead.
        return (
            f"ItemToEval(id={self.id!r}, "
            f"valid_in_catalog={self.valid_in_catalog!r}, "
            f"raw_description={self.raw_description!r}, "
            f"num_features={len(self.features)}, "
            f"relevant_columns={self.relevant_columns!r})"
        )


@dataclass
class ClarifyingQuestion:
    question: str
    relevant_columns: List[str]


@dataclass
class Explanation:
    relevant_column: str
    # True if the explanation mentions or includes the target value for this feature (from target_df)
    target_value_mentioned_or_in_range: bool


@dataclass
class ParsedMessage:
    items_to_evaluate: List[ItemToEval]
    clarifying_questions: List[ClarifyingQuestion]
    explanations: List[Explanation]


ITEM_PARSING_PROMPT = """You are an information extraction assistant.

You will be given:
- A message from a customer service agent to a shopper.
- The id of a specific catalog item referenced in the message.

Your task (for the given item_id):
1. Identify the span of text in the assistant message that directly describes this item.
2. Copy that span **verbatim** (no rephrasing) as the `raw_description`.

Important requirements:
- `raw_description` should be an attempt to capture all text in the message about this item. For example, for a line like:
  "<item><id>919257001</id><information>Harry Jersey Midi Dress: Calf-length viscose jersey with black-and-white floral print, short puff sleeves, stand-up collar with back button opening, waist tie, unlined.</information></item>"
  a good `raw_description` would be:
  "Harry Jersey Midi Dress: Calf-length viscose jersey with black-and-white floral print, short puff sleeves, stand-up collar with back button opening, waist tie, unlined."

INCLUDE ALL INFORMATION THAT THE ASSISTANT HAS GIVEN ABOUT THE ITEM.

Inputs:
- message: {message}
- item_id: {item_id}

Output format:
Return ONLY a single JSON object of the form:
{{
  "raw_description": "<span of text copied from the assistant message that describes this item>"
}}

- `raw_description` MUST be a string.
- Do not include any extra keys, explanations, or text outside this JSON object."""


NOT_CLARIFYING_QUESTION_KEYWORDS = ["would you like me to", "which of these"]


CLARIFYING_QUESTIONS_PROMPT = """You are a question extractor. Given a message from a customer service agent to a shopper, extract ONLY the clarifying questions FROM THE MESSAGE.

Clarifying questions ask about the user's underlying preferences, requirements, background, or constraints for the product search. These questions gather missing information the assistant needs in order to better tailor future recommendations. Examples:
- "Have you ever purchased wool?"
- "What size do you prefer?"
- "Do you have a budget in mind?"
- "Are you willing to relax any constraints?"
- "You said X. Could you clarify what you mean?"
- "Let me know if you have other preferences (e.g., sustainability)" -> "Do you have a preference for sustainability?"

Do NOT include questions that:
- ask for feedback or opinions on specific items or sets of items that were recommended,
- ask the user to choose between items already shown,
- ask what the assistant should do next (e.g., see more details, restart the search, show similar items),
- check whether an explanation was helpful,
- or otherwise talk about the conversation/meta-strategy without asking for new constraints or preferences.

These are NOT clarifying questions and must be EXCLUDED:
- "What do you think of item <item><id>123</id><information>Blue dress.</information></item>?"
- "How do you feel about this one?"
- "Would you like me to start my search over again?"
- "Do you want to see more details?"
- "Do you have any questions?"
- "Would you like to see more colors?"
- "Would you like to see items similar to this one?" 
- "Let me know if you'd like to narrow the list down."
- "Which of these is your favorite?"
- "Which of these three catches your interest the most?"
- "Which of these sounds like the best fit for you?"
- "Do any of these items look interesting to you?"
- "Does that explanation help?"
- "Do you have a preference among these options?"

Breaking Down Questions into Atomic Units

Split compound questions into individual atomic questions, where each question asks about exactly one attribute. For example:
"To narrow it down, could you let me know if you have any preferences for color, material (e.g., cotton, linen, silk), or a price range you'd like to stay within?"
becomes:
1. "Do you have any preferences for color?"
2. "Do you have any preferences for material (e.g., cotton, linen, silk)?"
3. "Do you have any preferences for price range?"

Likewise, any question that bundles multiple attributes with "or" must be split. For example:
"Do you have any preferred colors or patterns?"
becomes:
1. "Do you have any preferred colors?"
2. "Do you have any preferred patterns?"

Exception — Binary Choice Questions. Do NOT split a question that asks the user to choose between exactly two options. For example:
"Do you prefer a standalone novel, or are you open to reading a book that's part of a series?"
This remains a single question.

Making Each Question Standalone
Every question must be fully self-contained with no unresolved references.
- Rewrite follow-ups that reference a prior question. For example, "Do you want red or blue? And how much more do you like one than the other?" becomes:
  1. "Do you want red or blue?"
  2. "How much more do you like red than blue, or vice versa?"
- Omit questions that cannot be rewritten because their reference cannot be resolved. For example, "Are you truly open to either style?" must be dropped if "either style" is undefined.
- Expand vague references to options. For example, "Do you have a preference among these fabric options?" must be rewritten to name the options explicitly: "Do you have a preference among these fabric options (cotton / polyester)?"

Return each clarifying question on its own line. If there are no clarifying questions, return exactly: NONE
Do not include any other text, numbering, or explanation. DO NOT MAKE UP QUESTIONS: ONLY EXTRACT THEM FROM THE FOLLOWING MESSAGE."""


EXPLANATIONS_PARSING_PROMPT = """You are a helper that identifies and extracts extended educational explanations from a customer service agent message.

An EXPLANATION is when the agent dedicates AT LEAST A PARAGRAPH of text to teaching the user about a feature: what it means or what values/options it can take (e.g. "Sustainable items are made with...", "Some lengths include floor-length, ankle-length, calf-length...", "Some patterns include stripes & florals."). Do NOT treat questions as explanations: e.g. "What skirt length are you looking for?" is a question, not an explanation—return no explanation for that. Do not treat short lists inside questions ("What colors (e.g. blue, red) are you looking for?") as explanations. This is not a paragraph-length explanation.

You are given:
1. The assistant message.
2. A JSON object whose KEYS are catalog column names and whose values describe each column and give example values. 
3. A JSON object "target_values" whose KEYS are the same catalog column names and whose values are lists of the TARGET value(s) for that feature (the item(s) we care about). Use this to set "target_value_mentioned_or_in_range".

Your task: If the agent explains a feature (describes what the feature is or what options/values it can take), return an object with:
- "relevant_column": the exact catalog column name from the JSON keys that is being explained (e.g. dress_length for skirt/dress length, graphical_appearance_name for patterns).
- "target_value_mentioned_or_in_range": boolean. True if the explanation mentions or includes the target value for this feature. The explanation may describe a range or list of options; if the target value (or a broader category that includes it) is mentioned, set true. Examples: if the explanation says "Dress patterns include stripes and polka dots" and the target value is "large stripes", set true (stripes is mentioned). If the target value is "chevron" or "solid", set false (not mentioned). If the explanation lists "floor-length, ankle-length, calf-length" and the target value is "calf-length", set true.
- "justification": a short sentence explaining why the target value is mentioned or in range.
There should only be one object per feature in the list.

Examples:
- Message: "A dress's skirt length refers to how long it is on the knee. Some common lengths include floor-length, ankle-length, calf-length, knee-length, and mini, meaning above the knee. Typically longer lengths are considered more modest. Shorter lengths can sometimes elongate the leg." with target value "calf-length" → {{"relevant_column": "dress_length", "target_value_mentioned_or_in_range": true, "justification": "The explanation lists 'calf-length' as a common length."}}
- Message: "What skirt length are you looking for?" → [] (no explanation; this is a question)
- Message: "Some patterns for dresses include stripes & florals." with target value "large stripes" → {{"relevant_column": "graphical_appearance_name", "target_value_mentioned_or_in_range": true, "justification": "The explanation lists 'stripes' as a pattern from which an average internet user could derive the target value 'large stripes'."}}
- Message: "Some patterns for dresses include stripes & florals." with target value "chevron" → {{"relevant_column": "graphical_appearance_name", "target_value_mentioned_or_in_range": false, "justification": "The explanation does not mention the target value 'chevron'."}}
- Message: "Price refers to the cost of the dress. A typical price for a dress is $50 to $100." with target value "under $70" → {{"relevant_column": "price", "target_value_mentioned_or_in_range": true, "justification": "The target range (0-70) overlaps with the explanation range (50-100)."}}
- Message: "Price refers to the cost of the dress. A typical price for a dress is around $100." with target_value "under $20" → {{"relevant_column": "price", "target_value_mentioned_or_in_range": false, "justification": "The target range (0-20) does not overlap with the explanation range, which is around 100. 'Around 100' can be treated as (80-120)."}}
- Message: "What kind of material a dress is made of affects its look. For the specifications you gave, I would recommend materials like cotton and polyester." with target value "viscose" → {{"relevant_column": "material", "target_value_mentioned_or_in_range": false, "justification": "The explanation doesn't list the target value 'viscose', and an average internet user could not derive viscose from the other presented options."}}

Assistant message:
{message}

Catalog columns with target values:
{features_descriptions}

If there are no such explanations, return an empty list.

REMEMBER: an explanation is at least a paragraph of text dedicated to teaching the user about a single feature.

Output format: Return ONLY a JSON array of objects, each with exactly: "relevant_column" (string) and "target_value_mentioned_or_in_range" (boolean). No other keys or text.
"""
from typing import Any, Dict


def _fmt(js: Dict[str, Any]) -> str:
    """Flatten nested dict into 'a.b.c=value;...' with no braces."""

    parts = []

    def walk(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                walk(v, new_prefix)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                walk(v, new_prefix)
        else:
            parts.append(f"{prefix}={obj}")

    walk(js)
    return ";".join(parts)


class MessageParser:
    """
    Parses a message from the user or assistant into a structured format.
    Ideally tags should be exploded out in the catalog first.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict,
        verbosity: int,
        target_df: pd.DataFrame,
        catalog: pd.DataFrame,
        column_descriptions: Dict[str, str],
        max_features_to_reveal: int = None,
        use_item_jsons: bool = False,
        use_oracle_item_representations: bool = False,
        use_actual_item_values: bool = False,
        representation: Any = None,
        parser_reasoning_effort: str = "medium",
        max_text_len: Optional[int] = None,
    ):
        """
        Args:
            target_df: this should be from ds.simulator_catalog -> subset to xstar ids -> explode tags
            catalog: the original catalog
            column_descriptions: a dict mapping column names to human-readable descriptions
            model_name: the name of the model to use
            model_kwargs: a dict of kwargs to pass to the model
            verbosity: the verbosity level
            use_item_jsons: if True, expect items as per-line JSON objects (with "id" and feature keys) instead of <item>...</item> tags.
            use_oracle_item_representations: if True, build ItemToEval from catalog only (no LLM calls for item content).
            representation: used with use_oracle for non-JSON raw_description (row_to_str). Required when use_oracle and not use_item_jsons.
            max_text_len: when use_oracle and not use_item_jsons, truncate row_to_str to this length (match query/query.py).
        """
        mk = model_kwargs.copy()
        mk["temperature"] = 0.0
        # Keep MessageParser extraction consistently high-effort regardless of caller kwargs.
        mk["reasoning_effort"] = parser_reasoning_effort

        self.lm = configure_lotus(model_name, mk)
        self.verbosity = verbosity
        self.catalog = catalog
        self.target_df = target_df
        # Keep column_descriptions as provided (for human-readable text), but do not
        # use it to gate which catalog columns LOTUS may reference.
        self.column_descriptions = column_descriptions or {}
        self._max_features_to_reveal = max_features_to_reveal
        self._use_item_jsons = use_item_jsons
        self._use_oracle_item_representations = use_oracle_item_representations
        self._use_actual_item_values = use_actual_item_values
        self._representation = representation
        self._max_text_len = max_text_len

        # Precompute a feature summary restricted to the target/xstar ids, over
        # *all* catalog columns (not just those with explicit descriptions).
        feature_summary: Dict[str, Any] = {}
        for col_name in self.target_df.columns:
            desc = self.column_descriptions.get(col_name, col_name)
            series = self.target_df[col_name]
            vals = series.dropna().tolist()
            if not vals:
                continue
            feature_summary[col_name] = {
                "description": desc or col_name,
                "values": [str(v) for v in vals],
            }
        self._feature_summary = feature_summary
        # Cache column order for consistent feature merging across parses.
        self._column_names: List[str] = list(self.target_df.columns)
        self._non_ota_cols = [
            c
            for c in self._column_names
            if not self.target_df[c].isin([OPEN_TO_ANYTHING]).any()
        ]
        self._parsed_history: List[ParsedMessage] = []
        # Running per-item state aggregated across all previous parses. For each
        # item id, we keep the most recent non-null presented_value per feature.
        self._item_state: Dict[str, ItemToEval] = {}

    @property
    def parsed_history(self) -> List[ParsedMessage]:
        """History of parsed messages in order (oldest first)."""
        return self._parsed_history

    def clear_parsed_history(self) -> None:
        """Clear the parsed message history (e.g. on simulator reset)."""
        self._parsed_history = []
        self._item_state = {}

    @property
    def items_by_id(self) -> Dict[str, ItemToEval]:
        """
        Aggregated view of all parsed items so far, keyed by id.

        For each feature, we keep the most recent non-None match seen for that
        item across all messages. A None match never overwrites a non-None one.
        """
        return self._item_state

    def _columns_previously_sole_relevant(self) -> Set[str]:
        """
        Columns that have appeared as the sole relevant_columns entry for any
        clarifying question in parsed history. Used to bias future mappings
        toward columns that have not yet been the sole focus of a question.
        """
        cols: Set[str] = set()
        for pm in self._parsed_history:
            for cq in pm.clarifying_questions:
                if len(cq.relevant_columns) == 1:
                    cols.add(cq.relevant_columns[0])
        return cols

    def _get_column_types(self, column_names: List[str]) -> Dict[str, str]:
        """
        Return a mapping column_name -> column_type ('numeric', 'boolean', or 'string').

        - Columns present in the catalog with a numeric dtype → 'numeric'
        - Columns present in the catalog with a boolean dtype → 'boolean'
        - Tag columns (present in target_df but absent from catalog.columns) → 'boolean'
        - All other columns → 'string'
        """
        result: Dict[str, str] = {}
        for col in column_names:
            if col not in self.catalog.columns:
                result[col] = "boolean"
            elif pd.api.types.is_bool_dtype(self.target_df[col]):
                result[col] = "boolean"
            elif pd.api.types.is_numeric_dtype(self.target_df[col]):
                result[col] = "numeric"
            else:
                result[col] = "string"
        return result

    def _merge_items_across_history(self, items: List[ItemToEval]) -> List[ItemToEval]:
        """
        Given items parsed from the current message, update the running per-id
        state and return the representations for the ids that appeared in this
        message.

        Item features and raw_descriptions are assumed to already reflect
        cross-turn history (LOTUS was run on merged raw text), so we simply
        overwrite prior state for each id with the latest ItemToEval.
        """
        if not items:
            return []

        # Ignore items with empty id.
        items = [it for it in items if it.id and str(it.id).strip()]

        # First update global state in the order items are seen.
        for it in items:
            self._item_state[it.id] = it

        # Then return one merged ItemToEval per distinct id in this message,
        # preserving first-seen order within the message.
        seen_ids = set()
        result: List[ItemToEval] = []
        for it in items:
            if it.id in seen_ids:
                continue
            seen_ids.add(it.id)
            merged = self._item_state.get(it.id)
            if merged is not None:
                result.append(merged)
        return result

    def _get_oracle_description(
        self,
        item_id: str,
        active_columns: List[str] = None,
        use_raw_column_names: bool = False,
    ) -> Optional[str]:
        """Return the oracle (catalog) representation for an item, or None if not in catalog or no representation."""
        if not item_id or not str(item_id).strip() or item_id not in self.catalog.index:
            return None
        row = self.catalog.loc[item_id]
        if self._use_item_jsons:
            return json.dumps(
                {
                    (
                        col
                        if use_raw_column_names
                        else self.column_descriptions.get(col, col)
                    ): str(row[col])
                    for col in row.index
                    if col in (active_columns or self._non_ota_cols)
                },
                indent=2,
            )
        if self._representation is not None:
            desc = self._representation.row_to_str(row)
            if self._max_text_len is not None:
                desc = desc[: self._max_text_len]
            return desc
        return None

    def _item_to_eval_from_catalog(
        self,
        item_id: str,
        true_values: Optional[Dict[str, str]] = None,
    ) -> Optional[ItemToEval]:
        """Build one ItemToEval from the oracle/catalog description via LOTUS match."""
        oracle_desc = self._get_oracle_description(item_id)
        if oracle_desc is None:
            return None
        raw_description = oracle_desc
        column_names = list(self.target_df.columns)
        column_types = self._get_column_types(column_names)

        match_by_item: Dict[str, Dict[str, Optional[bool]]] = {}
        if true_values:
            lotus.settings.configure(lm=self.lm)
            feature_rows: List[Dict[str, Any]] = []
            for column_name in column_names:
                tv = true_values.get(column_name)
                if tv is None or tv == OPEN_TO_ANYTHING:
                    continue
                feature_rows.append(
                    {
                        "message": raw_description,
                        "item_id": item_id,
                        "column_name": column_name,
                        "column_description": self.column_descriptions.get(
                            column_name, None
                        ),
                        "column_type": column_types[column_name],
                        "true_value": tv,
                    }
                )
            if feature_rows:
                if self.verbosity > 1:
                    print_debug(
                        "Using LOTUS feature match for item to eval from catalog.",
                        "MessageParser._item_to_eval_from_catalog",
                        "orange",
                    )
                match_by_item = parse_feature_match_with_lotus(
                    pd.DataFrame(feature_rows),
                    verbosity=getattr(self, "verbosity", 0),
                )

        per_item_matches = match_by_item.get(item_id, {})
        feature_values: List[FeatureValue] = [
            FeatureValue(column_name=col, match=per_item_matches.get(col))
            for col in column_names
        ]
        return ItemToEval(
            id=item_id,
            valid_in_catalog=True,
            raw_description=raw_description,
            features=feature_values,
            relevant_columns=[
                f.column_name for f in feature_values if f.match is not None
            ],
        )

    def _apply_oracle_match_override(
        self,
        agent_match_by_item: Dict[str, Dict[str, Optional[bool]]],
        oracle_descs: Dict[str, str],
        build_feature_rows_fn,
        run_feature_match_fn,
        column_names: List[str],
    ) -> Dict[str, Dict[str, Optional[bool]]]:
        """
        For items in oracle_descs, run a second feature-match call against the oracle
        catalog description and use those results wherever the agent's message indicated
        a feature was mentioned (non-None). Features the agent did not mention stay None.

        Items that have no oracle description are left with their original agent-message
        match results.
        """
        oracle_rows: List[Dict[str, Any]] = []
        for item_id, desc in oracle_descs.items():
            oracle_rows.extend(build_feature_rows_fn(item_id, desc))
        oracle_match_by_item = run_feature_match_fn(oracle_rows)

        merged: Dict[str, Dict[str, Optional[bool]]] = {}
        for item_id, agent_matches in agent_match_by_item.items():
            if item_id in oracle_descs:
                oracle_matches = oracle_match_by_item.get(item_id, {})
                merged[item_id] = {
                    col: oracle_matches.get(col)
                    if agent_matches.get(col) is not None
                    else None
                    for col in column_names
                }
            else:
                merged[item_id] = agent_matches
        return merged

    def _extract_items_with_descs(
        self,
        message: str,
    ) -> List[Dict[str, Any]]:
        """
        Prepare items for LOTUS by merging the current raw description with any historical
        raw_description before running LOTUS, so feature parsing sees the full
        cross-turn context.
        Also filters out invalid ids
        """
        item_descs: Dict[str, str] = {}

        if self._use_item_jsons:
            # Collect all non-empty lines as ("item", js) or ("text", stripped),
            # preserving order so we can compute before/after context per item.
            line_entries: List[Tuple[str, Any]] = []
            for line in message.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                js = parse_json(stripped)
                if isinstance(js, dict) and "id" in js:
                    item_id = str(js["id"]).strip()
                    if item_id:
                        line_entries.append(("item", js))
                        continue
                line_entries.append(("text", stripped))

            item_entries: List[Dict[str, Any]] = [
                e[1] for e in line_entries if e[0] == "item"
            ]

            # First, concatenate multiple JSON lines for the same id within this
            # message, preserving order.
            current_info_by_id: Dict[str, str] = {}
            for js in item_entries:
                item_id = str(js.get("id", "")).strip()
                if not item_id:
                    continue
                if item_id not in self.catalog.index:
                    # Keep invalid ids, but mark their descs as None so callers
                    # can decide how to handle them.
                    if item_id not in item_descs:
                        item_descs[item_id] = None
                    continue
                current_raw = json.dumps(js, ensure_ascii=False)
                if item_id in current_info_by_id:
                    current_info_by_id[item_id] = (
                        current_info_by_id[item_id]
                        + "\nAdditional updated information:\n"
                        + current_raw
                    )
                else:
                    current_info_by_id[item_id] = current_raw

            # Then merge with oracle or historical descriptions.
            for item_id, info_text in current_info_by_id.items():
                if self._use_oracle_item_representations:
                    it = self._get_oracle_description(item_id)
                    item_descs[item_id] = it
                else:
                    prev = self._item_state.get(item_id)
                    if prev is not None and prev.raw_description:
                        merged_raw = (
                            prev.raw_description
                            + "\nAdditional updated information:\n"
                            + info_text
                        )
                    else:
                        merged_raw = info_text
                    item_descs[item_id] = merged_raw

            return item_descs

        item_ids = re.compile(
            r"<id>\s*([^<]+?)\s*</id>",
            re.DOTALL,
        )
        item_ids = [m.group(1).strip() for m in item_ids.finditer(message)]

        if self._use_oracle_item_representations:
            for item_id in item_ids:
                it = self._get_oracle_description(item_id)
                item_descs[item_id] = it
            return item_descs

        # Attempt to find <item><id>...</id><information>...</information></item> tags.
        # Id can be numeric (e.g. HM) or ISBN-style with hyphens (e.g. Goodreads 978-1-938771-45-2).
        item_tag_pattern = re.compile(
            r"<item>\s*<id>\s*([^<]+?)\s*</id>\s*<information>(.*?)</information>\s*</item>",
            re.DOTALL,
        )
        # Collect all <item><id>...</id><information>...</information></item> tags
        # in order, allowing the same id to appear multiple times. For each id,
        # we concatenate the information spans in the order they occur so that
        # later references (e.g., short follow-ups) are preserved.
        info_by_id: Dict[str, str] = {}
        for m in item_tag_pattern.finditer(message):
            id_str = m.group(1).strip()
            if not id_str:
                continue
            info = m.group(2).strip()
            if id_str in info_by_id:
                info_by_id[id_str] = (
                    info_by_id[id_str] + "\nAdditional updated information:\n" + info
                )
            else:
                info_by_id[id_str] = info

        # For ids we couldn't find this way, use LOTUS to extract the raw description.
        missing_ids = [id_str for id_str in item_ids if id_str not in info_by_id]
        if missing_ids:
            lotus.settings.configure(lm=self.lm)
            extraction_df = pd.DataFrame(
                [{"message": message, "item_id": item_id} for item_id in missing_ids]
            )

            def _raw_desc_validation_fn(x: str) -> bool:
                try:
                    parsed = parse_json(x)
                    return isinstance(parsed, dict) and isinstance(
                        parsed.get("raw_description"), str
                    )
                except Exception:
                    return False

            if self.verbosity > 1:
                print_debug(
                    "Using LOTUS raw_description fallback for item parsing.",
                    "MessageParser._parse_items",
                    "orange",
                )
            try:
                extraction_df = sem_map_with_retries(
                    extraction_df,
                    ITEM_PARSING_PROMPT,
                    validation_fn=_raw_desc_validation_fn,
                )
                extracted_item_info: Dict[str, str] = {}
                for idx, item_id in enumerate(missing_ids):
                    raw_map = (
                        extraction_df["_map"].iloc[idx]
                        if idx < len(extraction_df)
                        else ""
                    )
                    parsed = parse_json(raw_map)
                    raw_desc = ""
                    if isinstance(parsed, dict):
                        raw_desc_val = parsed.get("raw_description")
                        if isinstance(raw_desc_val, str):
                            raw_desc = raw_desc_val.strip()
                    extracted_item_info[item_id] = raw_desc if raw_desc else message
            except Exception as e:
                # Use the full msg
                extracted_item_info = {item_id: message for item_id in missing_ids}

            info_by_id.update(extracted_item_info)

        # Now build final descriptions matching the formatting used in _parse_items:
        # - wrap non-fallback descriptions with an ID header/footer,
        # - merge with historical raw_description,
        # - keep invalid ids but set their descs to None.
        for item_id in item_ids:
            info_text = info_by_id.get(item_id)
            if info_text is None:
                # If we somehow have an id with no info (should be rare), mark as None.
                item_descs[item_id] = None
                continue

            # Detect whether this id ever appeared inside a full <item> tag.
            in_item_tag = item_id in [
                m.group(1).strip()
                for m in item_tag_pattern.finditer(message)
                if m.group(1).strip()
            ]

            if in_item_tag:
                base_raw = (
                    f"--------------- ID: {item_id} -------------\n"
                    f"{info_text}\n"
                    "---------------------------------------"
                )
            else:
                # For fallback-only ids, mimic the used_lotus_raw_desc_fallback branch.
                base_raw = info_text

            prev = self._item_state.get(item_id)
            if prev is not None and prev.raw_description:
                merged_raw = (
                    prev.raw_description
                    + "\nAdditional updated information:\n"
                    + base_raw
                )
            else:
                merged_raw = base_raw

            # Keep invalid ids, but with descs=None.
            if item_id not in self.catalog.index:
                item_descs[item_id] = None
            else:
                item_descs[item_id] = merged_raw

        return item_descs

    def feature_match_item_descriptions(
        self,
        items: Dict[str, str],
        column_names: List[str],
    ) -> Tuple[Dict[str, Dict[str, Optional[bool]]], List[str]]:
        """
        Run LOTUS feature match on explicit item descriptions, restricted to the
        given catalog columns (same row shape as ``_parse_items`` / ``_build_feature_rows``).
        Calls :func:`user_simulator.feature_utils.parse_feature_match_with_lotus` directly.
        Does not append to ``_parsed_history``.

        Returns:
            (match_by_item, active_columns): ``active_columns`` lists columns with a
            concrete target (not missing, not ``OPEN_TO_ANYTHING``), in ``column_names`` order.
        """
        if not items:
            return {}, []

        true_values = {
            col: " OR ".join(self.target_df[col].astype(str).unique())
            for col in self.target_df.columns
        }

        active_columns: List[str] = []
        for col in column_names:
            if col not in self.target_df.columns:
                continue
            tv = true_values.get(col)
            if tv is None or tv == OPEN_TO_ANYTHING:
                continue
            active_columns.append(col)

        if not active_columns:
            return {}, []

        column_types = self._get_column_types(list(self.target_df.columns))
        all_rows: List[Dict[str, Any]] = []
        for item_id, msg_text in items.items():
            for col in active_columns:
                tv = true_values.get(col)
                all_rows.append(
                    {
                        "message": msg_text,
                        "item_id": item_id,
                        "column_name": col,
                        "column_description": self.column_descriptions.get(col, None),
                        "column_type": column_types[col],
                        "true_value": tv,
                    }
                )

        lotus.settings.configure(lm=self.lm)
        if self.verbosity > 1:
            print_debug(
                "feature_match_item_descriptions: LOTUS feature match",
                "MessageParser.feature_match_item_descriptions",
                "orange",
            )
        match_by_item = parse_feature_match_with_lotus(
            pd.DataFrame(all_rows),
            verbosity=getattr(self, "verbosity", 0),
        )
        return match_by_item, active_columns

    def _build_feature_rows(
        self, item_id: str, msg_text: str, column_names: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Build one row per (item, column) for parse_feature_match_with_lotus."""
        if column_names is None:
            column_names = self._column_names

        rows = []
        for col in column_names:
            tv = " OR ".join(self.target_df[col].astype(str).unique())
            if tv is None or tv == OPEN_TO_ANYTHING:
                continue
            rows.append(
                {
                    "message": msg_text,
                    "item_id": item_id,
                    "column_name": col,
                    "column_description": self.column_descriptions.get(col, None),
                    "true_value": tv,
                }
            )
        return rows

    def _build_rows_for_items(self, message: str):
        item_descs = self._extract_items_with_descs(message)

        # Then build LOTUS rows.
        all_rows: List[Dict[str, Any]] = []
        for item_id, info_text in item_descs.items():
            if info_text is None:
                continue
            all_rows.extend(self._build_feature_rows(item_id, info_text))
        return all_rows, item_descs

    def _parse_items(
        self,
        message: str,
    ) -> List[ItemToEval]:
        """
        Parse items from the message, directly producing match booleans per feature.

        Args:
            message: the assistant message to parse.
            true_values: mapping column_name -> user's requirement string.
                When provided, each FeatureValue.match is True/False (or None if
                the feature was not mentioned). When None, all matches are None.
        """
        if not message or not message.strip():
            return []

        column_names = list(self.target_df.columns)

        def _run_feature_match(
            all_rows: List[Dict[str, Any]],
        ) -> Dict[str, Dict[str, Optional[bool]]]:
            if not all_rows:
                return {}
            if self.verbosity > 1:
                print_debug(
                    "Using LOTUS feature match for item parsing.",
                    "MessageParser._run_feature_match",
                    "orange",
                )
            return parse_feature_match_with_lotus(
                pd.DataFrame(all_rows),
                verbosity=getattr(self, "verbosity", 0),
            )

        all_rows, item_descs = self._build_rows_for_items(message)
        if all_rows:
            match_by_item = _run_feature_match(all_rows)
        else:
            match_by_item = {}

        # For non-None matches, apply oracle match override.
        if self._use_actual_item_values:
            oracle_descs: Dict[str, str] = {}
            for item_id, info_text in item_descs.items():
                if info_text is None:
                    continue
                desc = self._get_oracle_description(item_id)
                if desc is not None:
                    oracle_descs[item_id] = desc
            if oracle_descs:
                match_by_item = self._apply_oracle_match_override(
                    match_by_item,
                    oracle_descs,
                    self._build_feature_rows,
                    _run_feature_match,
                    column_names,
                )

        # Combine into ItemToEval objects.
        items: List[ItemToEval] = []
        for item_id, info_text in item_descs.items():
            raw_description = info_text
            valid_in_catalog = item_id in self.catalog.index
            per_item_matches = match_by_item.get(item_id, {})
            feature_values = [
                FeatureValue(column_name=col, match=per_item_matches.get(col))
                for col in column_names
            ]
            relevant_columns = [
                f.column_name for f in feature_values if f.match is not None
            ]
            items.append(
                ItemToEval(
                    id=item_id,
                    valid_in_catalog=valid_in_catalog,
                    raw_description=raw_description,
                    features=feature_values,
                    relevant_columns=relevant_columns,
                )
            )

            if getattr(self, "verbosity", 0) > 0:
                print_debug(
                    f"Built ItemToEval for id= {item_id} raw_description= {raw_description!r} num_features= {len(feature_values)}",
                    "MessageParser._parse_items",
                )

        if getattr(self, "verbosity", 0) > 0 and not items:
            print_debug(
                f"Completed with zero items; item_ids= {item_descs.keys()}",
                "MessageParser._parse_items",
            )
        return items

    def _parse_clarifying_questions(
        self,
        message: str,
        question_columns_to_include: Optional[List[str]] = None,
    ) -> List[ClarifyingQuestion]:
        """
        Parse the clarifying questions from the message.
        """
        if not message or not message.strip():
            return []

        # Step 1: use LOTUS to extract the clarifying question texts.
        lotus.settings.configure(lm=self.lm)

        df_for_lotus = pd.DataFrame([{"message": message}])
        prompt = CLARIFYING_QUESTIONS_PROMPT + "\n\nAssistant message:\n{message}\n"
        if self.verbosity > 1:
            print_debug(
                "Using LOTUS clarifying questions extraction.",
                "MessageParser._parse_clarifying_questions",
                "orange",
            )
        try:
            df_for_lotus = sem_map_with_retries(df_for_lotus, prompt)
        except Exception as e:
            if self.verbosity > 0:
                print_debug(
                    f"MessageParser._parse_clarifying_questions LOTUS sem_map failed: {e}",
                    "MessageParser._parse_clarifying_questions",
                )
            return []
        content = df_for_lotus["_map"].iloc[0] if not df_for_lotus.empty else ""
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content or content.upper() == "NONE":
            return []

        # Parse the model output into a list of atomic clarifying question strings,
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        question_texts: List[str] = []
        for line in lines:
            # Strip common list/numbering prefixes like "1.", "1)", "- ", "* ", "• "
            line = re.sub(r"^\s*\d+[.)]\s*", "", line)
            line = re.sub(r"^\s*[-*•]\s*", "", line)
            if not line or line.upper() == "NONE":
                continue
            # Keyword-based rejection for common non-clarifying questions.
            lower = line.lower()
            if any(kw in lower for kw in NOT_CLARIFYING_QUESTION_KEYWORDS):
                continue
            question_texts.append(line)

        if not question_texts:
            return []

        # Step 2: for each clarifying question, use LOTUS sem_filter to get relevant catalog columns.
        # Build a DataFrame with one row per catalog column (from precomputed feature summary).
        lotus.settings.configure(lm=self.lm)
        feature_summary = self._feature_summary.copy()
        if question_columns_to_include is not None:
            allowed = set(question_columns_to_include)
            feature_summary = {
                col_name: info
                for col_name, info in feature_summary.items()
                if col_name in allowed
            }
        feature_rows = [
            {
                "column_name": col_name,
                "description": info.get("description", ""),
                "values": ", ".join(info.get("values", [])) or "",
            }
            for col_name, info in feature_summary.items()
        ]
        features_df = pd.DataFrame(feature_rows)
        if features_df.empty:
            return [
                ClarifyingQuestion(question=q, relevant_columns=[])
                for q in question_texts
            ]

        filter_instruction = (
            "The user has a preference for feature {column_name} with description '{description}'. "
            "The preferred value is {values}. Decide if this preference is directly relevant to answering this clarification question from a customer service agent: {question}"
        )

        def _filter_validation_fn(filt: pd.DataFrame) -> bool:
            return isinstance(filt, pd.DataFrame) and "column_name" in filt.columns

        clarifying_questions: List[ClarifyingQuestion] = []
        for q in question_texts:
            df_with_question = features_df.copy()
            df_with_question["question"] = q
            filtered = sem_filter_with_retries(
                df_with_question,
                filter_instruction,
                validation_fn=_filter_validation_fn,
            )
            relevant_columns = (
                filtered["column_name"].tolist()
                if not filtered.empty and "column_name" in filtered.columns
                else []
            )
            relevant_columns = [
                c for c in relevant_columns if c in self.target_df.columns
            ]
            # Order by catalog column order (coarser / earlier columns first)
            col_order = {c: i for i, c in enumerate(self._column_names)}
            relevant_columns = sorted(
                relevant_columns, key=lambda c: col_order.get(c, len(col_order))
            )
            # Bias toward columns that have not previously been the sole
            # relevant column of any clarifying question.
            previously_sole = self._columns_previously_sole_relevant()
            if previously_sole:
                new_cols = [c for c in relevant_columns if c not in previously_sole]
                old_cols = [c for c in relevant_columns if c in previously_sole]
                relevant_columns = new_cols + old_cols

            if (
                self._max_features_to_reveal is not None
                and len(relevant_columns) > self._max_features_to_reveal
            ):
                relevant_columns = relevant_columns[: self._max_features_to_reveal]

            clarifying_questions.append(
                ClarifyingQuestion(question=q, relevant_columns=relevant_columns)
            )

        return clarifying_questions

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
        filtered_feature_summary: Dict[str, Any] = {
            col: self._feature_summary[col]
            for col in self._feature_summary
            if col in explanation_columns_to_include
        }
        if not filtered_feature_summary:
            return []
        target_values = {
            col: filtered_feature_summary[col]["values"]
            for col in filtered_feature_summary
            if col in explanation_columns_to_include
        }

        desc = "\n".join(
            [
                f"{i + 1}. COLUMN NAME `{col}` ({filtered_feature_summary[col]['description']}) | LIST OF TARGET VALUES: `{target_values[col]}`"
                for i, col in enumerate(explanation_columns_to_include)
            ]
        )

        df_for_lotus = pd.DataFrame(
            [
                {
                    "message": message,
                    "features_descriptions": desc,
                }
            ]
        )
        prompt = EXPLANATIONS_PARSING_PROMPT

        def _validation_fn(x: str) -> bool:
            try:
                parsed = parse_json(x)
                if not isinstance(parsed, list):
                    return False
                for item in parsed:
                    if not isinstance(item, dict):
                        return False
                    if (
                        "relevant_column" not in item
                        or "target_value_mentioned_or_in_range" not in item
                    ):
                        return False
                    tv = item["target_value_mentioned_or_in_range"]
                    if not isinstance(tv, bool):
                        return False
                return True
            except Exception:
                return False

        if self.verbosity > 1:
            print_debug(
                "Using LOTUS explanations parsing.",
                "MessageParser._parse_explanations",
                "orange",
            )
        try:
            df_for_lotus = sem_map_with_retries(
                df_for_lotus, prompt, validation_fn=_validation_fn
            )
        except Exception as e:
            if self.verbosity > 0:
                print_debug(
                    f"MessageParser._parse_explanations LOTUS sem_map failed: {e}",
                    "MessageParser._parse_explanations",
                )
            return []
        raw_map = df_for_lotus["_map"].iloc[0] if not df_for_lotus.empty else None
        if raw_map is None or (isinstance(raw_map, float) and pd.isna(raw_map)):
            return []

        try:
            parsed = parse_json(raw_map)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []

        explanations: List[Explanation] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            col = item.get("relevant_column")
            target_in_range = item.get("target_value_mentioned_or_in_range", False)
            if not isinstance(target_in_range, bool):
                target_in_range = bool(target_in_range)
            if not col or not isinstance(col, str):
                continue
            if col in self.target_df.columns:
                explanations.append(
                    Explanation(
                        relevant_column=col,
                        target_value_mentioned_or_in_range=target_in_range,
                    )
                )
        return explanations

    def parse(
        self,
        message: str,
        question_columns_to_include: Optional[List[str]] = None,
        explanation_columns_to_include: List[str] = None,
    ) -> ParsedMessage:
        """
        Parse a message from the user or assistant into a structured format.
        Appends the result to parsed_history (in order).

        Args:
            message: the assistant message to parse.
            question_columns_to_include: If provided, restrict which catalog columns
                LOTUS is allowed to select as relevant for the clarifying-question
                extraction step.
        """
        if not message or not message.strip():
            parsed = ParsedMessage(
                items_to_evaluate=[],
                clarifying_questions=[],
                explanations=[],
            )
        else:
            raw_items = self._parse_items(message)
            merged_items = self._merge_items_across_history(raw_items)
            parsed = ParsedMessage(
                items_to_evaluate=merged_items,
                clarifying_questions=self._parse_clarifying_questions(
                    message,
                    question_columns_to_include=question_columns_to_include,
                ),
                explanations=self._parse_explanations(
                    message,
                    explanation_columns_to_include=explanation_columns_to_include,
                ),
            )
        self._parsed_history.append(parsed)
        return parsed


class MessageParserHMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ...data.hm.data import HMDataset

        hm = HMDataset(dev=True, max_xstar=1)
        cls.hm_catalog = hm.catalog
        cls.hm_simulator_catalog = hm.simulator_catalog
        cls.hm_column_descriptions = hm.true_features

    def _build_parser(self, target_ids=None, use_item_jsons=False):
        if target_ids is None:
            target_ids = self.hm_catalog.index[:3].tolist()
        # target_df: simulator view subset to target ids, with tags exploded out
        target_df = self.hm_simulator_catalog.loc[
            self.hm_simulator_catalog.index.intersection(pd.Index(target_ids))
        ].copy()
        if "tags" in target_df.columns:
            from ...utils.misc import explode_df

            target_df = explode_df(target_df, "tags")
        parser = MessageParser(
            model_name="gpt-5-nano",
            model_kwargs={},
            verbosity=1,
            target_df=target_df,
            catalog=self.hm_catalog,
            column_descriptions=self.hm_column_descriptions,
            use_item_jsons=use_item_jsons,
        )
        return parser, target_ids

    def test_parse_empty_message(self):
        parser, _ = self._build_parser()
        parsed = parser.parse("   ")
        print_debug(f"parsed: {parsed}", "test_parse_empty_message")
        self.assertEqual(parsed.items_to_evaluate, [])
        self.assertEqual(parsed.clarifying_questions, [])
        self.assertEqual(parsed.explanations, [])

    def test_parse_clarifying_questions_only(self):
        parser, _ = self._build_parser()
        msg = (
            "Quick question to narrow things down further: "
            "do you have any preferences for color, material, or a price range?"
        )
        parsed = parser.parse(msg)
        lines = ["clarifying_questions:"]
        for cq in parsed.clarifying_questions:
            lines.append(f"  - {cq.question} | cols: {cq.relevant_columns}")
        print_debug("\n".join(lines), "test_parse_clarifying_questions_only")

        # No explicit item tags -> no items to evaluate.
        self.assertEqual(len(parsed.items_to_evaluate), 0, f"parsed={parsed}")
        # Should extract at least one clarifying question.
        self.assertGreaterEqual(len(parsed.clarifying_questions), 1, f"parsed={parsed}")
        for cq in parsed.clarifying_questions:
            self.assertIsInstance(cq.question, str)
            self.assertIsInstance(cq.relevant_columns, list)

    def test_parse_items_and_clarifying_questions(self):
        parser, target_ids = self._build_parser()
        target_id = target_ids[0]
        msg = (
            f"I think <item><id>{target_id}</id><information>Looks warm and is made from a soft material.</information></item> "
            "Do you have any preferences for color or price range?"
        )
        parsed = parser.parse(msg)
        lines = ["items_to_evaluate:"]
        for it in parsed.items_to_evaluate:
            lines.append(f"   - id: {it.id}")
            lines.append(f"     raw_description: {it.raw_description}")
            lines.append(
                f"     features: {[(fv.column_name, fv.presented_value) for fv in it.features]}"
            )
        lines.append("  clarifying_questions:")
        for cq in parsed.clarifying_questions:
            lines.append(f"   - {cq.question} | cols: {cq.relevant_columns}")
        print_debug("\n".join(lines), "test_parse_items_and_clarifying_questions")

        self.assertGreaterEqual(
            len(parsed.items_to_evaluate),
            1,
            f"expected at least one item; parsed={parsed}",
        )
        item_ids = {it.id for it in parsed.items_to_evaluate}
        self.assertIn(
            target_id, item_ids, f"target_id={target_id}, item_ids={item_ids}"
        )
        self.assertGreaterEqual(
            len(parsed.clarifying_questions),
            1,
            f"expected at least one clarifying question; parsed={parsed}",
        )

    def test_parse_items_json_mode(self):
        """With use_item_jsons=True, items are parsed from per-line JSON objects."""
        parser, target_ids = self._build_parser(
            target_ids=["919257001"], use_item_jsons=True
        )
        msg = (
            '{"id": 919257001, "graphical_appearance_name": "black-and-white floral", '
            '"colour_group_name": "Black"}'
        )
        parsed = parser.parse(msg)
        self.assertGreaterEqual(
            len(parsed.items_to_evaluate),
            1,
            f"expected at least one item in JSON mode; parsed={parsed}",
        )
        item = parsed.items_to_evaluate[0]
        self.assertEqual(item.id, "919257001")
        self.assertGreater(
            len(item.features),
            0,
            "expected non-empty features after reconciliation",
        )

    def test_hm_material_question(self):
        """
        Single clarifying question mapped to 'material'.
        """
        target_id = "859399001"
        parser, _ = self._build_parser(target_ids=[target_id])
        msg = (
            "Thanks for the clarification! So, to summarize:\n\n"
            "Dress type: Women's, loose and comfortable\n"
            "Sleeves: Short with a bit of volume, specifically puff sleeves\n"
            "Length: Knee-length or midi (around mid-calf)\n"
            "Fabric: Soft, slightly shiny, drapes nicely, unlined, no stretch\n"
            "Waist: Tie or some waist detail (open to styles)\n"
            "Pockets: None\n"
            "Color/Pattern: Black or subtle black-and-white floral print\n"
            "Occasion: Casual daytime wear\n"
            "Size: To be confirmed by you\n"
            "Is there a specific fabric you prefer or want to avoid, like silk, satin, polyester, or something else?"
        )
        parsed = parser.parse(msg)

        lines = [
            f"  - {cq.question} | cols: {cq.relevant_columns}"
            for cq in parsed.clarifying_questions
        ]
        print_debug(
            "\n".join(lines) if lines else "(none)", "test_hm_material_question"
        )

        self.assertEqual(
            len(parsed.items_to_evaluate),
            0,
            f"expected no items; parsed={parsed}",
        )
        self.assertEqual(
            len(parsed.clarifying_questions),
            1,
            f"expected one clarifying question; parsed={parsed}",
        )
        cq = parsed.clarifying_questions[0]
        self.assertSetEqual(set(cq.relevant_columns), {"material"})

    def test_hm_multi_clarifying_questions(self):
        """
        Multiple clarifying questions with specific expected relevant columns.
        """
        target_id = "859399001"
        parser, _ = self._build_parser(target_ids=[target_id])
        msg = (
            "Thanks for sharing your preferences! To narrow down the options, I want to clarify a few things:\n\n"
            "What length do you prefer for the dress? (e.g., knee-length, midi, maxi)\n"
            "Do you have any preferred colors or patterns?\n"
            "Is there a particular occasion or style you want the dress for? (e.g., casual, work, evening)\n"
            "What size are you generally looking for?\n"
            "This will help me find dresses that best match what you want."
        )
        parsed = parser.parse(msg)

        lines = [
            f"  - {cq.question} | cols: {cq.relevant_columns}"
            for cq in parsed.clarifying_questions
        ]
        print_debug(
            "\n".join(lines) if lines else "(none)",
            "test_hm_multi_clarifying_questions",
        )

        self.assertEqual(
            len(parsed.items_to_evaluate),
            0,
            f"expected no items; parsed={parsed}",
        )
        # Expect exactly 5 atomic questions after splitting.
        self.assertEqual(
            len(parsed.clarifying_questions),
            5,
            f"expected 5 clarifying questions; parsed={parsed}",
        )

        for cq in parsed.clarifying_questions:
            q_lower = cq.question.lower()
            cols = set(cq.relevant_columns)
            if "length" in q_lower:
                self.assertSetEqual(cols, {"dress_length"})
            elif "color" in q_lower and "pattern" not in q_lower:
                self.assertSetEqual(
                    cols,
                    {
                        "colour_group_name",
                        "perceived_colour_value_name",
                        "perceived_colour_master_name",
                    },
                )
            elif "pattern" in q_lower:
                self.assertSetEqual(cols, {"graphical_appearance_name"})
            elif "occasion" in q_lower or "style" in q_lower:
                self.assertSetEqual(cols, set())
            elif "size" in q_lower:
                self.assertSetEqual(cols, set())

    def test_hm_no_items_no_clarifying_questions(self):
        """
        Strategy-only message: no <item> tags and only meta questions -> no items, no clarifying questions.
        """
        parser, _ = self._build_parser()
        msg = (
            "I found several dresses that match most of your preferences. Here are some options:\n\n"
            "919257001 - Harry Jersey Midi Dress: Calf-length viscose jersey with black-and-white floral print, short puff sleeves, "
            "stand-up collar with back button opening, waist tie, unlined.\n\n"
            "859399001 - Sam Dress: Knee-length viscose dress in black-and-white floral print, short puff sleeves, back-neck button opening, "
            "waist tie, unlined, soft draping fabric.\n\n"
            "919257002 - Similar to the first Harry Jersey Midi Dress but in solid black.\n\n"
            "Would you like to see more details about any of these dresses or explore other options?"
        )
        parsed = parser.parse(msg)
        print_debug(
            f"items_to_evaluate: {parsed.items_to_evaluate}\n  clarifying_questions: {parsed.clarifying_questions}",
            "test_hm_no_items_no_clarifying_questions",
        )

        self.assertEqual(len(parsed.items_to_evaluate), 0, f"parsed={parsed}")
        self.assertEqual(len(parsed.clarifying_questions), 0, f"parsed={parsed}")

    def test_hm_item_parsing_with_raw_description(self):
        """
        Single item with rich description; ensure id, raw_description, and key features are extracted.
        """
        parser, _ = self._build_parser(target_ids=["859399001", "919257001"])
        msg = (
            "I found several dresses that match most of your preferences. Here are some options:\n\n"
            "<item><id>919257001</id><information>Harry Jersey Midi Dress: Calf-length viscose jersey with black-and-white floral print, "
            "short puff sleeves, stand-up collar with back button opening, waist tie, unlined.</information></item>"
        )
        parsed = parser.parse(msg)

        lines = []
        for it in parsed.items_to_evaluate:
            lines.append(f"  id: {it.id}")
            lines.append(f"  raw_description: {it.raw_description}")
            lines.append(
                f"  features: {[(fv.column_name, fv.presented_value) for fv in it.features]}"
            )
        print_debug(
            "\n".join(lines) if lines else "(none)",
            "test_hm_item_parsing_with_raw_description",
        )

        self.assertEqual(
            len(parsed.items_to_evaluate),
            1,
            f"expected one item; parsed={parsed}",
        )
        item = parsed.items_to_evaluate[0]
        self.assertEqual(item.id, "919257001")
        # Raw description should quote the descriptive span from the assistant message.
        self.assertIn("Harry Jersey Midi Dress", item.raw_description)
        self.assertIn("Calf-length viscose jersey", item.raw_description)
        self.assertIn("black-and-white floral print", item.raw_description)

        # Basic feature coverage: ensure key columns are present.
        feature_by_col = {fv.column_name: fv for fv in item.features}
        expected_cols = {
            "prod_name",
            "dress_length",
            "material",
            "graphical_appearance_name",
            "sleeve_length",
            "has_lining",
        }
        self.assertTrue(
            expected_cols.issubset(set(feature_by_col.keys())),
            f"missing expected feature columns; have={set(feature_by_col.keys())}",
        )

    def test_hm_explanation_dress_length(self):
        """Agent explains dress/skirt length options → one Explanation with dress_length; target value should be in range."""
        parser, _ = self._build_parser()
        msg = (
            "A dress's skirt length refers to how long it is on the knee. Some common lengths include "
            "floor-length, ankle-length, calf-length, knee-length, and mini, meaning above the knee. "
            "Typically longer lengths are considered more modest. Shorter lengths can sometimes elongate the leg."
        )
        parsed = parser.parse(msg)
        print_debug(
            f"explanations: {parsed.explanations}", "test_hm_explanation_dress_length"
        )
        self.assertGreaterEqual(
            len(parsed.explanations),
            1,
            f"expected at least one explanation; parsed={parsed}",
        )
        dress_length_expl = [
            e for e in parsed.explanations if e.relevant_column == "dress_length"
        ]
        self.assertEqual(
            len(dress_length_expl),
            1,
            f"expected one dress_length explanation; explanations={parsed.explanations}",
        )
        # Target df has dress_length values; message lists floor/ankle/calf/knee/mini, so target should be in range.
        self.assertTrue(
            dress_length_expl[0].target_value_mentioned_or_in_range,
            f"dress_length explanation lists common lengths; target value should be in range; got {dress_length_expl[0]}",
        )

    def test_hm_explanation_none_for_question(self):
        """Question only (no teaching) → no explanations."""
        parser, _ = self._build_parser()
        msg = "What skirt length are you looking for?"
        parsed = parser.parse(msg)
        print_debug(
            f"explanations: {parsed.explanations}",
            "test_hm_explanation_none_for_question",
        )
        self.assertEqual(
            len(parsed.explanations),
            0,
            f"expected no explanations for a question; parsed={parsed}",
        )

    def test_hm_explanation_patterns(self):
        """Agent lists pattern options → one Explanation for graphical_appearance_name; target_value_mentioned_or_in_range depends on target."""
        parser, _ = self._build_parser()
        msg = "Some patterns for dresses include stripes & florals."
        parsed = parser.parse(msg)
        print_debug(
            f"explanations: {parsed.explanations}", "test_hm_explanation_patterns"
        )
        self.assertGreaterEqual(
            len(parsed.explanations),
            1,
            f"expected at least one explanation; parsed={parsed}",
        )
        pattern_expl = [
            e
            for e in parsed.explanations
            if e.relevant_column == "graphical_appearance_name"
        ]
        self.assertEqual(
            len(pattern_expl),
            1,
            f"expected one graphical_appearance_name explanation; explanations={parsed.explanations}",
        )
        # Must have the new field (value depends on whether target_df has stripes/florals or something else).
        self.assertIsInstance(
            pattern_expl[0].target_value_mentioned_or_in_range,
            bool,
            f"target_value_mentioned_or_in_range must be bool; got {pattern_expl[0]}",
        )

    def test_clarifying_history_bias_and_no_truncation(self):
        """
        Clarifying-question mapping should:
        - keep all relevant columns from LOTUS (no truncation by max_features_to_reveal),
        - bias toward columns that have not previously been the sole relevant column.

        We simulate LOTUS behavior via a local stub for sem_map_with_retries.
        """
        from user_simulator import message_parser as mp

        original_sem_map = mp.sem_map_with_retries
        original_sem_filter = mp.sem_filter_with_retries

        def fake_sem_map_with_retries(df, prompt, validation_fn=None):
            # First pass: question extraction (df has only 'message')
            if "message" in df.columns and "question" not in df.columns:
                msg = str(df["message"].iloc[0])
                if "GENERAL_PATTERN_MSG" in msg:
                    content = "What kind of pattern would you like?"
                elif "SPECIFIC_PATTERN_MSG" in msg:
                    content = "What specific kind of pattern would you like?"
                else:
                    content = "NONE"
                return pd.DataFrame({"_map": [content]}, index=df.index)

            # Fallback: just echo a valid empty list (second pass now uses sem_filter)
            return pd.DataFrame(
                {"_map": [json.dumps([], ensure_ascii=False)]}, index=df.index
            )

        def fake_sem_filter_with_retries(
            df, user_instruction, validation_fn=None, **kwargs
        ):
            # Second pass: filter features by relevance to the clarification question
            if "question" not in df.columns or "column_name" not in df.columns:
                return df
            q = str(df["question"].iloc[0])
            if "specific kind of pattern" in q:
                keep = ["graphical_appearance_name", "colour_group_name", "material"]
            elif "kind of pattern" in q:
                keep = ["graphical_appearance_name"]
            else:
                keep = []
            return df[df["column_name"].isin(keep)].copy()

        try:
            mp.sem_map_with_retries = fake_sem_map_with_retries
            mp.sem_filter_with_retries = fake_sem_filter_with_retries

            # Build a parser with a strict max_features_to_reveal to ensure that
            # MessageParser itself is not truncating relevant_columns.
            parser, _ = self._build_parser()

            # First message: general pattern question → sole relevant column.
            msg1 = "GENERAL_PATTERN_MSG"
            parsed1 = parser.parse(msg1)
            self.assertEqual(len(parsed1.clarifying_questions), 1)
            cq1 = parsed1.clarifying_questions[0]
            self.assertEqual(cq1.relevant_columns, ["graphical_appearance_name"])

            # Second message: more specific pattern question → multiple columns,
            # but history should bias toward the columns that were not previously
            # the sole relevant column.
            msg2 = "SPECIFIC_PATTERN_MSG"
            parsed2 = parser.parse(msg2)
            self.assertEqual(len(parsed2.clarifying_questions), 1)
            cq2 = parsed2.clarifying_questions[0]
            # All three columns should be present (no truncation), and the two
            # that have never been sole columns should come first.
            self.assertEqual(
                cq2.relevant_columns,
                ["colour_group_name", "material", "graphical_appearance_name"],
            )
        finally:
            mp.sem_map_with_retries = original_sem_map
            mp.sem_filter_with_retries = original_sem_filter


if __name__ == "__main__":
    unittest.main()
