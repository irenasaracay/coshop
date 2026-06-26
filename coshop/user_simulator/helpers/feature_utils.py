from __future__ import annotations

from typing import Dict, Optional, List

import json
import pandas as pd

from ...utils.lotus import sem_map_with_retries
from ...utils.misc import print_debug, parse_json


FEATURE_PARSING_PROMPT = """You are an information extraction assistant.

You will be given:
- A text about catalog item(s)
- The id of a specific catalog item to pay attention to
- The name of a single catalog column (feature) to look for.
- An optional human-readable description of that column.
- The column_type for this feature, which is one of: numeric, boolean, string.

Your task (for the given item_id, column_name, and column_type):
1. Decide whether the text says anything about the value of this feature for this specific item.
2. If the text clearly states a value for this feature for this item, return that value as a short phrase that matches column_type:
   - For numeric: return a short numeric description (e.g. "42", "$50–$70", "under 20").
   - For boolean: return a short phrase indicating whether it applies (e.g. "true", "false", "has lining", "no pockets").
   - For string: return a short descriptive phrase (e.g. "navy blue", "midi length", "floral print").
3. If the feature is NOT mentioned at all for this item, return exactly: not_mentioned
4. If the feature is mentioned but the value is explicitly described as uncertain, unknown, or not in the catalog, return exactly: agent_uncertain

Inputs:
- message: {message}
- item_id: {item_id}
- column_name: {column_name}
- column_description: {column_description}
- column_type: {column_type}

Output format:
- Return ONLY a single string:
  - either a short value phrase (guided by column_type)
  - or one of the special tokens: not_mentioned, agent_uncertain
Do NOT return JSON or any additional explanation."""


def parse_feature_values_with_lotus(
    df_features: pd.DataFrame,
    verbosity: int = 0,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Run LOTUS per-feature parsing over the provided dataframe and return a nested
    mapping item_id -> column_name -> normalized presented_value.

    This helper can be used both for:
    - parsing presented values from assistant messages or JSON lines, and
    - applying boolean/tag-style features to oracle/catalog descriptions.

    The dataframe must have columns: message, item_id, column_name,
    column_description, column_type.

    Normalization rules:
    - 'not_mentioned' or empty → None
    - 'agent_uncertain' (case-insensitive) → 'agent_uncertain'
    - otherwise: stripped string value (including 'true'/'false' for boolean tags)
    """
    if df_features.empty:
        return {}

    if verbosity > 0:
        item_ids = sorted({str(v).strip() for v in df_features["item_id"].tolist()})
        print_debug(
            f"Per-feature parsing for item_ids={item_ids}, num_rows_for_lotus= {len(df_features)}",
            "feature_utils.parse_feature_values_with_lotus",
        )

    df_features = sem_map_with_retries(df_features, FEATURE_PARSING_PROMPT)

    feature_values_by_item: Dict[str, Dict[str, Optional[str]]] = {}
    for _, row in df_features.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            continue
        col_name = str(row.get("column_name"))
        raw_val = row.get("_map", "")
        if raw_val is None:
            val_str = ""
        else:
            if not isinstance(raw_val, str):
                try:
                    val_str = str(raw_val)
                except Exception:
                    val_str = ""
            else:
                val_str = raw_val
            val_str = val_str.strip()

        lower = val_str.lower()
        if lower == "not_mentioned" or val_str == "":
            presented_value: Optional[str] = None
        elif lower == "agent_uncertain":
            presented_value = "agent_uncertain"
        else:
            presented_value = val_str

        feature_values_by_item.setdefault(item_id, {})[col_name] = presented_value

    return feature_values_by_item


FEATURE_MATCH_PROMPT = """Evaluate whether the text supports the following requirement.

Feature: {column_name}
Meaning: {column_description}
Required value: {true_value}

Decision process:
1. Look for text that provides deliberate, intentional evidence about the feature {column_description}. If the speaker is not trying to point out the feature to you, return not_mentioned.
2. Check if the text mentions the value. If the text does not discuss this feature, return not_mentioned. Otherwise, continue.
3. Compare the inferred value to {true_value}. Determine whether the implied value semantically matches the user's requirement. Be generous: treat synonyms, paraphrases, compatible values, and values that fall within the user's stated range as matching. Only say NO if the implied value clearly contradicts or does not satisfy the user's requirement.

If the feature name is "buttons" and the presented value is "buttoned-cuffs" and the requirement is True, these are the same: if something has buttoned cuffs, then it also has buttons. If the feature name is "product type" and the presented value is "dress" and the actual value is "garment full body" these are also the same: a dress is a kind of full body garment. Another example: a series with 0 books in it is the same as a series with 1 book in it (both values mean 'standalone book'). A movie rated PG-13 or G is the same as being "not rated R." If a movie has Latin dub and the requirement is "Spoken language: Swedish or Latin", then this movie matches the requirement. If a movie is mentioned as having two protagonists (a male and a female), and the requirements is "Protagonist gender: female", then this movie matches the requirement: at least one protagonist has the right gender.

Only say NO if the implied value clearly contradicts the user's requirement. If a book described as a biogrpahy, then a feature like 'genre = fantasy' is NO.

Output rules:
- YES → the text implies a value that does not contradict {true_value} or semantically matches it.
- NO → the text implies a value that contradicts {true_value}.
- agent_uncertain → the text mentions {column_description} but says the value is unknown or not in the catalog.
- not_mentioned → the text contains no evidence about {column_description}.

Return exactly one of:
YES
NO
agent_uncertain
not_mentioned

Only return YES or NO if there is enough information presented to make a clear inference. If the feature is not explicitly mentioned, favor not_mentioned.

Text:
{message}
"""


def parse_feature_match_with_lotus(
    df_features: pd.DataFrame,
    verbosity: int = 0,
) -> Dict[str, Dict[str, Optional[bool]]]:
    """
    In a single LOTUS call, extract each feature's presented value from the text
    and determine whether it matches the user's requirement (true_value).

    Input dataframe columns (same as parse_feature_values_with_lotus, plus true_value):
      message, item_id, column_name, column_description, column_type, true_value

    Returns:
      item_id -> column_name -> True (match) | False (mismatch) | None (not_mentioned or agent_uncertain)
    """
    if df_features.empty:
        return {}

    if verbosity > 0:
        item_ids = sorted({str(v).strip() for v in df_features["item_id"].tolist()})
        print_debug(
            f"Per-feature match for item_ids={item_ids}, num_rows_for_lotus= {len(df_features)}",
            "feature_utils.parse_feature_match_with_lotus",
        )

    def validation_fn(x: str) -> bool:
        if not isinstance(x, str):
            return False
        return x.strip().upper() in ("YES", "NO", "NOT_MENTIONED", "AGENT_UNCERTAIN")

    try:
        df_features = sem_map_with_retries(
            df_features, FEATURE_MATCH_PROMPT, validation_fn=validation_fn, retries=5
        )
    except Exception as e:
        if verbosity > 0:
            print_debug(
                f"parse_feature_match_with_lotus LOTUS sem_map failed: {e}",
                "parse_feature_match_with_lotus",
            )
        # assume all are NOT_MENTIONED
        return {
            item_id: {
                col_name: None for col_name in df_features["column_name"].unique()
            }
            for item_id in df_features["item_id"].unique()
        }

    match_by_item: Dict[str, Dict[str, Optional[bool]]] = {}
    for _, row in df_features.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            continue
        col_name = str(row.get("column_name"))
        raw_val = row.get("_map", "")
        if raw_val is None:
            val_str = ""
        else:
            if not isinstance(raw_val, str):
                try:
                    val_str = str(raw_val)
                except Exception:
                    val_str = ""
            else:
                val_str = raw_val
        upper = val_str.strip().upper()
        if upper == "YES":
            result: Optional[bool] = True
        elif upper == "NO":
            result = False
        else:
            result = None

        match_by_item.setdefault(item_id, {})[col_name] = result

    return match_by_item


def normalize_feature_name(name: Optional[str]) -> Optional[str]:
    """
    Normalize feature column name for display:
    - strip any parenthetical elaboration ("short desc (elaboration)" -> "short desc")
    - convert hyphens to spaces
    - lowercase, then capitalize the first letter
    """
    if not name or not isinstance(name, str):
        return name
    # Column descriptions are of the form "short description (elaboration)"
    # Remove the elaboration part
    if "(" in name and ")" in name:
        name = name.split("(")[0].strip()
    return name.strip().lower().replace("-", " ").capitalize()


FILTER_DUPLICATE_FEATURES_PROMPT = """You are helping to clean up a list of catalog feature columns.

You are given a list of candidate feature columns, each described by:
- column_name: the internal feature name
- description: a human-readable description of the feature

Goal:
- Decide which subset of these features should be KEPT in the final list, in order to avoid redundant, overly-granular features.
- Prefer COARSER, more user-facing features over very granular or derivative ones.

Examples of redundancy:
- If there is a general 'Pattern' feature and also a more specific 'Granular pattern type (e.g., polka dots, pinstripes, melange, colour blocking)', you should:
  - KEEP the coarse 'Pattern' feature (graphical_appearance_name)
  - DROP the granular 'Granular pattern type' feature (pattern_detail_type)
- If there is both 'Color' and 'Color family (e.g., warm vs cool)', you would typically KEEP 'Color' and DROP 'Color family' for a concise user-facing list, unless only one of them is present.

Decision rules:
1. Use the description text to understand how coarse vs granular each feature is.
2. If a column clearly represents a more granular or derivative split of another broader feature (e.g., "granular", "detail", "subtype", "secondary", "specific variant"), favor dropping that granular column when both are present.
3. If a column is the main, user-facing way someone would talk about this concept (e.g., "Pattern", "Color", "Price"), favor keeping it.
4. If you are unsure for a particular feature, default to keeping it.

Input:
- features_json: a JSON array of objects, each with:
  - "column_name": string
  - "description": string

Output:
- Return ONLY a JSON array of the column_name strings that should be KEPT, in the order you recommend presenting them.
  Example: ["graphical_appearance_name", "colour_group_name"]

features_json:
{features_json}
"""


def filter_duplicate_features(
    column_names: List[str], column_descriptions: Dict[str, str]
) -> List[str]:
    """
    Given a list of feature column names, call LOTUS to decide which ones to
    keep, preferring coarser, more user-facing features over granular variants.

    This is used to prune `relevant_columns` for items and clarifying questions.
    """
    if not column_names:
        return []

    # Deduplicate while preserving order before sending to LOTUS.
    unique_cols: List[str] = list(dict.fromkeys(column_names))

    features = [
        {
            "column_name": col,
            "description": column_descriptions.get(col, col),
        }
        for col in unique_cols
    ]
    features_json = json.dumps(features, ensure_ascii=False)

    df = pd.DataFrame(
        [
            {
                "features_json": features_json,
            }
        ]
    )

    def _validation_fn(x: str) -> bool:
        try:
            parsed = parse_json(x)
        except Exception:
            return False
        if not isinstance(parsed, list):
            return False
        # All entries must be strings and must be subset of unique_cols.
        return all(isinstance(v, str) and v in unique_cols for v in parsed)

    try:
        df = sem_map_with_retries(
            df, FILTER_DUPLICATE_FEATURES_PROMPT, validation_fn=_validation_fn
        )
    except Exception as e:
        # assume that all columns are unique and keep them all
        return unique_cols

    raw = df["_map"].iloc[0] if not df.empty else "[]"

    try:
        kept = parse_json(raw)
    except Exception:
        kept = unique_cols

    if not isinstance(kept, list) or not kept:
        kept = unique_cols

    # Preserve the original order from `unique_cols`, restricted to those we kept.
    kept_set = {c for c in kept if isinstance(c, str)}
    return [c for c in unique_cols if c in kept_set]
