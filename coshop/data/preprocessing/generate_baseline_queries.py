"""
Generate z0 baseline queries.

For each user, prompt an LLM (LOTUS sem_map) to write one baseline search query
from the spec's z0 preference summary. Used by the random and popularity
baseline policies. Requires a gpt-oss-120b vLLM server. The SEC split and
initial known features must exist first (z0 is built from them).

Output JSON format:
  {
    "100": {"z0": {"query": "<baseline query from z0>"}},
    ...
  }

Usage:
    python -m coshop.data.preprocessing.generate_baseline_queries --dataset all \\
        --vllm-url http://localhost:3005
"""

import argparse
import json
from typing import Any, Dict, List

import pandas as pd

from ...utils.lotus import configure_lotus, sem_map_with_retries
from ._common import (
    add_common_args,
    load_dataset,
    output_path,
    read_json,
    resolve_datasets,
    todo_indexes,
    user_indexes,
    write_json,
)

MODEL_NAME = "openai/gpt-oss-120b"


def _build_query_instruction(item_type: str) -> str:
    """
    Instruction for LOTUS sem_map to turn a single preference summary into a baseline search query.
    """
    return (
        "You are an expert search assistant.\n\n"
        f'You will see a text field called "preferences" describing a user\'s preferences for {item_type}.\n'
        "Your task is to generate exactly ONE short search query that you might type into a search engine to find items matching this single summary.\n"
        "The queries should be phrases you would type into a search engine to find matching items.\n"
        "They should be in natural language. The more detailed the query, the better.\n\n"
        "OUTPUT FORMAT:\n"
        "- Return ONLY a single JSON object.\n"
        '- It must have exactly one string field named "query" containing the search query.\n'
        '- Example (structure only): {{"query": "<baseline search query>"}}\n\n'
        "Here is the input text you should base your queries on:\n\n"
        "{preferences}"
    )


def _extract_json_object(text: Any) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from arbitrary model output.
    """
    if text is None:
        return {}
    s = str(text)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = s[start : end + 1]
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _parse_query(raw: Any) -> str:
    obj = _extract_json_object(raw)
    val = obj.get("query") or obj.get("q") or obj.get("baseline_query")
    if val is None:
        return ""
    if isinstance(val, list) and val:
        return str(val[0]).strip()
    return str(val).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM-based z0 baseline queries."
    )
    add_common_args(parser)
    parser.add_argument("--vllm-url", required=True, help="gpt-oss-120b vLLM base URL.")
    args = parser.parse_args()

    configure_lotus(
        MODEL_NAME,
        {"model_provider": "vllm", "vllm_api_url": args.vllm_url, "temperature": 0.0},
    )

    for name in resolve_datasets(args):
        ds = load_dataset(name, args)
        out = output_path(ds, "baseline_queries_extended", args)
        existing = {k: v for k, v in read_json(out).items() if v.get("z0")}
        todo = todo_indexes(user_indexes(ds, args), existing, args, out)
        if not todo:
            continue

        rows: List[Dict[str, Any]] = [
            {"spec_index": ix, "preferences": str(ds[ix].z0)} for ix in todo
        ]
        df = pd.DataFrame(rows)
        lm_df = sem_map_with_retries(df, _build_query_instruction(ds.item_name), retries=3)

        for spec_ix, raw in zip(df["spec_index"].tolist(), lm_df["_map"].tolist()):
            existing.setdefault(str(spec_ix), {})["z0"] = {"query": _parse_query(raw)}

        write_json(out, existing)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
