"""
Utils for LangChainAgent policies
"""

import json
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional, Callable


from coshop.utils.langchain_agent import parse_langchain_response_to_actions
from coshop.utils.misc import parse_json, print_debug
from coshop.utils.model import LangChainModel


def _extract_seen_ids_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """
    Extract item IDs from search_web tool calls.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        List of unique item IDs seen from search_web tool calls
    """
    seen_ids = set()

    for tool_call in tool_calls:
        if (
            tool_call.get("name")
            in ["search_web", "search_catalog", "search_user_purchase_history"]
            and tool_call.get("status") == "success"
        ):
            response = tool_call.get("response", "")
            # Try to parse response as JSON
            try:
                if isinstance(response, str):
                    response_data = json.loads(response)
                else:
                    response_data = response

                if isinstance(response_data, list):
                    for item in response_data:
                        if isinstance(item, dict) and "id" in item:
                            seen_ids.add(str(item["id"]))
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

    return list(seen_ids)


def _estimate_items_requested_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> int:
    """
    Roughly estimate how many catalog items the agent *requested* across a
    sequence of search tool calls, by summing the max_items argument when
    available and falling back to the number of returned items otherwise.
    """
    total = 0
    for tc in tool_calls:
        if tc.get("name") not in [
            "search_web",
            "search_catalog",
            "search_user_purchase_history",
        ]:
            continue
        raw_kwargs = tc.get("kwargs", "{}")
        max_items = None
        try:
            if isinstance(raw_kwargs, str):
                parsed_kwargs = json.loads(raw_kwargs)
            else:
                parsed_kwargs = raw_kwargs
            max_items = parsed_kwargs.get("max_items")
        except Exception:
            parsed_kwargs = {}
            max_items = None
        try:
            if max_items is not None:
                max_items_int = int(max_items)
            else:
                max_items_int = None
        except Exception:
            max_items_int = None

        if max_items_int is not None and max_items_int > 0:
            total += max_items_int
            continue

        # Fallback: use length of response payload (if it looks like a list).
        raw_resp = tc.get("response", "[]")
        try:
            if isinstance(raw_resp, str):
                parsed_resp = json.loads(raw_resp)
            else:
                parsed_resp = raw_resp
            if isinstance(parsed_resp, list):
                total += len(parsed_resp)
        except Exception:
            # If parsing fails, we just skip this call.
            continue

    return total


_CONVERSATIONAL_PROMPTS_PATH = (
    Path(__file__).resolve().parent / "conversational_prompts.json"
)
with _CONVERSATIONAL_PROMPTS_PATH.open() as f:
    _CONVERSATIONAL_PROMPTS = json.load(f)
_FINAL_RECOMMENDATION_SYSTEM_MESSAGE_TEMPLATE = _CONVERSATIONAL_PROMPTS[
    "final_recommendation_system_message_template"
]
_JSON_FMT_INSTRUCTIONS = _CONVERSATIONAL_PROMPTS["json_fmt_instructions"]


def create_temp_agent(
    original_agent: LangChainModel,
    retrieval_function: Optional[Callable] = None,
    execution_max_per_retrieval: Optional[int] = None,
    execution_max_queries: int = None,
    execution_global_max: Optional[int] = None,
    min_react_steps: int = None,
    max_react_steps: int = None,
    prediction_summarize_after: Optional[int] = None,
    **kwargs,
):
    """
    Create a temporary agent executor with updated query tool for get_final_predictions.
    This allows using a higher execution_max_per_retrieval value during prediction while keeping the original agent intact.

    Args:
        retrieval_function: The query function to use
        execution_max_per_retrieval: The max_items_limit for the new tool
        execution_max_queries: Maximum number of queries
        execution_global_max: Maximum total items that can be retrieved across all query calls
        prediction_summarize_after: If set (not None), summarize the inherited conversation
            history once up front so the prediction agent runs on a shorter context, and arm
            per-step mid-rollout compression after this many fresh tool results. None (default)
            leaves the prediction agent on the full inherited context (original behavior).

    Returns:
        A temporary LangChainModel with updated tools, or None if not a LangchainAgent
    """
    from coshop.tools.catalog_retrieval import get_retrieval_tool
    from coshop.tools.reflect import get_reflect_tool

    new_tools = (
        [
            get_retrieval_tool(
                retrieval_function,
                max_items_limit=execution_max_per_retrieval,
                execution_max_queries=execution_max_queries,
                execution_global_max=execution_global_max,
            )
        ]
        if retrieval_function is not None
        else []
    )
    thought_tool_access = kwargs.pop("thought_tool_access", False)
    if thought_tool_access:
        new_tools.append(get_reflect_tool())

    # Create a copy of the original agent with new tools and optional step limits
    temp_agent = original_agent.copy_for_prediction(
        tools=new_tools,
        min_react_steps=min_react_steps,
        max_react_steps=max_react_steps,
    )
    old_state = original_agent.get_state()
    temp_agent.load_state(old_state)

    # Shrink the inherited context for the prediction agent (opt-in).
    if prediction_summarize_after is not None:
        # (A) Summarize the inherited tool history once so prediction starts on a short prefix.
        if len(temp_agent) > 0:
            temp_agent.compress_state()
        # (D) Arm per-step mid-rollout compression via the pre_model_hook after this many
        # fresh tool results.
        temp_agent._summarize_state_after = prediction_summarize_after

    # Cache the inherited conversation-history prefix: a single Anthropic cache
    # breakpoint on the last history message lets every downstream generate()
    # call (prediction retries, the foregone-recall continuation, report/rank)
    # re-read the long shared history at ~0.1x input cost instead of full price.
    # No-op for non-Anthropic models or when prompt caching is disabled.
    temp_agent.mark_history_cache_breakpoint()

    return temp_agent


def get_final_predictions_agentic(
    original_agent: LangChainModel,
    k: int,
    retrieval_function: Optional[Callable] = None,
    execution_max_per_retrieval: Optional[int] = None,
    execution_max_queries: Optional[int] = None,
    execution_global_max: Optional[int] = None,
    max_retries: int = 2,
    final_recommendation_system_message_template: Optional[str] = None,
    catalog_ids: Optional[List[str]] = None,
    rng_seed: Optional[int] = None,
    temperature: float = 0.0,
    **kwargs,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Given the state of the agent, return the final predictions and metadata.

    Args:
        original_agent: The agent executor to use
        retrieval_function: The query function to use
        k: Number of items to return
        execution_max_per_retrieval: Maximum items per query (optional). If None and execution_global_max is set,
                the tool will enforce execution_global_max across all queries.
        execution_max_queries: Maximum number of queries (optional)
        execution_global_max: Maximum total items across all queries (optional)
        max_retries: Maximum number of retries for getting valid predictions
        catalog_ids: Optional list of catalog IDs for padding when we fail to
            get exactly k predictions after all retries.
        rng_seed: Optional random seed used when sampling padding IDs.
    """
    if not isinstance(max_retries, int) or max_retries < 1:
        max_retries = 1

    # Create temporary agent with updated query tool if needed
    temp_agent = create_temp_agent(
        original_agent=original_agent,
        retrieval_function=retrieval_function,
        execution_max_per_retrieval=execution_max_per_retrieval,
        execution_max_queries=execution_max_queries,
        execution_global_max=execution_global_max,
        temperature=temperature,
        **kwargs,
    )

    # Build prompt with appropriate constraints
    if retrieval_function is None:
        # No query access - agent must use only items from conversation history
        query_constraint_str = "You do NOT have access to search the web. You must base your recommendations only on items that you have already seen."
    else:
        query_constraints = []
        if execution_max_queries is not None:
            query_constraints.append(f"up to {execution_max_queries} queries")
        if execution_max_per_retrieval is not None:
            query_constraints.append(
                f"max_items up to {execution_max_per_retrieval} per query"
            )
        if execution_global_max is not None:
            query_constraints.append(
                f"total of {execution_global_max} items across all queries"
            )

        query_constraint_str = (
            ". ".join(query_constraints) if query_constraints else "queries as needed"
        )
        if query_constraints:
            query_constraint_str = f"You can make a {query_constraint_str}."

    # Make prompt (use override from counterfactual run if provided)
    template = (
        final_recommendation_system_message_template
        if final_recommendation_system_message_template is not None
        else _FINAL_RECOMMENDATION_SYSTEM_MESSAGE_TEMPLATE
    )
    if "{json_fmt_instructions}" not in template:
        template += "\n\n{json_fmt_instructions}"
    if "{{json_fmt_instructions}}" in template:
        template = template.replace(
            "{{json_fmt_instructions}}", "{json_fmt_instructions}"
        )
    prediction_system_msg = template.format(
        k=k,
        query_constraint_str=query_constraint_str,
        json_fmt_instructions=_JSON_FMT_INSTRUCTIONS,
    )

    try:
        # Retry loop for getting valid JSON
        js: Optional[Dict[str, Any]] = None
        raw = None
        raw_messages = None
        # Track tool calls for the first (standard) prediction step separately so we can
        # compute foregone-recall statistics without being confused by continuation calls.
        all_tool_calls_step1: List[Dict[str, Any]] = []
        best_items: Optional[List[Any]] = None

        for attempt in range(max_retries):
            # Generate the prediction using agentic exploration
            # Use persist_state=True to capture tool calls, then clear state afterwards
            prompt = [("system", prediction_system_msg)]

            # Call generate directly to get raw messages for tool call extraction
            raw_messages = temp_agent.generate(
                dialogs=[prompt],
                persist_state=True,
                remove_thinking_tokens=True,
            )[0]

            # Extract tool calls from this attempt
            actions = parse_langchain_response_to_actions(raw_messages)
            for action in actions:
                all_tool_calls_step1.extend(action.tool_calls)

            # Get the final text response
            from langchain_core.messages import AIMessage

            aimessages = [msg for msg in raw_messages if isinstance(msg, AIMessage)]
            if aimessages:
                raw = aimessages[-1].content
            else:
                raw = ""

            # Try to parse the JSON object
            parsed = parse_json(raw)
            current_items: Optional[List[Any]] = None
            current_js: Optional[Dict[str, Any]] = None

            if (
                parsed is not None
                and isinstance(parsed, dict)
                and "top_k_items" in parsed
                and isinstance(parsed["top_k_items"], list)
                and len(parsed["top_k_items"]) > 0
            ):
                current_js = parsed
                current_items = parsed["top_k_items"]
            elif (
                isinstance(parsed, list)
                and len(parsed) > 0
                and isinstance(parsed[0], dict)
                and "top_k_items" in parsed[0]
                and isinstance(parsed[0]["top_k_items"], list)
                and len(parsed[0]["top_k_items"]) > 0
            ):
                current_js = parsed[0]
                current_items = parsed[0]["top_k_items"]

            if current_items is not None:
                # Record best-so-far items in case we never get exactly k.
                best_items = current_items
                # Accept and break early only if we have exactly k items.
                if len(current_items) == k:
                    js = current_js
                    break

            if attempt < max_retries - 1:
                print_debug(
                    f"Failed to parse valid top_k_items of length {k}, retrying... (attempt {attempt + 1}/{max_retries}). Original response: {raw}",
                    "get_final_predictions_agentic",
                    color="yellow",
                )

        # Helper to pad/truncate ids to exactly k using catalog_ids when available.
        def _pad_ids_to_k(base_items: Optional[List[Any]]) -> List[str]:
            # Deduplicate while preserving order
            dedup: List[str] = []
            seen_local: set[str] = set()
            for x in base_items or []:
                sid = str(x)
                if sid not in seen_local:
                    seen_local.add(sid)
                    dedup.append(sid)

            if catalog_ids is None or k <= 0:
                return dedup[:k]

            import random as _random

            rng = _random.Random(rng_seed if rng_seed is not None else 0)
            catalog_ids_str = [str(i) for i in catalog_ids]
            if not catalog_ids_str:
                return dedup[:k]

            result: List[str] = dedup[:]
            seen_result = set(result)
            available = [cid for cid in catalog_ids_str if cid not in seen_result]

            while len(result) < k and catalog_ids_str:
                if available:
                    cid = rng.choice(available)
                    available.remove(cid)
                else:
                    # Sample with replacement as a last resort
                    cid = rng.choice(catalog_ids_str)
                if cid not in seen_result:
                    seen_result.add(cid)
                    result.append(cid)

            return result[:k]

        # ids from the *first* prediction step (before any possible continuation)
        top_k_ids: List[str]
        if js is None or not isinstance(js, dict) or "top_k_items" not in js:
            # Could not get a valid dict with top_k_items of length k; fall back to padding.
            top_k_ids = _pad_ids_to_k(best_items)
            used_catalog_padding = True
        else:
            top_k_items = js.get("top_k_items", [])
            top_k_ids = _pad_ids_to_k(top_k_items)
            used_catalog_padding = False

        # Build metadata for the first step, including seen_ids.
        seen_ids = _extract_seen_ids_from_tool_calls(all_tool_calls_step1)

        metadata: Dict[str, Any] = {
            "tool_calls": [
                {
                    "name": tc["name"],
                    "kwargs": tc["kwargs"],
                    "response": tc["response"],
                    "status": tc["status"],
                }
                for tc in all_tool_calls_step1
            ],
            "seen_ids": seen_ids,
        }
        if used_catalog_padding:
            metadata["used_catalog_padding"] = True

        return top_k_ids, metadata
    except Exception as e:
        return [], {"error": str(e)}


def get_final_report_agentic(
    original_agent: LangChainModel,
    items: Dict[str, str],
    max_retries: int = 2,
    temperature: float = 0.0,
    use_item_jsons: bool = False,
    **kwargs,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Ask the agent to write a per-item user-facing payload for each candidate item.

    Args:
        use_item_jsons: If True, inputs are JSON catalog records and the model must
            output nested JSON objects per id (same style as JSON item lines in the
            main conversation). If False, inputs are plain catalog text and each value
            must be one <item><id>...</id><information>...</information></item> string.

    Returns:
        (report_map, metadata) where report_map maps item_id (str) -> per-item payload
        (str, or JSON-serialized string for nested dict/list values when use_item_jsons
        is True).
    """
    if not isinstance(max_retries, int) or max_retries < 1:
        max_retries = 1

    if not items:
        return {}, {}

    temp_agent = create_temp_agent(
        original_agent=original_agent,
        retrieval_function=None,
        temperature=temperature,
        **kwargs,
    )

    valid_ids = [str(iid) for iid in items.keys()]
    valid_id_set = set(valid_ids)

    item_field = "catalog_json" if use_item_jsons else "description"
    items_description = "\n".join(
        f"- id: {iid}\n  {item_field}: {repr(text)}" for iid, text in items.items()
    )

    if use_item_jsons:
        output_spec = """Your task is to produce one JSON object per catalog id (in the order shown
below) explaining fit for the user, using the SAME output convention as when you
describe items in this task: each value MUST be a JSON object (not a prose string)
with an "id" field matching that catalog id, feature keys with updated values where
helpful, and any extra keys you need to justify fit (e.g. rationale in string
fields). Use double-quoted keys and string values. Do not wrap item details in
free-floating prose outside those objects.

Aim to make each object self-contained so the user can decide from these JSON lines alone."""
        example_block = """Return ONLY a JSON object mapping item ids (strings) to JSON objects.
Example:
{{
  "id_1": {{"id": "id_1", "color": "navy", "why_it_fits": "Matches your preference for dark solids."}},
  "id_2": {{"id": "id_2", "pattern": "striped", "why_it_fits": "Conflicts with your dislike of busy patterns."}}
}}"""
    else:
        output_spec = """Your task is to produce one user-facing item block per catalog id (in the order
shown below), using the SAME output convention as in this task when you describe
products: each value MUST be a single string containing exactly one
<item><id>...</id><information>...</information></item> block. Put the catalog id
in <id>. Put ALL item-specific detail AND your explanation of fit (why it is a
good or bad match) inside <information>. Do not put item-specific facts outside
that block."""
        example_block = """Return ONLY a JSON object mapping item ids (strings) to strings (each string is one <item> block).
Example:
{{
  "123456": "<item><id>123456</id><information>Blue cotton dress, knee-length. Given you wanted pockets, this is a strong match because ...</information></item>",
  "789012": "<item><id>789012</id><information>Silk blouse. Less aligned with your casual preference because ...</information></item>"
}}"""

    system_msg = f"""You are given a set of candidate items to recommend to a user.
You have already seen the conversation with the user in your existing context.

{output_spec}

Items (in recommended order):
{items_description}

{example_block}

Principles:
- You are an impartial, comprehensive, and honest advisor. Your main job is to provide the user with the information they need to make a decision on what is best for themselves.
- Write to the user in second person, and reflect their preferences and constraints in your response.
- Make sure your descriptions are entirely self-contained.

Include EVERY item. Do not include any other text."""

    if use_item_jsons:

        def _fallback_item_payload(iid: str, note: str) -> str:
            return json.dumps({"id": iid, "note": note}, ensure_ascii=False)

    else:

        def _fallback_item_payload(iid: str, note: str) -> str:
            return f"<item><id>{iid}</id><information>{note}</information></item>"

    raw_last = ""
    for attempt in range(max_retries):
        try:
            prompt = [("system", system_msg)]
            raw_messages = temp_agent.generate(
                dialogs=[prompt],
                persist_state=False,
                remove_thinking_tokens=True,
            )[0]

            from langchain_core.messages import AIMessage

            aimessages = [m for m in raw_messages if isinstance(m, AIMessage)]
            raw = aimessages[-1].content if aimessages else ""
            raw_last = raw

            parsed = parse_json(raw)
            if not isinstance(parsed, dict):
                if attempt < max_retries - 1:
                    print_debug(
                        f"get_final_report_agentic: parsed value is not a dict, retrying... (attempt {attempt + 1}/{max_retries})",
                        "get_final_report_agentic",
                        color="yellow",
                    )
                continue

            cleaned: Dict[str, str] = {}
            for key, value in parsed.items():
                sid = str(key)
                if sid not in valid_id_set:
                    continue
                if isinstance(value, (dict, list)):
                    cleaned[sid] = json.dumps(value, ensure_ascii=False)
                else:
                    cleaned[sid] = str(value)

            if not cleaned or len(cleaned) != len(valid_ids):
                if attempt < max_retries - 1:
                    print_debug(
                        f"get_final_report_agentic: empty mapping after filtering ids, retrying... (attempt {attempt + 1}/{max_retries})",
                        "get_final_report_agentic",
                        color="yellow",
                    )
                continue

            # Ensure every candidate id has at least some text (fallback if omitted).
            for iid in valid_ids:
                if iid not in cleaned:
                    cleaned[iid] = _fallback_item_payload(iid, "N/A")

            return cleaned, {}
        except Exception as e:
            if attempt < max_retries - 1:
                print_debug(
                    f"get_final_report_agentic: attempt {attempt + 1} error: {e}",
                    "get_final_report_agentic",
                    color="yellow",
                )
            else:
                return {sid: _fallback_item_payload(sid, "N/A") for sid in valid_ids}, {
                    "error": str(e),
                    "raw": raw_last,
                }

    # Fallback if all attempts failed without raising an exception
    _verbose_note = (
        "I cannot provide a detailed explanation for this item based on the "
        "conversation; please treat the catalog description as the main source of truth."
    )
    fallback_mapping = {
        sid: _fallback_item_payload(sid, _verbose_note) for sid in valid_ids
    }
    return fallback_mapping, {
        "error": "could not parse valid reports after retries",
        "raw": raw_last,
    }
