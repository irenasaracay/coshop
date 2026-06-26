"""Agentic item ranking for user-simulator evaluation.

Provides :func:`rank_items_agentic`, which asks a copy of the ReAct agent
to rank a set of catalog items from best to worst, and
:func:`create_temp_agent`, which clones an agent with modified tools and
step limits for the ranking step without altering the original agent state.
"""

from typing import Any, Tuple, List, Dict, Optional, Callable

from ...utils.misc import parse_json, print_debug
from ...utils.model import LangChainModel


def create_temp_agent(
    original_agent: LangChainModel,
    retrieval_function: Optional[Callable] = None,
    execution_max_per_retrieval: Optional[int] = None,
    execution_max_queries: int = None,
    execution_global_max: Optional[int] = None,
    min_react_steps: int = None,
    max_react_steps: int = None,
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
    return temp_agent


def rank_items_agentic(
    original_agent: LangChainModel,
    items: Dict[str, str],
    max_retries: int = 2,
    temperature: float = 0.0,
    **kwargs,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Ask the agent to rank items (by id) from best to worst using a temp agent with no query function.

    Args:
        original_agent: The agent executor to use (state is copied to temp agent).
        items: Dict mapping item id to string representation of that item.
        max_retries: Maximum number of retries for getting a valid list of item ids.

    Returns:
        (ordered_ids, metadata): List of item ids from best to worst, and a dict with
            "error", "raw", and/or "attempt" on failure; empty metadata on success.
    """
    if not isinstance(max_retries, int) or max_retries < 1:
        max_retries = 1

    temp_agent = create_temp_agent(
        original_agent=original_agent,
        retrieval_function=None,
        temperature=temperature,
        **kwargs,
    )

    valid_ids = set(items.keys())
    items_description = "\n".join(
        f"- id: {iid}\n  item: {repr(text)}" for iid, text in items.items()
    )

    ranking_system_msg = f"""You are given a set of items. Your task is to rank them from best to worst for the user, based on the conversation and your understanding of their preferences.

Items:
{items_description}

Return a JSON array of item ids in order from best to worst. Use only the ids listed above. Example: ["id_1", "id_2", "id_3"]
You may return a JSON object with a key "ranking" or "order" containing that array instead.
Do not include any other text or explanation.
"""

    raw = ""
    for attempt in range(max_retries):
        try:
            prompt = [("system", ranking_system_msg)]
            raw_messages = temp_agent.generate(
                dialogs=[prompt],
                persist_state=False,
                remove_thinking_tokens=True,
            )[0]

            from langchain_core.messages import AIMessage

            aimessages = [m for m in raw_messages if isinstance(m, AIMessage)]
            raw = aimessages[-1].content if aimessages else ""

            parsed = parse_json(raw)
            ordered_ids: Optional[List[str]] = None

            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(x, str) for x in parsed):
                    ordered_ids = [str(x) for x in parsed]
            elif isinstance(parsed, dict):
                for key in ("ranking", "order", "item_ids", "ids"):
                    if key in parsed and isinstance(parsed[key], list):
                        ordered_ids = [str(x) for x in parsed[key]]
                        break

            if ordered_ids is not None and ordered_ids:
                # Restrict to valid ids and preserve order; drop duplicates (first occurrence wins)
                seen = set()
                result = []
                for iid in ordered_ids:
                    if iid in valid_ids and iid not in seen:
                        seen.add(iid)
                        result.append(iid)
                if result:
                    return result, {}

            if attempt < max_retries - 1:
                print_debug(
                    f"rank_items_agentic: could not parse valid ranking (attempt {attempt + 1}/{max_retries}). raw={raw[:200]}...",
                    "rank_items_agentic",
                    color="yellow",
                )
        except Exception as e:
            if attempt < max_retries - 1:
                print_debug(
                    f"rank_items_agentic: attempt {attempt + 1} error: {e}",
                    "rank_items_agentic",
                    color="yellow",
                )
            return [], {"error": str(e), "raw": raw}

    return [], {"error": "could not parse valid ranking after retries", "raw": raw}

