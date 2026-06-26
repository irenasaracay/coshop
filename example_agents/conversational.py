from typing import List, Tuple, Optional, Dict, Any
import json
import os
import re

from .policy import InteractionPolicy
from coshop.utils.langchain_agent import LangchainAgent
from coshop.utils.misc import print_debug, strip_thinking_tokens
from coshop.data.dataset import Specification
from .langchain_prediction_utils import (
    get_final_predictions_agentic,
    get_final_report_agentic,
)

# Load prompts from shared JSON file
_PROMPTS_FILE = os.path.join(os.path.dirname(
    __file__), "conversational_prompts.json")
with open(_PROMPTS_FILE, "r") as f:
    _PROMPTS = json.load(f)

# Default message-format instructions used for conversational policies and
# simulators. These tell the policy to wrap item ids in <item>...</item> tags.
MSG_FMT_INSTRUCTIONS = (
    "Communicate with the user in natural language. "
    "Whenever you reference a specific catalog item — whether introducing it, answering a "
    "follow-up question about it, or adding new details — you MUST wrap it in item tags:\n"
    "<item><id>id from web results</id><information>FULL ITEM DESCRIPTION</information></item>\n"
    "Every detail the user needs to know about the item must appear inside <information>; "
    "do not mention item-specific details anywhere outside that tag. "
    "This applies even when the user asks a follow-up question about an item you already showed: "
    "answer in prose if you like, but always include a fresh <item> tag with the complete description. "
    "Example: '<item><id>123456</id><information>Blue cotton dress, knee-length, "
    "with pockets and a floral print.</information></item>'."
)

# Alternate message-format instructions for policies that should emit items
# as JSON objects instead of <item>...</item> tags. The policy is expected
# to output strict JSON (double-quoted keys/strings) when describing items,
# for example on a separate line:
#   {"id": 1234, "color": "black", "pattern": "black-and-white floral"}
POLICY_MSG_FMT_INSTRUCTIONS_ITEM_JSON = (
    "Communicate with the user in natural language. "
    "Whenever you reference a specific catalog item — whether introducing it, answering a "
    "follow-up question about it, or adding new details — you MUST include a JSON line for it. "
    "Each JSON line must appear on its own line, contain an 'id' field with the item's id "
    "from the web results, and include all item-specific details as additional keys. "
    "Use double quotes for all keys and string values. Do not put any other text on that line. "
    "Example: {\"id\": 1234, \"color\": \"black\", \"pattern\": \"black-and-white floral\"}.\n"
    "CRITICAL: If the user asks a follow-up question about a specific item, your text answer "
    "is fine, but you MUST still include a JSON line for that item in the same response — "
    "even if you already showed it before. All item-specific details belong in the JSON; "
    "do not reference them only in prose."
)

# Standalone instructions when --use_structured_actions is True: replaces
# MSG_FMT_INSTRUCTIONS and POLICY_MSG_FMT_INSTRUCTIONS_ITEM_JSON (mutually exclusive).
# Items for user feedback use SHOW_ITEM_FOR_FEEDBACK / ITEM_FOLLOWUP lines only —
# no <item> tags or per-line item JSON from the legacy formats.
MSG_FMT_STRUCTURED_DIALOG_ACTIONS = (
    "Communicate with the user in natural language when appropriate, but you MUST encode "
    "specific dialog actions as separate lines using this exact format: "
    "one keyword, a single space, then one JSON object (double-quoted keys and strings).\n\n"
    "1) ASK_QUESTION — ask the user about a preference or constraint.\n"
    'ASK_QUESTION {"question": "What is your budget?", "relevant_features": ["price"]}\n'
    "relevant_features: catalog feature names this question is about (may be multiple).\n\n"
    "2) SHOW_ITEM_FOR_FEEDBACK — proactively present an item for the user to react to.\n"
    'SHOW_ITEM_FOR_FEEDBACK {"item_id": "12345", "features_for_feedback": {"color": "navy", "material": "cotton"}}\n'
    "item_id: id from search/web results. features_for_feedback: map from catalog feature "
    "name to the value you are showing the user (all item-specific details must appear here).\n\n"
    "3) ITEM_FOLLOWUP — use this whenever you answer a user question about a specific item, "
    "or need to surface additional details about an item you already showed. "
    "Your prose answer is fine, but you MUST also emit this action so the item details are captured.\n"
    'ITEM_FOLLOWUP {"item_id": "12345", "features_for_feedback": {"color": "navy", "material": "cotton"}}\n'
    "Use the same field names as SHOW_ITEM_FOR_FEEDBACK. Include all item-specific details "
    "the user needs inside features_for_feedback — not in the surrounding prose.\n\n"
    "4) EXPLAIN — a paragraph or more teaching the user about a feature or concept.\n"
    'EXPLAIN {"explanation_text": "Dress length usually means...", "relevant_features": ["dress_length"]}\n'
    "relevant_features: catalog feature names the explanation addresses.\n\n"
    "Actions can be freely combined within the same message. For example, you can show an item "
    "highlighting certain features with SHOW_ITEM_FOR_FEEDBACK, then immediately follow it with "
    "an ASK_QUESTION about one of those features — this lets you ground a question in something "
    "concrete the user just saw. Similarly, an EXPLAIN can be paired with an ASK_QUESTION to "
    "teach and then probe in a single turn.\n\n"
    "Rules:\n"
    "- One action per line; freely mix action lines with normal prose on other lines.\n"
    "- If you ask a question in prose without ASK_QUESTION, the user cannot respond to it.\n"
    "- If you discuss an item without SHOW_ITEM_FOR_FEEDBACK or ITEM_FOLLOWUP, "
    "the item details will not be captured.\n"
    "- All feature names must exactly match catalog column names."
)


class RawLLM(LangchainAgent, InteractionPolicy):
    """
    Base class for conversational policies with shared elicitation logic.
    Subclasses implement different get_final_predictions methods.
    """

    def __init__(
        self,
        *args,
        model_name: str = "gpt-5-nano",
        model_kwargs: dict = {},
        max_react_steps: int = 25,
        spec: Optional[Specification] = None,
        msg_fmt_instructions: str = None,
        budget_turns: Optional[int] = None,
        budget_questions: Optional[int] = None,
        budget_unique_items: Optional[int] = None,
        elicitation_system_message_template: Optional[str] = None,
        final_recommendation_system_message_template: Optional[str] = None,
        **kwargs,
    ):
        # MRO: RawLLM -> LangchainAgent -> InteractionPolicy -> Agent
        # Use super() to follow MRO properly and avoid double initialization.
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            max_react_steps=max_react_steps,
            spec=spec,
            msg_fmt_instructions=msg_fmt_instructions,
            *args,
            **kwargs,
        )
        self.budget_turns = budget_turns
        self.budget_questions = budget_questions
        self.budget_unique_items = budget_unique_items
        if self.msg_fmt_instructions:
            self.agent_executor._pre_compress_hook = lambda: self.agent_executor.deduplicate_msgs_by_content({self.msg_fmt_instructions})
        self._elicitation_system_message_template = elicitation_system_message_template
        self._final_recommendation_system_message_template = (
            final_recommendation_system_message_template
        )

    def insert_user_msg(self, user_response: str) -> None:
        """
        Insert a user response into the conversation history without getting a response from the policy.
        """
        self.agent_executor.insert_message(
            role="user", content=user_response, persist_state=True
        )

    def _get_elicitation_template(self) -> str:
        """Return the elicitation system message template (override or default)."""
        if self._elicitation_system_message_template is not None:
            return self._elicitation_system_message_template
        return _PROMPTS["elicitation_system_message_template"]

    def get_elicitation_system_msg(self) -> str:
        """
        System message for the elicitation phase.
        Combines base message with subclass-specific "after conversation" message.
        """
        budget_msg = ""
        if (
            self.budget_turns is not None
            or self.budget_questions is not None
            or self.budget_unique_items is not None
        ):
            budget_turns_line = (
                _PROMPTS["budget_turns_line_template"].format(
                    budget_turns=self.budget_turns if self.budget_turns is not None else "unlimited"
                )
                + ". "
            )
            budget_questions_line = (
                _PROMPTS["budget_questions_line_template"].format(
                    budget_questions=self.budget_questions if self.budget_questions is not None else "unlimited"
                )
                + ". "
            )
            budget_items_line = (
                _PROMPTS["budget_items_line_template"].format(
                    budget_unique_items=self.budget_unique_items if self.budget_unique_items is not None else "unlimited"
                )
                + ". "
            )
            budget_msg = _PROMPTS["budget_message_template"].format(
                budget_turns_line=budget_turns_line,
                budget_questions_line=budget_questions_line,
                budget_items_line=budget_items_line,
            )

        return self._get_elicitation_template().format(
            item_name=self.item_name,
            msg_fmt_instructions=self.msg_fmt_instructions or "",
            budget_msg=budget_msg,
        )

    def _rewrite_ids_with_tags(self, message: str) -> str:
        """Wrap bare item IDs as <id>...</id>, using dataset/catalog-aware rules."""
        if message is None:
            return message
        text = str(message)
        if not text.strip():
            return text

        # Preserve already-tagged ids to avoid double wrapping.
        # Id can be numeric (e.g. HM) or ISBN-style with hyphens (e.g. Goodreads).
        preserved: List[str] = []

        def _preserve_existing(match: re.Match) -> str:
            preserved.append(match.group(0))
            return f"__PRESERVED_ID_TAG_{len(preserved) - 1}__"

        text = re.sub(
            r"<id>\s*[^<]+?\s*</id>",
            _preserve_existing,
            text,
            flags=re.IGNORECASE,
        )

        spec = getattr(self, "spec", None)
        dataset_name = (getattr(spec, "dataset_name", "") or "").lower()
        catalog = getattr(spec, "catalog", None)
        catalog_ids = None
        if catalog is not None and hasattr(catalog, "index"):
            try:
                catalog_ids = {str(i) for i in catalog.index}
            except Exception:
                catalog_ids = None

        # Dataset-aware candidate token regexes.
        if dataset_name == "hm":
            candidate_token_pattern = r"(?<![\w<])(\d{8,10})(?![\w>])"
        elif dataset_name in {"movielens", "goodreads"}:
            candidate_token_pattern = r"(?<![\w<])(\d+)(?![\w>])"
        else:
            candidate_token_pattern = r"(?<![\w<])([A-Za-z0-9_-]{2,32})(?![\w>])"

        if catalog_ids is not None:
            # Only wrap tokens that are actual IDs in this dataset's catalog.
            def _wrap_if_catalog_id(match: re.Match) -> str:
                token = match.group(1)
                return f"<id>{token}</id>" if token in catalog_ids else token

            text = re.sub(candidate_token_pattern, _wrap_if_catalog_id, text)
        else:
            # Conservative fallback: wrap only numbers explicitly labeled as IDs.
            text = re.sub(
                r"(?i)\b(id)\s*[:#-]?\s*(\d+)\b",
                lambda m: f"{m.group(1)} <id>{m.group(2)}</id>",
                text,
            )

        for i, tagged in enumerate(preserved):
            text = text.replace(f"__PRESERVED_ID_TAG_{i}__", tagged)
        return text

    def _generate_message(
        self, user_response: Optional[str] = None
    ) -> Tuple[str, int, float, bool]:
        """
        Generate the next message in the conversation.

        Returns:
            Tuple[str, int, float, bool]: (message, token_cost, runtime_cost, wants_to_end_conversation)
        """
        # If this is the first turn, prepend the generate prompt
        if not self.has_seen_system_prompt:
            system_msg = self.get_elicitation_system_msg()
            prompt = [("system", system_msg)]
            if user_response is not None:
                prompt.append(("user", user_response))
            self.has_seen_system_prompt = True
        else:
            assert user_response is not None, (
                "User response cannot be None if system prompt has already been seen"
            )
            prompt = [("system", self.msg_fmt_instructions), ("user", user_response)]

        if self.verbosity == 2:
            print_debug(
                f"Generating message with prompt:\n{prompt}",
                "_generate_message",
                color="blue",
            )

        # Call generate
        raw, token_cost, runtime_cost = self._call_agent_executor(
            *prompt, persist_state=True
        )
        if raw is None:
            return None, 0, 0.0, False

        raw = strip_thinking_tokens(raw, start_token="<think>", end_token="</think>")
        raw = strip_thinking_tokens(raw, start_token="<thinking>", end_token="</thinking>")

        # Parse for an unquoted <END_CONVERSATION> tag
        # Match <END_CONVERSATION> but not when it's inside quotes or backticks (e.g., "<END_CONVERSATION>", '<END_CONVERSATION>', or `<END_CONVERSATION>`)
        wants_to_end_conversation = (
            re.search(
                r'(?<!["\'`])<END_CONVERSATION[^>]*>(?!["\'`])', raw) is not None
        )
        response = self._rewrite_ids_with_tags(
            raw.replace("<END_CONVERSATION>", "")
        )

        if self.verbosity:
            print_debug(
                f"User message: {user_response}\n\nGenerated policy message: {response}", "_generate_message", color="orange"
            )

        return response, token_cost, runtime_cost, wants_to_end_conversation

    def get_final_predictions(
        self,
        k: int,
        retrieval_function=None,
        execution_max_per_retrieval: Optional[int] = None,
        execution_max_queries: Optional[int] = None,
        execution_global_max: Optional[int] = None,
        max_retries: int = 3,
        min_react_steps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Get the final predictions from the policy after elicitation.
        Produces x_1, ..., x_k (in ranked order) for evaluation.

        Args:
            min_react_steps: Minimum number of react steps to take during agentic prediction

        Returns:
            Tuple[List[str], Dict[str, Any]]: A tuple of (ranked_item_ids, metadata)
                - ranked_item_ids: A list of item IDs in ranked order (x_1, ..., x_k)
                - metadata: A dictionary containing prediction metadata (e.g., search_queries, tool_calls)
        """
        return get_final_predictions_agentic(
            self.agent_executor,
            k,
            retrieval_function,
            execution_max_per_retrieval,
            execution_max_queries,
            execution_global_max,
            max_retries,
            min_react_steps=min_react_steps,
            final_recommendation_system_message_template=(
                self._final_recommendation_system_message_template
            ),
            **kwargs,
        )

    def get_final_report(
        self,
        items: Dict[str, str],
        max_retries: int = 3,
        use_item_jsons: bool = True,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Agentic final report over a set of candidate items.

        Returns a mapping from item id (str) to per-item report text (str).
        """
        report_map, _ = get_final_report_agentic(
            self.agent_executor,
            items,
            max_retries=max_retries,
            use_item_jsons=use_item_jsons,
            **kwargs,
        )
        return report_map

class CoPrefAwareLLM(RawLLM):
    """
    RawLLM variant that appends an SEC-framework awareness addendum to the system
    message, describing how user preferences differ in how accessible they are
    (search / experience / credence) and how to structure the conversation accordingly.
    """

    def get_elicitation_system_msg(self) -> str:
        base_msg = super().get_elicitation_system_msg()
        return base_msg + "\n\n" + _PROMPTS["copref_aware_addendum"]


class CoPrefAwareHistoryLLM(CoPrefAwareLLM):
    """
    CoPrefAwareLLM variant that additionally injects a history DataFrame tool so the
    policy can analyze the user's purchase/rating history — and uses that context to
    calibrate which features are already known (search), surface-able via examples
    (experience), or need explanation (credence).
    """

    def get_elicitation_system_msg(self) -> str:
        base_msg = super().get_elicitation_system_msg()
        return base_msg + "\n\n" + _PROMPTS["copref_aware_history_addendum"]

    def __init__(
        self,
        *args,
        spec=None,
        actions=None,
        **kwargs,
    ):
        actions = list(actions or kwargs.get("actions", []) or [])
        historical_df = getattr(spec, "historical_df", None) if spec else None
        if historical_df is not None and len(historical_df) > 0:
            from coshop.tools.history_df import get_history_df_tool

            try:
                history_tool = get_history_df_tool(historical_df=historical_df)
                actions.insert(0, history_tool)
            except Exception:
                import traceback
                traceback.print_exc()
                print("Error getting history df tool")
        super().__init__(*args, spec=spec, actions=actions, **kwargs)
