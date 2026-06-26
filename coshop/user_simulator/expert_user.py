"""
ExpertUser: simulated user with column-level dropout and a growing known-features set.

Inherits from LangchainAgent and UserSimulator. Prompts are loaded from
simulator_prompts.json.
"""

from __future__ import annotations

import json
import os
from typing import Tuple, Optional, List, Set, Dict, Any

from ..utils.langchain_agent import LangchainAgent
from ..data.dataset import Specification
from .user import UserSimulator
from .helpers.feature_tracker import FeatureTracker, randomly_init_known_features
from .helpers.item_comparer import ItemComparer
from .helpers.question_answerer import QuestionAnswerer
from .helpers.message_parser import (
    MessageParser,
    ClarifyingQuestion,
    Explanation,
    ItemToEval,
)
from .helpers.structured_action_message_parser import (
    StructuredActionMessageParser,
)
from .helpers.feature_utils import normalize_feature_name
from .helpers.final_rank_items import rank_items_agentic
from ..data.utility import ColumnMatchingUtilityFunction
from ..utils.misc import print_debug, parse_for_answer_tags, parse_json
import random


_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "simulator_prompts.json")
with open(_PROMPTS_FILE) as f:
    _PROMPTS = json.load(f)



class ExpertUser(LangchainAgent, UserSimulator):
    """LLM-powered user simulator with SEC-aware feature revelation.

    ``ExpertUser`` simulates a shopper whose preferences are encoded in a
    :class:`~coshop.data.dataset.Specification`.  It uses a LangChain ReAct
    agent to respond to the shopping assistant and a set of helper classes to
    track which features have been revealed, parse the assistant's messages, and
    compare items.

    **Benchmark modes**

    - *Natural language mode* (``use_structured_actions=False``, default):
      The simulator responds in free-form text, including questions and answers.
    - *Structured action mode* (``use_structured_actions=True``):
      The simulator responds with structured actions (parsed by
      :class:`~coshop.user_simulator.helpers.structured_action_message_parser.StructuredActionMessageParser`).

    **Feature revelation**

    Features are drawn from three SEC categories
    (``search_features``, ``experience_features``, ``credence_features``).
    Initially, only ``search`` features that pass through the dataset's dropout
    settings are known.  As the conversation progresses, the simulator may
    reveal additional features up to ``max_features_per_turn`` per turn.

    **Official benchmark settings** (as configured by ``evaluate_agent.py``):
    ``use_structured_actions=False``, ``max_features_per_turn=5``,
    ``early_stop_on_xstar=True``, ``proactive_user=False``,
    ``use_oracle_item_representations=False``, ``use_actual_item_values=False``.
    """

    def __init__(
        self,
        spec: Specification,
        dataset,
        seed: int = 0,
        max_features_per_turn: int = 5,
        early_stop_on_xstar: bool = False,
        search_features: Optional[List[str]] = None,
        experience_features: Optional[List[str]] = None,
        credence_features: Optional[List[str]] = None,
        *args,
        model_name: str = "gpt-5-nano",
        model_kwargs: dict = {},
        max_react_steps: int = 25,
        randomly_choose_initial: bool = False,
        initial_dropout_rate: Optional[float] = None,
        use_oracle_item_representations: bool = False,
        use_actual_item_values: bool = False,
        use_item_jsons: bool = True,
        use_structured_actions: bool = False,
        parser_reasoning_effort: str = "medium",
        max_text_len: Optional[int] = None,
        proactive_user: bool = False,
        **kwargs,
    ):
        """
        Args:
            spec: Task specification containing ``xstar``, z-variants, utility
                function, historical data, and simulator persona.
            dataset: Dataset instance; provides ``catalog``, ``true_features``
                (column descriptions), and ``representation``.
            seed: Random seed (currently unused; reserved for future use).
            max_features_per_turn: Maximum number of new preference features the
                simulator may reveal in a single turn.  Official setting: ``5``.
            early_stop_on_xstar: If ``True``, the simulator stops the
                conversation once it believes ``xstar`` has been found.
                Official setting: ``True``.
            search_features: List of column names classified as *Search* features
                for this episode.  If all three SEC lists are ``None``, all
                feature columns are treated as search features.
            experience_features: List of column names classified as *Experience*
                features (revealed after the user interacts with the product).
            credence_features: List of column names classified as *Credence*
                features (hard-to-verify, revealed last).
            model_name: LLM model name for the ReAct agent and all helpers.
                Defaults to ``"gpt-5-nano"``.
            model_kwargs: Extra keyword arguments forwarded to the LLM client
                (e.g. ``api_base``, ``reasoning_effort``).
            max_react_steps: Maximum ReAct loop iterations per turn.
            randomly_choose_initial: If ``True``, the simulator's initial known
                features are drawn randomly rather than from
                ``spec.initial_known_features``.  Use together with
                ``initial_dropout_rate``.
            initial_dropout_rate: Fraction of features to drop from the initial
                feature set when ``randomly_choose_initial=True``.  ``None``
                means no dropout.
            use_oracle_item_representations: If ``True``, the simulator sees the
                ground-truth (non-vagueified) item representations when
                evaluating items shown by the agent.
            use_actual_item_values: If ``True``, the simulator uses exact feature
                values instead of vagueified ones when describing its preferences.
            use_item_jsons: If ``True``, item representations are formatted as
                JSON dicts instead of the default paragraph format.
            use_structured_actions: If ``True``, use structured-action mode;
                otherwise natural-language mode (official default).
            parser_reasoning_effort: Reasoning effort level for the response
                parsing LLM.  One of ``"low"``, ``"medium"`` (default),
                ``"high"``.
            max_text_len: Maximum character length of item text shown to the
                simulator.  ``None`` disables truncation.
            proactive_user: If ``True``, the simulator proactively hints at
                features it needs even when the agent does not ask.  Official
                setting: ``False``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~coshop.utils.langchain_agent.LangchainAgent`.
        """
        if dataset is None:
            raise ValueError("dataset is required for ExpertUser")

        # The user simulator runs without tools.
        tools: List = []

        LangchainAgent.__init__(
            self,
            actions=tools,
            model_name=model_name,
            model_kwargs=model_kwargs,
            max_react_steps=max_react_steps,
            spec=spec,
            **kwargs,
        )
        self.features_star = spec.xstar_simulator_view
        self._use_structured_actions = use_structured_actions
        parser_cls = (
            StructuredActionMessageParser if use_structured_actions else MessageParser
        )
        self.message_parser = parser_cls(
            model_name=model_name,
            model_kwargs=model_kwargs,
            target_df=self.features_star,
            catalog=dataset.catalog,
            max_features_to_reveal=max_features_per_turn,
            column_descriptions=dataset.true_features,
            verbosity=self.verbosity,
            use_item_jsons=use_item_jsons,
            use_oracle_item_representations=use_oracle_item_representations,
            use_actual_item_values=use_actual_item_values,
            representation=dataset.representation,
            max_text_len=max_text_len,
            parser_reasoning_effort=parser_reasoning_effort,
        )
        if (
            search_features is None
            and experience_features is None
            and credence_features is None
        ):
            _search = list(self.features_star.columns)
            _experience: List[str] = []
            _credence: List[str] = []
        else:
            _search = search_features if search_features is not None else []
            _experience = experience_features if experience_features is not None else []
            _credence = credence_features if credence_features is not None else []
        self.feature_tracker = FeatureTracker(
            target_df=self.features_star,
            search_features=_search,
            experience_features=_experience,
            credence_features=_credence,
            item_name=spec.item_name,
            max_features_to_reveal=max_features_per_turn,
            column_descriptions=dataset.true_features,
            verbosity=self.verbosity,
        )
        self._true_features = dataset.true_features
        self._catalog = dataset.catalog
        self._representation = dataset.representation
        self._use_oracle_item_representations = use_oracle_item_representations
        self._use_item_jsons = use_item_jsons
        self.item_comparer = ItemComparer(
            feature_tracker=self.feature_tracker,
            true_features=self._true_features,
            model_name=model_name,
            model_kwargs=model_kwargs,
            verbosity=self.verbosity,
            catalog=self._catalog,
            representation=self._representation,
            hint_missing_features=proactive_user,
            use_actual_item_values=use_actual_item_values,
        )
        # Initialize known features
        if randomly_choose_initial:
            randomly_init_known_features(
                self.feature_tracker,
                num_features=int(
                    (1 - initial_dropout_rate) * len(self.features_star.columns)
                ),
                dropout_rate=initial_dropout_rate,
            )
        else:
            self.feature_tracker.reveal_features(
                spec.initial_known_features, categories=["search"]
            )

        if self.verbosity > 0:
            print_debug(
                f"known features: {self.feature_tracker.known_features}",
                "ExpertUser.__init__",
            )

        self.spec = spec
        self.item_name = spec.item_name
        self.dataset_name = spec.dataset_name
        self._proactive_user = proactive_user
        self._budget_tracker = None
        self._seen_item_evaluations: Dict[str, Any] = {}
        self._seen_item_ids: Set[str] = set()
        self._last_auto_eval_item_ids: List[str] = []
        self._assistant_messages: List[str] = []
        self._early_stop_on_xstar = early_stop_on_xstar
        self._simulator_persona = getattr(spec, "simulator_persona", "") or ""

        # Question answerer
        self._question_answerer = QuestionAnswerer(
            model_name=model_name,
            model_kwargs=model_kwargs,
            simulator_persona=self._simulator_persona,
            true_features=self._true_features,
            verbosity=self.verbosity,
        )

        # Cache the initial known-features state so we can later simulate
        # rankings from the very start of the conversation.
        self._initial_known_feature_names: List[str] = [
            f.column_name for f in self.feature_tracker.known_features
        ]
        self._initial_z_context: str = self.feature_tracker.get_known_context()

        # Per-turn state for get_state_history: z at end of turn and features added this turn by source.
        self._per_turn_state: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self._seen_item_evaluations = {}
        self._seen_item_ids = set()
        self._last_auto_eval_item_ids = []
        self._assistant_messages = []
        self._per_turn_state = []
        self.item_comparer.clear_feedback_prompt_history()
        self._question_answerer.clear_question_prompt_history()
        self.message_parser.clear_parsed_history()
        self.feature_tracker.clear_reveal_history()
        super().reset()

    def get_state_history(self) -> Dict[str, Any]:
        """Return history of state per turn (z at end of turn, features added by source) plus existing histories."""
        from dataclasses import asdict, is_dataclass

        # Return copies of the list-valued histories so the returned dict is a true
        # snapshot. The driver captures this BEFORE the team-accuracy rerank branches
        # (which mutate reveal_history via the full-state reveal, and the comparer/
        # question-answerer histories via the simulator reranks); without copying,
        # those mutations would leak into the saved state because the lists are
        # exposed by reference. GT avoids this via a restore-on-exit context manager.
        return {
            "message_parser_history": [
                asdict(pm) if is_dataclass(pm) else pm
                for pm in self.message_parser.parsed_history
            ],
            "feature_tracker_history": list(self.feature_tracker.reveal_history),
            "comparer_history": list(self.item_comparer.feedback_prompt_history),
            "question_answerer_history": list(
                self._question_answerer.question_prompt_history
            ),
            "per_turn_state": [
                {
                    "turn": i,
                    "z_end_of_turn": entry["z_end_of_turn"],
                    "features_added": entry["features_added"],
                }
                for i, entry in enumerate(self._per_turn_state)
            ],
        }

    def get_current_z(self) -> str:
        """Return the current preference string built from known_features."""
        return self.feature_tracker.get_known_context()

    def _append_turn_state(self, reveal_history_len_at_turn_start: int) -> None:
        """Append per-turn state: z at end of turn and features added this turn by source."""
        new_reveals = self.feature_tracker.reveal_history[
            reveal_history_len_at_turn_start:
        ]
        features_added = [
            {"feature": e["column_name"], "source": e["source"]} for e in new_reveals
        ]
        self._per_turn_state.append(
            {
                "z_end_of_turn": self.get_current_z(),
                "features_added": features_added,
                # Snapshot of credence queue at end of this turn
                "credence_queue_end_of_turn": list(self.feature_tracker.credence_queue),
            }
        )

    def set_budget_tracker(self, budget_tracker) -> None:
        """Store the budget tracker for checking questions/items limits."""
        self._budget_tracker = budget_tracker

    def _other_budget_exhausted_message(self, default_message: str) -> str:
        """
        If both budgets are exhausted, avoid contradictory instructions
        (e.g., asking for more questions when question budget is already 0).
        """
        if self._budget_tracker is None:
            return default_message

        remaining_questions = self._budget_tracker.get_remaining_questions()
        remaining_items = self._budget_tracker.get_remaining_unique_items()
        if remaining_questions == 0 and remaining_items == 0:
            return (
                "I don't want to answer any more questions or review any more items. "
                "Please move on to the research time now and find me the perfect item."
            )

        return default_message

    def _format_conversation_history(self, limit_msgs: bool = True) -> str:
        """Format conversation_history and _assistant_messages for prompts (first user msg, omitted, last 2 turns)."""
        parts = []
        if not limit_msgs:
            for i in range(
                max(len(self._assistant_messages), len(self.conversation_history))
            ):
                if i < len(self.conversation_history):
                    parts.append(
                        _PROMPTS["conversation_shopper_template"].format(
                            message=self.conversation_history[i].msg
                        )
                    )
                if i < len(self._assistant_messages):
                    parts.append(
                        _PROMPTS["conversation_agent_template"].format(
                            message=self._assistant_messages[i]
                        )
                    )
            return "\n".join(parts)

        # First user msg, then "... (turns omitted) ...", then last agent, last user, last agent
        if len(self.conversation_history) > 0:
            parts.append(
                _PROMPTS["conversation_shopper_template"].format(
                    message=self.conversation_history[0].msg
                )
            )
        if len(self._assistant_messages) > 0 or len(self.conversation_history) > 1:
            parts.append(_PROMPTS["conversation_turns_omitted"])
        if len(self._assistant_messages) >= 2:
            parts.append(
                _PROMPTS["conversation_agent_template"].format(
                    message=self._assistant_messages[-2]
                )
            )
        if len(self.conversation_history) > 1:
            parts.append(
                _PROMPTS["conversation_shopper_template"].format(
                    message=self.conversation_history[-1].msg
                )
            )
        if len(self._assistant_messages) >= 1:
            parts.append(
                _PROMPTS["conversation_agent_template"].format(
                    message=self._assistant_messages[-1]
                )
            )
        return "\n".join(parts)

    def _check_item_recommendations_format(
        self, assistant_msg: str
    ) -> Tuple[bool, int, float]:
        """
        Check if the message has item recommendations in the expected format.
        Returns (ok_to_proceed, token_cost, runtime_cost).
        If ok_to_proceed is False, the message shows item recommendations but without
        the required format (tags or JSON); caller should emit the pre-coded wrap message.
        """
        if not assistant_msg or not assistant_msg.strip():
            return (True, 0, 0.0)
        if self._use_structured_actions:
            for line in assistant_msg.split("\n"):
                line = line.strip()
                if line.startswith("SHOW_ITEM_FOR_FEEDBACK"):
                    return (True, 0, 0.0)
        # Already in expected format?
        if self._use_item_jsons:
            for line in assistant_msg.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                js = parse_json(stripped)
                if isinstance(js, dict) and "id" in js:
                    return (True, 0, 0.0)
        else:
            raw_ids = parse_for_answer_tags(
                assistant_msg,
                keyword="item",
                return_all=True,
                return_none_if_not_found=True,
            )
            raw_id_tags = parse_for_answer_tags(
                assistant_msg,
                keyword="id",
                return_all=True,
                return_none_if_not_found=True,
            )
            if (raw_ids and len(raw_ids) > 0) or (raw_id_tags and len(raw_id_tags) > 0):
                return (True, 0, 0.0)
        # Single LLM check: does the message show item recommendations?
        prompt = """You are a classifier. Given the following message from a customer service agent to a shopper, answer with exactly one word: yes or no.

Question: does this message present or discuss a specific catalog item the user should evaluate?

Classify as "yes" when the agent is either:
- presenting one or more specific candidate items (by id, title/name, or explicit item description) as recommendations, OR
- answering a follow-up question from the user about a specific item it previously recommended (i.e., providing additional details, clarifications, or feature information about that item).

Recommending authors / actors / directors / brands != recommending items.

Classify as "no" when the message does NOT present or discuss a specific item, including:
- saying no good match was found
- discussing constraints/criteria at a high level
- asking the shopper to relax or change requirements
- giving general guidance or uncertainty
- mentioning abstract examples but not recommending them
- giving examples of books by authors they are providing context about

Message:
"""
        prompt += assistant_msg
        response, tc, rc = self._call_agent_executor_no_history(
            ("system", prompt), allow_tools=False
        )
        response_lower = (response or "").strip().lower()
        if response_lower.startswith("yes"):
            return (False, tc, rc)
        return (True, tc, rc)

    def _generate_response(
        self, assistant_msg: Optional[str]
    ) -> Tuple[str, int, float]:
        # First turn: user goes first, single agent call
        if assistant_msg is None:
            response, tc, rc = self._generate_first_turn_response()
            self._per_turn_state.append(
                {
                    "z_end_of_turn": self.get_current_z(),
                    "features_added": [],
                    # Snapshot of credence queue after the first user turn
                    "credence_queue_end_of_turn": list(
                        self.feature_tracker.credence_queue
                    ),
                }
            )
            return response, tc, rc

        self._assistant_messages.append(assistant_msg)
        reveal_history_len_at_turn_start = len(self.feature_tracker.reveal_history)

        # Single LLM check: if message has item recommendations but wrong format, ask to fix
        ok_to_proceed, check_tc, check_rc = self._check_item_recommendations_format(
            assistant_msg
        )
        if not ok_to_proceed:
            if self._use_structured_actions:
                precoded = (
                    "Can you please present items using one line per item: "
                    'SHOW_ITEM_FOR_FEEDBACK {"item_id": "<id>", "features_for_feedback": {...}}?'
                )
            elif self._use_item_jsons:
                precoded = "Can you please format item recommendations as one JSON object per line with an 'id' field?"
            else:
                precoded = "Can you please wrap any informationa bout items in <item><id>...</id><information>...</information></item> tags?"
            self._append_turn_state(reveal_history_len_at_turn_start)
            return precoded, check_tc, check_rc

        # Parse assistant message (oracle: MessageParser builds from catalog, no LLM for items)
        # Provide the current credence queue so the parser can focus explanation
        # extraction on those specific features.
        self.message_parser.credence_queue = list(self.feature_tracker.credence_queue)
        # Gate which catalog columns LOTUS is allowed to consider for the
        # clarifying-question extraction step.
        #
        # We include:
        # - currently known features (across any category)
        # - all search features (so we can ask about missing search constraints)
        # - all credence features
        # (so unknown experience features are skipped)
        question_columns_to_include: List[str] = []
        seen: Set[str] = set()
        for feature in self.feature_tracker.known_features:
            if feature.column_name not in seen:
                seen.add(feature.column_name)
                question_columns_to_include.append(feature.column_name)
        for feature in self.feature_tracker.search_features:
            if feature.column_name not in seen:
                seen.add(feature.column_name)
                question_columns_to_include.append(feature.column_name)
        for feature in self.feature_tracker.credence_features:
            if feature.column_name not in seen:
                seen.add(feature.column_name)
                question_columns_to_include.append(feature.column_name)

        # For explanations, allow search + credence features not yet known to the user.
        # Use a separate seen set so the question_columns seen set doesn't bleed over.
        known_col_names: Set[str] = {
            f.column_name for f in self.feature_tracker.known_features
        }
        explanation_seen: Set[str] = set()
        explanation_columns_to_include: List[str] = []
        for feature in self.feature_tracker.search_features:
            if (
                feature.column_name not in known_col_names
                and feature.column_name not in explanation_seen
            ):
                explanation_seen.add(feature.column_name)
                explanation_columns_to_include.append(feature.column_name)
        for feature in self.feature_tracker.credence_features:
            if (
                feature.column_name not in known_col_names
                and feature.column_name not in explanation_seen
            ):
                explanation_seen.add(feature.column_name)
                explanation_columns_to_include.append(feature.column_name)

        parsed_message = self.message_parser.parse(
            assistant_msg,
            question_columns_to_include=question_columns_to_include,
            explanation_columns_to_include=explanation_columns_to_include,
        )
        if self.verbosity > 0:
            print_debug(
                f"parsed_message: {parsed_message}",
                "ExpertUser._generate_response",
                color="blue",
            )

        # If we have explanations but the credence queue is empty, clear
        n_revealed_this_turn = 0
        credence_to_answer, credence_to_acknowledge, credence_to_ask = [], [], []
        for explanation in parsed_message.explanations:
            _n, _cred, _ack = self.feature_tracker.process_credence_reveals(explanation)
            n_revealed_this_turn += _n
            credence_to_answer.extend(_cred)
            credence_to_acknowledge.extend(_ack)
        self.feature_tracker.clear_credence_queue()

        # Questions
        questions_to_answer: List[ClarifyingQuestion] = []
        credence_questions: List[ClarifyingQuestion] = []
        experience_questions: List[ClarifyingQuestion] = []
        for clarifying_question in parsed_message.clarifying_questions:
            # If the question is asking ONLY about a feature that was just explained, skip it
            if len(
                clarifying_question.relevant_columns
            ) == 1 and clarifying_question.relevant_columns[0] in (
                credence_to_answer + credence_to_acknowledge
            ):
                continue

            _n, _cred = self.feature_tracker.process_question_reveals(
                clarifying_question
            )
            n_revealed_this_turn += _n

            if self._proactive_user:
                if _cred:
                    credence_to_ask.extend(_cred)
                    credence_questions.append(clarifying_question)
                else:
                    # Check if this question is solely about experience features
                    ftypes = self.feature_tracker.get_unknown_feature_types(
                        clarifying_question.relevant_columns
                    )
                    if ftypes["experience"] and not ftypes["search"] and not ftypes["credence"]:
                        experience_questions.append(clarifying_question)
                    else:
                        questions_to_answer.append(clarifying_question)
            else:
                # Non-proactive: answer all questions normally; ignore credence routing
                questions_to_answer.append(clarifying_question)

        # Track newly revealed experience features per item during feedback
        new_reveals_by_item_id: Dict[str, List[str]] = {}
        for item_to_eval in parsed_message.items_to_evaluate:
            _n, newly_revealed_experience = (
                self.feature_tracker.process_feedback_reveals(item_to_eval)
            )
            n_revealed_this_turn += _n
            if newly_revealed_experience:
                new_reveals_by_item_id[str(item_to_eval.id)] = newly_revealed_experience

        # Stitch together response
        cr, cr_token_cost, cr_runtime_cost = self._get_credence_response(
            parsed_message.explanations, credence_to_answer, credence_to_acknowledge
        )
        qr, qr_token_cost, qr_runtime_cost = self._get_question_response(
            questions_to_answer, credence_to_ask, credence_questions
        )
        er, er_token_cost, er_runtime_cost = self._get_experience_question_response(
            experience_questions
        )
        fb, fb_token_cost, fb_runtime_cost, early_return = self._get_feedback(
            parsed_message.items_to_evaluate,
            new_reveals_by_item_id=new_reveals_by_item_id,
        )

        if early_return:
            self._append_turn_state(reveal_history_len_at_turn_start)
            return fb, 0, 0.0

        response = "\n\n".join([cr, qr, er, fb]).strip()

        # Fallback on normal LLM if message doesn't contain structured content
        if response.strip() == "":
            out = self._generate_freeform_response()
            self._append_turn_state(reveal_history_len_at_turn_start)
            return out

        self._append_turn_state(reveal_history_len_at_turn_start)
        total_token_cost = cr_token_cost + qr_token_cost + er_token_cost + fb_token_cost
        total_runtime_cost = cr_runtime_cost + qr_runtime_cost + er_runtime_cost + fb_runtime_cost
        return response, total_token_cost, total_runtime_cost

    def _build_system_msg_from_z(self, z: str) -> str:
        """Build the system message given an explicit preference context z."""
        system_msg = _PROMPTS["simulator_system_message_template"].format(
            item_name=self.item_name,
        )
        role_reminder_key = (
            "role_reminder_json" if self._use_item_jsons else "role_reminder"
        )
        role_reminder = _PROMPTS[role_reminder_key].strip()
        if z:
            z_section = _PROMPTS["simulator_thought_z0_section_template"].format(
                z0=z.strip()
            )
        else:
            z_section = ""
        simulator_persona = (
            self._simulator_persona.strip() + "\n\n" if self._simulator_persona else ""
        )
        instructions = (
            _PROMPTS["simulator_thought_template"]
            .format(simulator_persona=simulator_persona, z0_section=z_section)
            .strip()
        )
        return system_msg + "\n\n" + instructions + "\n\n" + role_reminder

    def _get_system_msg(self) -> str:
        """Get the system message for the first turn using current known features."""
        current_z = self.feature_tracker.get_known_context()
        return self._build_system_msg_from_z(current_z)

    def _generate_first_turn_response(self) -> Tuple[str, int, float]:
        """First turn when user goes first: one agent call with current_z + first_turn_instruction_extra."""
        # Special case: if there are no known features, to prevent hallucination, hard code a message
        if not self.feature_tracker.known_features:
            return (
                "I am looking for a "
                + self.item_name
                + " that matches my preferences.",
                0,
                0.0,
            )

        full_prompt = self._get_system_msg()
        full_prompt += "\n\n" + _PROMPTS["first_turn_instruction_extra"].format(
            z=self.feature_tracker.get_known_context()
        )
        return self._call_agent_executor_no_history(
            ("system", full_prompt), allow_tools=True
        )

    def _generate_freeform_response(self) -> Tuple[str, int, float]:
        full_prompt = self._get_system_msg()

        # Add conversation history
        conversation_history = self._format_conversation_history()
        if conversation_history:
            full_prompt += (
                "\n\nHere is the conversation history:\n\n" + conversation_history
            )
        full_prompt += "\n\n" + "Generate your response."

        return self._call_agent_executor_no_history(
            ("system", full_prompt), allow_tools=True
        )

    def _get_credence_response(
        self,
        explanations: List[Explanation],
        credence_to_answer: List[str],
        credence_to_acknowledge: List[str],
    ) -> Tuple[str, int, float]:
        """
        Generate a response to the agent's credence explanations.
        """
        if not explanations or not (credence_to_answer or credence_to_acknowledge):
            return "", 0, 0.0

        # Prompt the executor to communicate the target values for the features in credence_to_answer
        # and only acknowledge the explanation (but say nothing about the target values) for the features in credence_to_acknowledge
        relevant_z = self.feature_tracker.get_known_context(
            relevant_columns=credence_to_answer
        )
        prompt = _PROMPTS["credence_response_template"].format(
            z=relevant_z,
            credence_to_answer="\n".join(
                [
                    normalize_feature_name(self._true_features.get(f, f))
                    for f in credence_to_answer
                ]
            ),
            credence_to_acknowledge="\n".join(
                [
                    normalize_feature_name(self._true_features.get(f, f))
                    for f in credence_to_acknowledge
                ]
            ),
        )
        return self._call_agent_executor_no_history(("system", prompt))

    def _get_question_response(
        self,
        clarifying_questions: List[ClarifyingQuestion],
        credence_to_ask: List[str],
        credence_questions: Optional[List[ClarifyingQuestion]] = None,
    ) -> Tuple[str, int, float]:
        """Generate a response to the agent's clarifying questions."""
        if not clarifying_questions and not credence_to_ask:
            return "", 0, 0.0

        remaining_questions = (
            None
            if self._budget_tracker is None
            else self._budget_tracker.get_remaining_questions()
        )
        if (
            self._budget_tracker is not None
            and remaining_questions == 0
            and len(clarifying_questions) > 0
        ):
            msg = self._other_budget_exhausted_message(
                default_message=_PROMPTS["questions_budget_exhausted_template"],
            )
            return msg, 0, 0.0

        num_to_answer = (
            len(clarifying_questions)
            if remaining_questions is None
            else min(len(clarifying_questions), remaining_questions)
        )
        questions_to_answer = clarifying_questions[:num_to_answer]
        all_relevant_columns = [
            col
            for q in questions_to_answer
            for col in q.relevant_columns
            if col in self.feature_tracker.known_features
        ]

        z_context = self.feature_tracker.get_known_context()
        feature_values = {}
        for col in all_relevant_columns:
            val = self.feature_tracker.get_value(col)
            feature_values[col] = val if val is not None else ""
        qr, qr_tc, qr_rt = self._question_answerer.answer_questions(
            questions_to_answer, z_context, feature_values=feature_values
        )
        if credence_to_ask:
            # Render each triggering question in the same style as answered questions.
            credence_resp = "\n\n".join(
                f"*{q.question}*: I don't really know what that means."
                for q in credence_questions
            )
            qr = (
                (qr.strip() + "\n\n" + credence_resp.strip()).strip()
                if qr.strip()
                else credence_resp.strip()
            )
        if self._budget_tracker is not None:
            self._budget_tracker.add_questions(num_to_answer)
        if len(clarifying_questions) > num_to_answer:
            qr = (
                qr.strip() + "\n\n" + _PROMPTS["questions_budget_exhausted_template"]
            ).strip()
        return qr, qr_tc, qr_rt

    def _get_experience_question_response(
        self,
        experience_questions: List[ClarifyingQuestion],
    ) -> Tuple[str, int, float]:
        """
        When proactive_user=True, generate a response asking the agent to show examples
        for questions about experience features the user hasn't formed a preference for yet.
        """
        if not experience_questions:
            return "", 0, 0.0
        resp = "\n\n".join(
            f"*{q.question}*: Could you show me a few example options so I can tell you what I like?"
            for q in experience_questions
        )
        return resp, 0, 0.0

    def _get_feedback(
        self,
        items_to_evaluate: List[ItemToEval],
        new_reveals_by_item_id: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[str, int, float, bool]:
        """Generate feedback for items the agent showed via ItemComparer (structured comparison + LOTUS)."""
        if not items_to_evaluate:
            return "", 0, 0.0, False

        if self._early_stop_on_xstar:
            perfect_item = None
            for item in items_to_evaluate:
                if str(item.id) in self.spec.xstar:
                    perfect_item = item
                    break
            if perfect_item:
                return (
                    f"{perfect_item.id} is perfect for me, thanks. You can stop looking now.",
                    0,
                    0.0,
                    True,
                )

        remaining_items = (
            None
            if self._budget_tracker is None
            else self._budget_tracker.get_remaining_unique_items()
        )
        # When budget is set, only evaluate up to remaining_items *new* items (already-seen items don't count).
        items_to_eval_this_turn: List[ItemToEval] = []
        if remaining_items is None:
            items_to_eval_this_turn = list(items_to_evaluate)
        else:
            new_count = 0
            for item in items_to_evaluate:
                is_new = str(item.id) not in self._seen_item_ids
                if is_new and new_count >= remaining_items:
                    continue
                items_to_eval_this_turn.append(item)
                if is_new:
                    new_count += 1

        def feedback_fn(prompt: str) -> Tuple[str, int, float]:
            feedback, token_cost, runtime_cost = self._call_agent_executor_no_history(
                ("system", prompt), allow_tools=False
            )
            return feedback, token_cost, runtime_cost

        id_to_feedback, total_token_cost, total_runtime_cost = (
            self.item_comparer.compute_feedback(
                items_to_eval_this_turn,
                feedback_fn=feedback_fn,
                new_reveals_by_item_id=new_reveals_by_item_id or {},
            )
        )

        num_new_evaled = sum(
            1
            for item in items_to_eval_this_turn
            if str(item.id) not in self._seen_item_ids
        )
        for item in items_to_eval_this_turn:
            self._seen_item_ids.add(str(item.id))

        if self._budget_tracker is not None:
            self._budget_tracker.add_unique_items(num_new_evaled)

        parts = [
            "\n\n".join(
                f"On item {id}: {feedback}" for id, feedback in id_to_feedback.items()
            )
        ]
        if len(items_to_eval_this_turn) < len(items_to_evaluate):
            items_msg = self._other_budget_exhausted_message(
                default_message=_PROMPTS["items_budget_exhausted_template"],
            )
            parts.append(items_msg)
        return "\n\n".join(parts).strip(), total_token_cost, total_runtime_cost, False

    def _rank_items_with_parser(
        self,
        items: Dict[str, str],
        known_feature_names: List[str],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Ranks items by parsing the known features into a vector of true / false / null values.
        Ranks to minimize the number of failed features,
        and then by (# true) / (# total).
        Remaining ties are broken randomly.
        """
        match_by_item, active_columns = (
            self.message_parser.feature_match_item_descriptions(
                items, known_feature_names
            )
        )
        scores: Dict[str, float] = {}
        for item_id in items.keys():
            m = match_by_item.get(item_id, {})
            scores[item_id] = (
                -sum(1 for c in active_columns if m.get(c) is False),
                sum(1 for c in active_columns if m.get(c) is True)
                / len(active_columns),
                random.random(),  # break remaining ties randomly
            )
        ranked_ids = sorted(
            items.keys(),
            key=lambda iid: scores[iid],
            reverse=True,  # higher scores are better
        )
        xstar_scores = {
            item_id: (
                scores[item_id][0],
                scores[item_id][1],
                int(str(item_id) in self.spec.xstar),
            )
            for item_id in items.keys()
        }
        xstar_ranking = sorted(
            items.keys(),
            key=lambda iid: xstar_scores[iid],
            reverse=True,  # higher scores are better
        )
        return ranked_ids, {
            "ranking_if_ties_broken_by_xstar": xstar_ranking,
            "known_features": known_feature_names,
            "active_columns": active_columns,
            "scores": scores,
            "matches": match_by_item,
        }

    def rank_items(
        self,
        items: Dict[str, str],
        mode: str = "agentic",
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Rank a provided list of item_ids using either:
        - ``agentic``: LLM ranks from item text and conversation context.
        - ``parser``: LOTUS feature-match on item descriptions vs known features;
          score = (# True) / (# active columns); ``False`` and ``None`` count as fails.
        - otherwise (e.g. ``rank``): column-based utility (catalog vs xstar) restricted
          to known features.
        """
        if mode == "agentic":
            # make a copy in case we call this method multiple times
            agent_executor_copy = self.agent_executor.copy_for_prediction(
                tools=self.actions
            )
            # add system prompt + conversation history to state
            full_prompt = self._get_system_msg()
            conversation_history = self._format_conversation_history(limit_msgs=False)
            if conversation_history:
                full_prompt += (
                    "\n\nFor context, here is the conversation history:\n\n"
                    + conversation_history
                    + "\n\nHere is your current set of preferences:\n\n"
                    + self.get_current_z()
                )
            agent_executor_copy.insert_message("system", full_prompt)
            return rank_items_agentic(agent_executor_copy, items)
        elif mode == "parser":
            known_feature_names = [
                f.column_name for f in self.feature_tracker.known_features
            ]
            return self._rank_items_with_parser(items, known_feature_names)
        else:
            # Create a ColumnMatchingUtilityFunction using CURRENT known features
            known_feature_names = [
                f.column_name for f in self.feature_tracker.known_features
            ]
            utility_function = ColumnMatchingUtilityFunction(
                xstar=self.spec.xstar_series,
                catalog=self._catalog,
                cols_to_compare=known_feature_names,
            )
            scores = utility_function(list(items.keys()))
            ranked_ids = [
                id
                for id, score in sorted(
                    zip(items.keys(), scores), key=lambda x: x[1], reverse=True
                )
            ]
            return ranked_ids, {
                "known_features": known_feature_names,
                "scores": scores,
            }

    def rank_items_initial_state(
        self,
        items: Dict[str, str],
        mode: str = "agentic",
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Rank items as the simulator would have at the very start of the
        conversation, using only the initial known features / initial system
        message (i.e. before any clarifications or feedback updates).

        Same ``mode`` values as :meth:`rank_items` (``agentic``, ``parser``, or
        catalog ``rank``).
        """
        if mode == "agentic":
            # Copy executor and seed it with an initial-state system message only
            agent_executor_copy = self.agent_executor.copy_for_prediction(
                tools=self.actions
            )
            full_prompt = self._build_system_msg_from_z(self._initial_z_context)
            agent_executor_copy.insert_message("system", full_prompt)
            return rank_items_agentic(agent_executor_copy, items)
        elif mode == "parser":
            return self._rank_items_with_parser(
                items, self._initial_known_feature_names
            )
        else:
            # Use only the initially known features for the column-based utility
            utility_function = ColumnMatchingUtilityFunction(
                xstar=self.spec.xstar_series,
                catalog=self._catalog,
                cols_to_compare=self._initial_known_feature_names,
            )
            scores = utility_function(list(items.keys()))
            ranked_ids = [
                id
                for id, score in sorted(
                    zip(items.keys(), scores), key=lambda x: x[1], reverse=True
                )
            ]
            return ranked_ids, {
                "known_features": self._initial_known_feature_names,
                "scores": scores,
            }
