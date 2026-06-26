"""Type definitions and I/O helpers for elicitation interactions.

Defines the dataclass hierarchy used to record, serialise, and reload complete
elicitation episodes:

- :class:`ToolCall` — a single tool invocation within an agent action.
- :class:`Action` — one step of a ReAct loop (content + tool calls).
- :class:`ConversationTurn` — a full message turn with runtime/token costs.
- :class:`ElicitationTurn` — paired (user, policy) turn with per-side costs.
- :class:`ItemEvaluationInfo` — per-item utility scores for one evaluation.
- :class:`SeenEvaluation` / :class:`PredictedEvaluation` / :class:`Evaluation`
  — structured evaluation results.
- :class:`ElicitationInteraction` — top-level container for a complete episode.

Utility functions:

- :func:`save_elicitation_interaction` — build and persist an
  :class:`ElicitationInteraction` to JSON.
- :func:`load_interaction` — load a saved interaction from JSON.
- :func:`conversation_history_to_messages` — flatten turns into a
  ``[{"role": ..., "content": ...}]`` message list.
"""

from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import List, Dict, Any, Optional, Literal, Union, TypedDict, Tuple
import json
import os
from .misc import _clean_for_json

END_REASONS = Literal[
    "budget_exhausted",
    "policy_end",
    "user_end",
    "unknown",
    "oracle_simulator",
]


class ToolCall(TypedDict):
    """Represents a tool call in an action"""

    name: str
    kwargs: str
    response: str
    status: Literal["success", "error"]


@dataclass
class Action:
    """Represents an action in a conversation turn"""

    content: str  # Content of the action
    tool_calls: List[ToolCall]  # Associated tool calls
    status: Literal["success", "error"]  # Status of the action


@dataclass
class ConversationTurn:
    """Represents a turn in a conversation"""

    msg: str  # The message (user or assistant)
    actions: List[Action]  # Actions taken
    runtime_cost: float  # Total runtime cost (includes time to do actions)
    token_cost: int  # Total token cost (includes tokens from actions)


@dataclass
class ElicitationTurn:
    """Represents a single turn in an interaction"""

    user_msg: str
    user_actions: List[Dict[str, Any]]
    user_token_cost: Optional[float]
    user_runtime_cost: Optional[float]
    policy_msg: str
    policy_actions: List[Dict[str, Any]]
    policy_token_cost: Optional[float]
    policy_runtime_cost: Optional[float]


@dataclass
class ItemEvaluationInfo:
    """Information about a single item in evaluation"""

    id: str
    column_ustar: float
    embedding_ustar: float
    em: bool  # Exact match
    # Unified ustar (e.g., ColumnMatchWithServerCosinePercentileUtilityFunction)
    # Defaults to None for backward compatibility with older interaction logs.
    ustar: Optional[float] = None


@dataclass
class SeenEvaluation:
    """Evaluation metrics for seen items"""

    seen_ids: List[ItemEvaluationInfo]
    max_column_ustar: Optional[float]
    avg_column_ustar: Optional[float]
    max_embedding_ustar: Optional[float]
    avg_embedding_ustar: Optional[float]
    recall_at_seen: Optional[float]
    # Defaults to None for backward compatibility.
    max_ustar: Optional[float] = None
    avg_ustar: Optional[float] = None


@dataclass
class PredictedEvaluation:
    """Evaluation metrics for predicted items"""

    predicted_ids: List[ItemEvaluationInfo]
    max_column_ustar_at_k: Optional[float]
    avg_column_ustar_at_k: Optional[float]
    max_embedding_ustar_at_k: Optional[float]
    avg_embedding_ustar_at_k: Optional[float]
    max_ustar_at_k: Optional[float]
    avg_ustar_at_k: Optional[float]
    column_ustar_at_1: Optional[float]
    embedding_ustar_at_1: Optional[float]
    ustar_at_1: Optional[float]
    ndcg_column_ustar_at_k: Optional[float]
    ndcg_embedding_ustar_at_k: Optional[float]
    ndcg_ustar_at_k: Optional[float]
    ndcg_binary_at_k: Optional[float]
    recall_at_k: Optional[float]
    recall_at_1: Optional[float]


@dataclass
class Evaluation:
    """Represents evaluation results for an elicitation process with seen and predicted sections"""

    seen: SeenEvaluation
    predicted: PredictedEvaluation
    execution_max_per_retrieval: Optional[
        int
    ]  # Number of items queried per query (formerly 'm')
    execution_max_queries: Optional[int]  # Maximum number of queries during execution
    execution_global_max: Optional[
        int
    ]  # Maximum total items across all queries during execution
    k: int  # Top-k items evaluated


@dataclass
class ElicitationInteraction:
    """Represents a complete elicitation interaction and its evaluation"""

    turns: List[ElicitationTurn]  # Elicitation conversation history
    config: Dict[str, Any]
    budget_metrics: Dict[str, Any]  # Tracked budget metrics (turns, runtime, tokens)
    end_reason: END_REASONS
    filename: str
    retrieval_type: str  # Query function class name
    corrupt_representations: bool  # Whether to corrupt the representations
    policy: str  # Policy class name
    policy_model: str  # Policy model name
    representation_type: str = "paragraph"  # Representation key used
    spec_information: Dict[str, Any] = field(default_factory=dict)
    elicitation_evaluation: Optional[Evaluation] = None
    research_evaluation: Optional[Evaluation] = None
    elicitation_simulator_evaluation_oracle: Optional[Evaluation] = None
    elicitation_simulator_evaluation_oracle_initial_state: Optional[Evaluation] = None
    elicitation_simulator_evaluation_reports: Optional[Evaluation] = None
    elicitation_simulator_evaluation_reports_initial_state: Optional[Evaluation] = None
    research_simulator_evaluation_oracle: Optional[Evaluation] = None
    research_simulator_evaluation_oracle_initial_state: Optional[Evaluation] = None
    research_simulator_evaluation_reports: Optional[Evaluation] = None
    research_simulator_evaluation_reports_initial_state: Optional[Evaluation] = None
    policy_checkpoint_file: Optional[str] = None
    simulator_checkpoint_file: Optional[str] = None
    final_prediction_metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Metadata from final predictions (e.g., search_queries, tool_calls)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


def _check_config(config: dict):
    """
    Check that the config is valid.
    """
    required_keys = {
        "dataset",
        "dataset_kwargs",
        "spec_index",
        "policy",
        "policy_model",
        "policy_kwargs",
        "simulator",
        "simulator_model",
        "simulator_kwargs",
        "seed",
        "corrupt_representations",
        "representation_type",
        "retrieval_type",
        "execution_max_per_retrieval",
        "k",
    }
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config must include {key}")


###############################################


def save_elicitation_interaction(
    *,
    simulator,
    policy,
    output_path: str,
    spec,
    config: dict,
    end_reason: str,
    budget_metrics: Dict[str, Any],
    elicitation_evaluation: Optional[Evaluation] = None,
    research_evaluation: Optional[Evaluation] = None,
    connection=None,
    **kwargs,
) -> ElicitationInteraction:
    """
    Save the elicitation interaction and evaluation results.

    Args:
        simulator: The simulator instance
        policy: The policy instance
        output_path: Path to save results
        config: Configuration
        end_reason: The reason for the end of the interaction
        budget_metrics: Budget tracking metrics
        elicitation_evaluation: Elicitation evaluation results
        research_evaluation: Research evaluation results
        spec: Specification instance
        **kwargs: Additional arguments
    Returns:
        An ElicitationInteraction instance
    """
    _check_config(config)

    # Extract turn info
    user_conversation_history = (
        simulator.get_conversation_history() if simulator is not None else []
    )
    policy_conversation_history = (
        policy.get_conversation_history() if policy is not None else []
    )
    num_turns = max(len(user_conversation_history), len(policy_conversation_history))

    def _get_turn_info(
        ix: int,
        conversation_history: List[dict],
    ) -> Tuple[str, List[Dict[str, Any]], float, float]:
        """
        Get the turn information from the conversation history.
        """
        msg = (
            conversation_history[ix]["msg"] if ix < len(conversation_history) else None
        )
        actions = (
            conversation_history[ix].get("actions", [])
            if ix < len(conversation_history)
            else []
        )
        token_cost = (
            conversation_history[ix].get("token_cost")
            if ix < len(conversation_history)
            else None
        )
        runtime_cost = (
            conversation_history[ix].get("runtime_cost")
            if ix < len(conversation_history)
            else None
        )
        return msg, actions, token_cost, runtime_cost

    # Build turns from conversation history
    turns = []
    for i in range(num_turns):
        user_msg, user_actions, user_token_cost, user_runtime_cost = _get_turn_info(
            i, user_conversation_history
        )
        (
            policy_msg,
            policy_actions,
            policy_token_cost,
            policy_runtime_cost,
        ) = _get_turn_info(i, policy_conversation_history)
        turn = ElicitationTurn(
            user_msg=user_msg,
            user_actions=user_actions,
            user_token_cost=user_token_cost,
            user_runtime_cost=user_runtime_cost,
            policy_msg=policy_msg,
            policy_actions=policy_actions,
            policy_token_cost=policy_token_cost,
            policy_runtime_cost=policy_runtime_cost,
        )
        turns.append(turn)

    # Build spec information
    spec_information = {}
    if spec is not None:
        spec_information = {
            "dataset_name": spec.dataset_name,
            "index": spec.index,
            "dataset_kwargs": config.get("dataset_kwargs", {}),
            "item_name": spec.item_name,
            "z0": spec.z0,
            "zs": spec.zs,
            "zse": spec.zse,
            "zstar": spec.zstar,
            "simulator_persona": spec.simulator_persona,
            "historical_data": spec.historical_data,
            "xstar": spec.xstar,
            "sec_split": spec.sec_split,
        }

    # Only save checkpoint files if they exist
    policy_checkpoint_file = policy.checkpoint_file if policy is not None else None
    if policy_checkpoint_file is not None and not os.path.exists(
        policy_checkpoint_file
    ):
        policy_checkpoint_file = None
    simulator_checkpoint_file = (
        simulator.checkpoint_file if simulator is not None else None
    )
    if simulator_checkpoint_file is not None and not os.path.exists(
        simulator_checkpoint_file
    ):
        simulator_checkpoint_file = None

    # Create the interaction object
    interaction = ElicitationInteraction(
        turns=turns,
        config=config,
        budget_metrics=budget_metrics,
        end_reason=end_reason,
        filename=os.path.basename(output_path),
        retrieval_type=config["retrieval_type"],
        corrupt_representations=config["corrupt_representations"],
        policy=config["policy"],
        policy_model=config["policy_model"],
        spec_information=spec_information,
        elicitation_evaluation=elicitation_evaluation,
        research_evaluation=research_evaluation,
        elicitation_simulator_evaluation_oracle=kwargs.get(
            "elicitation_simulator_evaluation_oracle"
        ),
        elicitation_simulator_evaluation_oracle_initial_state=kwargs.get(
            "elicitation_simulator_evaluation_oracle_initial_state"
        ),
        elicitation_simulator_evaluation_reports=kwargs.get(
            "elicitation_simulator_evaluation_reports"
        ),
        elicitation_simulator_evaluation_reports_initial_state=kwargs.get(
            "elicitation_simulator_evaluation_reports_initial_state"
        ),
        research_simulator_evaluation_oracle=kwargs.get(
            "research_simulator_evaluation_oracle"
        ),
        research_simulator_evaluation_oracle_initial_state=kwargs.get(
            "research_simulator_evaluation_oracle_initial_state"
        ),
        research_simulator_evaluation_reports=kwargs.get(
            "research_simulator_evaluation_reports"
        ),
        research_simulator_evaluation_reports_initial_state=kwargs.get(
            "research_simulator_evaluation_reports_initial_state"
        ),
        policy_checkpoint_file=policy_checkpoint_file,
        simulator_checkpoint_file=simulator_checkpoint_file,
        final_prediction_metadata=kwargs.get("final_prediction_metadata", {}),
    )

    # Save to file
    out = {**asdict(interaction), **kwargs}
    out = _clean_for_json(out)

    try:
        if connection is not None:
            connection.write(output_path, json.dumps(out, indent=2))
        else:
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error saving interaction to {output_path}: {e}")

    return interaction


def _convert_to_dataclass(data: Dict[str, Any], target_class: type) -> Any:
    """
    Recursively convert a dictionary to a dataclass instance.
    Uses dataclasses.is_dataclass() to detect and convert nested dataclasses.

    Args:
        data: Dictionary to convert
        target_class: The dataclass type to convert to
    Returns:
        An instance of the target dataclass with all nested dataclasses properly converted
    """
    if data is None:
        return None

    # Handle lists - recursively convert each item if it's a dict
    if isinstance(data, list):
        # For lists, we need to determine the type of items in the list
        if hasattr(target_class, "__origin__") and target_class.__origin__ is list:
            item_type = target_class.__args__[0]
            return [
                (
                    _convert_to_dataclass(item, item_type)
                    if isinstance(item, dict)
                    else item
                )
                for item in data
            ]
        return data

    # Handle dictionaries - recursively convert nested dicts to their appropriate dataclass types
    if isinstance(data, dict):
        # Get the field names and types from the dataclass
        field_types = target_class.__annotations__
        field_names = {field.name for field in fields(target_class)}

        # Filter out unknown keys
        filtered_data = {k: v for k, v in data.items() if k in field_names}

        # For Interaction dataclass, capture unknown keys in extra_metadata
        if target_class == ElicitationInteraction:
            unknown_keys = {k: v for k, v in data.items() if k not in field_names}
            if unknown_keys:
                filtered_data["extra_metadata"] = unknown_keys

        # Fill missing fields with None
        for field_name in field_names:
            if field_name not in filtered_data:
                filtered_data[field_name] = None

        # Convert each field according to its type
        for field_name, field_value in filtered_data.items():
            field_type = field_types[field_name]

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Get the first non-None type from Optional[T]
                field_type = next(t for t in field_type.__args__ if t is not type(None))

            # Convert based on field type
            if isinstance(field_value, dict):
                if is_dataclass(field_type):
                    filtered_data[field_name] = _convert_to_dataclass(
                        field_value, field_type
                    )
            elif isinstance(field_value, list):
                if (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and is_dataclass(field_type.__args__[0])
                ):
                    filtered_data[field_name] = [
                        (
                            _convert_to_dataclass(item, field_type.__args__[0])
                            if isinstance(item, dict)
                            else item
                        )
                        for item in field_value
                    ]

        # Create the dataclass instance
        return target_class(**filtered_data)

    return data


def load_interaction(
    path: str,
    connection=None,
) -> ElicitationInteraction:
    """
    Load the results of an interaction evaluation from a file as an ElicitationInteraction object.
    Properly reconstructs all dataclass objects from JSON data.
    """
    try:
        # use connection if provided, otherwise fall back to direct file operations
        if connection is not None:
            data = connection.read(path)
        else:
            with open(path, "r") as f:
                data = json.load(f)
        return _convert_to_dataclass(data, ElicitationInteraction)
    except Exception as e:
        print(f"Error loading interaction from {path}: {e}")
        raise e


def conversation_history_to_messages(
    conversation_history: List[ElicitationTurn],
) -> List[Dict[str, Any]]:
    """
    Convert a list of Turn objects to a list of messages in
    {"role": "assistant" | "user", "content": str, "response_time": float, "tool_calls": List[ToolCall]}
    format.
    """
    messages = []
    for turn in conversation_history:
        if turn.user_msg is not None:
            # Extract tool calls from user actions
            user_tool_calls = []
            for action in turn.user_actions:
                tool_calls = action.get("tool_calls", [])
                user_tool_calls.extend(tool_calls)

            messages.append(
                {
                    "role": "user",
                    "content": turn.user_msg,
                    "response_time": turn.user_runtime_cost,
                    "tool_calls": user_tool_calls,
                }
            )
        if turn.policy_msg is not None:
            # Extract tool calls from policy actions
            policy_tool_calls = []
            for action in turn.policy_actions:
                tool_calls = action.get("tool_calls", [])
                policy_tool_calls.extend(tool_calls)

            messages.append(
                {
                    "role": "assistant",
                    "content": turn.policy_msg,
                    "response_time": turn.policy_runtime_cost,
                    "tool_calls": policy_tool_calls,
                }
            )
    return messages
