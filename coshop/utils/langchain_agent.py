"""LangChain-based agent implementation for coshop.

Extends :class:`~coshop.utils.agent.Agent` with a LangChain ReAct executor,
token/runtime tracking, and helper methods for copying agent state to temporary
agents (used during final item ranking).
"""

from typing import Tuple, Any, Literal, Dict, Optional, List
from langchain_core.tools import StructuredTool
from .misc import Stopwatch, strip_thinking_tokens
from .model import (
    LangChainModel,
    get_token_usage,
    get_token_breakdown,
    is_openai_model,
    is_anthropic_model,
    is_gemini_model,
)
from .custom_types import Action, ToolCall
from langchain_core.messages import AIMessage
from .agent import Agent


class LangchainAgent(Agent):
    """
    Base class for agents that use LangChain models.
    Provides common LangChain functionality.
    """

    def __init__(
        self,
        actions: List[StructuredTool] = [],
        model_name: str = "gpt-5-nano",
        model_kwargs: dict = {"reasoning_effort": "minimal"},
        max_react_steps: int = 25,
        verbosity: Literal[0, 1, 2] = 0,
        **kwargs,
    ):
        """
        Initialize the LangChain agent.

        Args:
            actions: List of tools/actions available to the agent (must be set before calling this)
            model_name: Name of the model to use
            model_kwargs: Additional keyword arguments for the model
            max_react_steps: Maximum number of ReAct steps
            verbosity: Whether to print verbose output
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(verbosity=verbosity, **kwargs)

        is_anthropic = is_anthropic_model(model_name)
        is_hf = (not is_openai_model(model_name)) and (
            not is_anthropic_model(model_name)
        ) and not is_gemini_model(model_name)
        if is_anthropic:
            # omit seed
            print("Warning: seed is not supported for Anthropic models")
            model_kwargs.pop("seed", None)
        self.agent_executor = LangChainModel(
            model_name=model_name,
            tools=actions,
            verbosity=self.verbosity,
            max_react_steps=max_react_steps,
            multiturn_memory=True,
            out_of_steps_msg="Sorry, I need some more time to think about this. Please give me the go-ahead to think some more.",
            list_tools_in_prompt=is_hf,
            add_thinking_tag=not is_hf,
            empty_message_filler="Hmm, I'm thinking...",
            **model_kwargs,
        )
        self.actions = actions
        self._is_hf = is_hf
        self._model_name = model_name
        self._model_kwargs = model_kwargs
        self._max_react_steps = max_react_steps

    @property
    def cumulative_token_breakdown(self) -> Dict[str, int]:
        """
        Cumulative token usage across every LLM call this agent makes —
        conversation turns, final-prediction / report calls, and summarizer /
        compression calls. Counted at the model layer (LangChainModel) and read
        through here; prediction/report agents share the executor's counter, so
        their usage is included automatically. Keys: "input", "input_cached",
        "output", "reasoning".
        """
        if not hasattr(self, "agent_executor"):
            return {"input": 0, "input_cached": 0, "output": 0, "reasoning": 0}
        return self.agent_executor.cumulative_token_breakdown

    def _call_agent_executor(
        self,
        *msgs: Tuple[Tuple[str, str], ...],
        persist_state: bool = True,
        **kwargs: Any,
    ) -> Tuple[str, int, float]:
        """
        Call the agent executor and return the raw response, token cost, and runtime cost.

        Args:
            msgs: The new messages to append to the chain
                msgs[i] = (role, content)
            persist_state: Whether to persist state across calls
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[str, int, float]: The final response, token cost, and runtime cost
        """
        with Stopwatch() as sw:
            # This method automatically handles out of steps errors & null prompts
            # Keep thinking tokens in state; we strip only from the response sent to the user
            raw = self.agent_executor.generate(
                dialogs=[msgs],
                persist_state=persist_state,
                remove_thinking_tokens=False,
                **kwargs,
            )[0]

        # Look at the new messages and extract the tool calls for saving
        action_history = parse_langchain_response_to_actions(raw)

        # Token usage (including any summarizer/compression calls made during
        # this generate) is counted inside LangChainModel; read the per-turn
        # output(+reasoning) cost it recorded. 
        total_token_cost = self.agent_executor.last_call_token_cost

        turn_index = len(self.conversation_history)
        self.action_history[turn_index].extend(action_history)

        # Anthropic models sometimes return lists of dicts in the 'content' field
        output = action_history[-1].content
        if isinstance(output, list):
            output = output[-1]
            output = output.get("text")

        # Strip thinking tokens only from the response we send to the user
        thinking_tokens = getattr(
            self.agent_executor, "_thinking_tokens", ("<think>", "</think>")
        )
        output = strip_thinking_tokens(
            output, thinking_tokens[0], thinking_tokens[1]
        )

        return (
            output,
            total_token_cost,
            sw.time,
        )

    def _call_agent_executor_no_history(self, *msgs: Tuple[Tuple[str, str], ...], allow_tools: bool = True, **kwargs: Any) -> Tuple[str, int, float]:
        """
        Call the agent executor but clear the history first. Does not persist state.
        """
        # Create a copy of the original agent with new tools and optional step limits
        temp_agent = self.agent_executor.copy_for_prediction(
            tools=self.actions if allow_tools else [],
            max_react_steps=self._max_react_steps,
        )
        with Stopwatch() as sw:
            # This method automatically handles out of steps errors & null prompts
            # Keep thinking tokens in state; we strip only from the response sent to the user
            raw = temp_agent.generate(
                dialogs=[msgs],
                persist_state=False,
                remove_thinking_tokens=False,
                **kwargs,
            )[0]

        # Look at the new messages and extract the tool calls for saving
        action_history = parse_langchain_response_to_actions(raw)

        # Usage is counted inside LangChainModel. temp_agent shares this agent's
        # cumulative_token_breakdown (see copy_for_prediction), so the breakdown
        # is already updated; read the per-call cost from temp_agent.
        total_token_cost = temp_agent.last_call_token_cost

        turn_index = len(self.conversation_history)
        self.action_history[turn_index].extend(action_history)

        # Anthropic models sometimes return lists of dicts in the 'content' field
        output = action_history[-1].content
        if isinstance(output, list):
            output = output[-1]
            output = output.get("text")

        # Strip thinking tokens only from the response we send to the user
        thinking_tokens = getattr(
            self.agent_executor, "_thinking_tokens", ("<think>", "</think>")
        )
        output = strip_thinking_tokens(
            output, thinking_tokens[0], thinking_tokens[1]
        )

        return (
            output,
            total_token_cost,
            sw.time,
        )

    def reset(self) -> None:
        """Reset the agent to its initial state."""
        super().reset()
        if hasattr(self, "agent_executor"):
            self.agent_executor.clear_state()
            self.agent_executor.reset_token_tracking()

    def _get_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get the current state of the agent including LangChain-specific state.
        """
        state = super()._get_state(*args, **kwargs)
        if hasattr(self, "agent_executor"):
            state["agent_executor_state"] = self.agent_executor.get_state()
        return state

    def _get_checkpoint_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get additional state to include in checkpoint (agent_executor_state).
        """
        state = super()._get_checkpoint_state(*args, **kwargs)
        if hasattr(self, "agent_executor"):
            state["agent_executor_state"] = self.agent_executor.get_state()
        return state

    def _extract_checkpoint_state(
        self, checkpoint_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract LangChain-specific state from checkpoint data.
        """
        state = super()._extract_checkpoint_state(checkpoint_data)
        if "agent_executor_state" in checkpoint_data:
            state["agent_executor_state"] = checkpoint_data["agent_executor_state"]
        return state

    def _restore_additional_state(
        self, agent_executor_state: Optional[str] = None, **kwargs
    ) -> None:
        """
        Restore LangChain-specific state (agent_executor_state).
        If this agent has has_seen_system_prompt (e.g. InteractionPolicy), set it
        to True when a system prompt is at the top of the restored executor state.
        """
        super()._restore_additional_state(**kwargs)
        if agent_executor_state is not None and hasattr(self, "agent_executor"):
            self.agent_executor.load_state(agent_executor_state)
            # Sync has_seen_system_prompt for policies that use it
            if hasattr(self, "has_seen_system_prompt"):
                try:
                    state = getattr(self.agent_executor, "state", None)
                    if state and len(state) > 0:
                        from langchain_core.messages import SystemMessage

                        if isinstance(state[0], SystemMessage):
                            self.has_seen_system_prompt = True
                except Exception:
                    pass

    def insert_message(
        self, role: str, content: str, persist_state: bool = True
    ) -> None:
        """
        Insert a message into the conversation history without making an LLM call.

        Args:
            role: Message role ("system", "user", "assistant", or "tool")
            content: Message content
            persist_state: If True, appends the message to the graph state
        """
        if hasattr(self, "agent_executor"):
            self.agent_executor.insert_message(
                role=role, content=content, persist_state=persist_state
            )




def parse_langchain_response_to_actions(raw: List[Any]) -> List[Action]:
    """
    Parse the raw response from a LangChain model into a list of Action objects.

    Args:
        raw: List of BaseMessage objects from LangChain

    Returns:
        List of Action objects with content, tool_calls, and status
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # Look at the new messages and extract the tool calls for saving
    action_history = []
    for i, msg in enumerate(raw):
        if not isinstance(msg, AIMessage):
            continue

        # collect all associated tool call info
        tool_call_kwargs, tool_call_names, tool_call_responses, statuses = (
            [],
            [],
            [],
            [],
        )
        # Track processed tool call IDs to avoid duplicates
        processed_tool_call_ids = set()

        # OpenAI / Anthropic tool call parsing
        for tool_call in getattr(msg, "additional_kwargs", {}).get("tool_calls", []):
            id = tool_call["id"]
            # Skip if we've already processed this tool call ID
            if id in processed_tool_call_ids:
                continue
            processed_tool_call_ids.add(id)

            kwargs = tool_call["function"].get("arguments")
            if kwargs is None:
                kwargs = tool_call["function"].get("args")
            name = tool_call["function"]["name"]

            # look ahead for response
            tool_response = None
            status = "error"
            for j in range(i + 1, len(raw)):
                if isinstance(raw[j], ToolMessage) and raw[j].tool_call_id == id:
                    tool_response = raw[j].content
                    status = getattr(raw[j], "status", "success")
                    break

            tool_call_kwargs.append(kwargs)
            tool_call_names.append(name)
            tool_call_responses.append(tool_response)
            statuses.append(status)

        # HuggingFace tool call parsing
        for tool_call in getattr(msg, "tool_calls", []):
            id = tool_call["id"]
            # Skip if we've already processed this tool call ID
            if id in processed_tool_call_ids:
                continue
            processed_tool_call_ids.add(id)

            kwargs = tool_call["args"]
            name = tool_call["name"]

            tool_response = None
            status = "error"
            for j in range(i + 1, len(raw)):
                if isinstance(raw[j], ToolMessage) and raw[j].tool_call_id == id:
                    tool_response = raw[j].content
                    status = getattr(raw[j], "status", "success")
                    break

            tool_call_kwargs.append(kwargs)
            tool_call_names.append(name)
            tool_call_responses.append(tool_response)
            statuses.append(status)

        tool_calls: List[ToolCall] = [
            {
                "name": name,
                "kwargs": kwargs,
                "response": response,
                "status": status,
            }
            for kwargs, name, response, status in zip(
                tool_call_kwargs, tool_call_names, tool_call_responses, statuses
            )
        ]

        # append result to action history
        action_history.append(
            Action(
                content=msg.content,
                tool_calls=tool_calls,
                status=(
                    "success"
                    if any(status == "success" for status in statuses)
                    or len(tool_calls) == 0
                    else "error"
                ),
            )
        )

    return action_history
