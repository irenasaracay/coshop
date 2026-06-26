"""Base Agent class for all conversational participants in coshop.

Provides conversation-history tracking, checkpoint save/restore, and a
hook system (pre/post-generation) shared by both the shopping-assistant
policy and the user simulator.
"""

from typing import List, Dict, Literal, Optional, Union, Callable, Any, Tuple
from collections import defaultdict
from dataclasses import asdict
from .custom_types import ConversationTurn, Action
from .misc import print_debug
import os
import json


DEFAULT_POST_GENERATION_HOOKS = [
    "get_state",
]


class Agent:
    """
    Base class for agents that participate in conversations.
    Provides common conversation tracking functionality.
    """

    def __init__(
        self,
        verbosity: Literal[0, 1, 2] = 0,
        pre_conversation_hooks: List[Union[str, Callable]] = [],
        pre_generation_hooks: List[Union[str, Callable]] = [],
        post_generation_hooks: List[
            Union[str, Callable]
        ] = DEFAULT_POST_GENERATION_HOOKS,
        checkpoint_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the agent.

        Args:
            verbosity: Whether to print verbose output. 0: print nothing, 1: print function outputs but not prompts, 2: print everything
            pre_conversation_hooks: Hooks to run before processing the first input message (only runs once)
            pre_generation_hooks: Hooks to run before generating response
            post_generation_hooks: Hooks to run after generating response
            checkpoint_file: File to save checkpoints. If None, checkpointing is disabled.
        """
        self.verbosity = verbosity
        self._pre_conversation_hooks = pre_conversation_hooks
        self._pre_generation_hooks = pre_generation_hooks
        self._post_generation_hooks = post_generation_hooks
        self.checkpoint_file = checkpoint_file

        # Track if we've seen the first input message (for pre_conversation_hooks)
        self._has_seen_first_input: bool = False

        # State tracking
        self.conversation_history: List[ConversationTurn] = []
        self.action_history: Dict[int, List[Action]] = defaultdict(list)
        self.hook_history: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.lock: bool = False

    def get_conversation_history(self) -> List[dict]:
        """
        Get the conversation history as a list of dicts.

        Returns:
            List[dict]: List of conversation turns as dictionaries
        """
        return [asdict(turn) for turn in self.conversation_history]

    def get_action_history(self) -> Dict[int, List[Dict]]:
        """
        Get the action history.

        Returns:
            Dict[int, List[Dict]]: Dictionary mapping turn numbers to lists of action dictionaries
        """
        return {
            k: [asdict(action) for action in actions]
            for k, actions in self.action_history.items()
        }

    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """
        self.conversation_history = []
        self.action_history = defaultdict(list)
        self.hook_history = defaultdict(dict)
        self.lock = False
        self._has_seen_first_input = False

    def run_hooks(self, hooks: List[Union[str, Callable]], *args, **kwargs) -> None:
        """
        Run the hooks.

        Args:
            hooks: List of hooks to run (can be strings or callables)
        """
        for hook in hooks:
            if self.verbosity == 2:
                print_debug(f"Running hook {hook}", "run_hooks", color="blue")
            out = self._run_hook(hook, *args, **kwargs)
            if out is None:
                continue
            if not isinstance(out, dict):
                out = {hook: out}
            self.hook_history[self.turn_count].update(out)

    def _run_hook(self, hook: Union[str, Callable], *args, **kwargs) -> Any:
        """
        Run a single hook.

        Args:
            hook: Hook to run (string or callable)

        Returns:
            Result of the hook execution
        """
        if isinstance(hook, str):
            # Try to get the method from the class
            if hasattr(self, hook):
                return getattr(self, hook)(*args, **kwargs)
            elif hook == "get_state":
                return self._get_state(*args, **kwargs)
            elif hook == "save_checkpoint":
                self.save_checkpoint(*args, **kwargs)
                return {}
            else:
                if self.verbosity:
                    print_debug(
                        f"Warning: Hook '{hook}' not found", "run_hooks", color="yellow"
                    )
                return None
        else:
            # It's a callable
            return hook(self, *args, **kwargs)

    def _get_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get the current state of the agent for checkpointing.
        Subclasses can override this to add additional state.

        Returns:
            Dictionary containing the agent's state
        """
        state = {
            "conversation_history": self.get_conversation_history(),
            "action_history": self.get_action_history(),
            "turn_count": self.turn_count,
        }
        return state

    @property
    def turn_count(self) -> int:
        """
        Index of the current turn in the conversation.

        Returns:
            int: The index of the current turn
        """
        return len(self.conversation_history)

    def __call__(self, input_msg: Optional[str] = None) -> str:
        """
        Process an input message and generate a response.
        This is the standardized entry point for all agents.

        Args:
            input_msg: The input message (e.g., user message for policy, assistant message for user simulator)

        Returns:
            str: The generated response message
        """
        # Check if agent is locked
        if self.lock:
            print("Agent is locked")
            return ""

        # Run pre-conversation hooks (only on first input message)
        if input_msg is not None and not self._has_seen_first_input:
            self.run_hooks(self._pre_conversation_hooks, input_msg=input_msg)
            self._has_seen_first_input = True

        # Run pre-generation hooks
        self.run_hooks(self._pre_generation_hooks, input_msg=input_msg)

        # Generate response (subclass implements _generate_response)
        response_result = self._generate_response(input_msg)

        # Handle different return formats
        if isinstance(response_result, tuple):
            if len(response_result) == 3:
                msg, token_cost, runtime_cost = response_result
                extra = {}
            elif len(response_result) == 4:
                msg, token_cost, runtime_cost, extra_value = response_result
                extra = {"extra": extra_value}
            else:
                msg, token_cost, runtime_cost = (
                    response_result[0],
                    response_result[1],
                    response_result[2],
                )
                extra = {f"extra_{i}": v for i, v in enumerate(response_result[3:])}
        else:
            # Fallback if _generate_response returns just a string
            msg = response_result
            token_cost = 0
            runtime_cost = 0.0
            extra = {}

        # Optional debug output
        if self.verbosity:
            print_debug(
                f"Runtime cost: {runtime_cost}s, Token cost: {token_cost}",
                "__call__",
                color="green",
            )

        # Get actions for this turn
        turn_actions = self.action_history.get(self.turn_count, [])

        # Create conversation turn and append to history
        self.conversation_history.append(
            ConversationTurn(
                msg=msg,
                actions=turn_actions,
                runtime_cost=runtime_cost,
                token_cost=int(token_cost),
            )
        )

        # Store any extra return values
        for key, value in extra.items():
            setattr(self, key, value)

        # Run post-generation hooks
        self.run_hooks(self._post_generation_hooks, input_msg=input_msg, msg=msg)

        return msg

    def _generate_response(
        self, input_msg: Optional[str]
    ) -> Union[str, Tuple[str, int, float], Tuple[str, int, float, Any]]:
        """
        Generate a response to the input message.
        Subclasses must implement this method.

        Args:
            input_msg: The input message

        Returns:
            Can return:
            - str: Just the message
            - Tuple[str, int, float]: (message, token_cost, runtime_cost)
            - Tuple[str, int, float, Any]: (message, token_cost, runtime_cost, extra_value)

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement _generate_response")

    ######## CHECKPOINTING ##########

    def save_checkpoint(self, connection=None) -> None:
        """
        Save the current state of the agent to a checkpoint file.
        Critically, this does not save configs for the agent, so the agent later must be initialized with the same configs as the checkpoint.

        Args:
            connection: Optional connection object for remote file access
        """
        if self.checkpoint_file is None:
            raise ValueError(
                "Checkpoint file not set. Set checkpoint_file in __init__ to enable checkpointing."
            )
        os.makedirs(os.path.dirname(self.checkpoint_file) or ".", exist_ok=True)

        # Get base state
        checkpoint_data = {
            "hook_history": self.hook_history,
            "conversation_history": self.get_conversation_history(),
            "action_history": self.get_action_history(),
            "turn_idx": self.turn_count,
        }

        # Get additional state from subclass (e.g., agent_executor_state, spec_state)
        additional_state = self._get_checkpoint_state()
        checkpoint_data.update(additional_state)

        # Save checkpoint
        if connection is not None:
            connection.write(
                self.checkpoint_file, json.dumps(checkpoint_data, indent=2)
            )
        else:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

        if self.verbosity >= 1:
            print(f"Checkpoint saved to {self.checkpoint_file}")

    def load_checkpoint(
        self,
        checkpoint_file: Optional[str] = None,
        turn_idx: Optional[int] = None,
        connection=None,
    ) -> None:
        """
        Load the agent state from a checkpoint file.
        Critically, this does not save configs for the agent, so the agent must be initialized with the same configs as the checkpoint.

        Relies on the hook history to load the state by turn.

        Args:
            checkpoint_file: Path to checkpoint file. If None, uses self.checkpoint_file.
            turn_idx: If provided, only load the state up to this turn. If None, load the entire checkpoint.
            connection: Optional connection object for remote file access
        """
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_file
        if checkpoint_file is None:
            raise ValueError("No checkpoint file specified")

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        if connection is not None:
            checkpoint_data = json.loads(connection.read(checkpoint_file))
        else:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

        # Correct string keys to int keys
        checkpoint_data["hook_history"] = {
            int(k): v for k, v in checkpoint_data["hook_history"].items()
        }
        checkpoint_data["action_history"] = {
            int(k): v for k, v in checkpoint_data["action_history"].items()
        }
        checkpoint_data["turn_idx"] = int(checkpoint_data["turn_idx"])

        if turn_idx is not None:
            # Restore a single turn based on the hook history
            assert turn_idx in checkpoint_data["hook_history"], (
                f"No hook history available for turn {turn_idx}"
            )
            turn_data = checkpoint_data["hook_history"][turn_idx]
            self._restore_full_state(
                conversation_history=turn_data.get("conversation_history", []),
                action_history=turn_data.get("action_history", {}),
                hook_history={
                    k: v
                    for k, v in checkpoint_data["hook_history"].items()
                    if k <= turn_idx
                },
                **self._extract_checkpoint_state(turn_data),
            )
        else:
            # Use the last available state
            self._restore_full_state(
                conversation_history=checkpoint_data["conversation_history"],
                action_history=checkpoint_data["action_history"],
                hook_history=checkpoint_data["hook_history"],
                **self._extract_checkpoint_state(checkpoint_data),
            )
            turn_idx = checkpoint_data["turn_idx"]

        if self.verbosity >= 1:
            print(
                f"Checkpoint loaded from {checkpoint_file}, currently at the start of turn {turn_idx}"
            )

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get additional state to include in checkpoint.
        Subclasses should override this to add agent-specific state (e.g., agent_executor_state, spec_state).

        Returns:
            Dictionary of additional state to save
        """
        return {}

    def _extract_checkpoint_state(
        self, checkpoint_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract additional state from checkpoint data for restoration.
        Subclasses should override this to extract agent-specific state.

        Args:
            checkpoint_data: The checkpoint data dictionary

        Returns:
            Dictionary of state to pass to _restore_full_state
        """
        return {}

    def _restore_full_state(
        self,
        *,
        conversation_history: List[dict],
        action_history: Dict[int, List[dict]],
        hook_history: Dict[int, Dict[str, Any]],
        **kwargs,
    ) -> None:
        """
        Restore the state of the agent based on the checkpoint data.

        Args:
            conversation_history: List of conversation turns as dictionaries
            action_history: Dictionary mapping turn numbers to lists of action dictionaries
            hook_history: Dictionary mapping turn numbers to hook state dictionaries
            **kwargs: Additional state to restore (subclass-specific)
        """
        # Restore conversation history - convert dictionaries back to dataclasses
        self.conversation_history = [
            ConversationTurn(**turn_dict) for turn_dict in conversation_history
        ]

        # Restore action history - convert dictionaries back to dataclasses
        self.action_history = defaultdict(list)
        for turn_num, actions_list in action_history.items():
            for action_dict in actions_list:
                self.action_history[int(turn_num)].append(Action(**action_dict))

        # Restore hook history as defaultdict for consistency with __init__/reset
        self.hook_history = defaultdict(dict, hook_history)

        # Subclasses can override to restore additional state
        self._restore_additional_state(**kwargs)

    def _restore_additional_state(self, **kwargs) -> None:
        """
        Restore additional state specific to the subclass.
        Subclasses should override this method.

        Args:
            **kwargs: Additional state to restore
        """
        pass
