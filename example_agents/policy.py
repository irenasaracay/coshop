from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Literal,
    Any,
)
from coshop.data.dataset import Specification
from coshop.utils.agent import Agent


class InteractionPolicy(Agent):
    """
    An abstract class for an assistant
    """

    def __init__(
        self,
        spec: Optional[Specification] = None,
        msg_fmt_instructions: str = None,
        verbosity: Literal[0, 1, 2] = 0,
        checkpoint_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the interaction policy.

        Args:
            spec: used for hooks and checkpointing only, not generation or prediction. If provided, the policy's save_checkpoint method will automatically call the specification's get_state method
            msg_fmt_instructions: Optional formatting instructions appended to the policy's messages
            verbosity: Whether to print verbose output. 0: print nothing, 1: print function outputs but not prompts, 2: print everything
            checkpoint_file: File to save checkpoints. If None, checkpointing is disabled.
            **kwargs: Additional keyword arguments forwarded to the base Agent
        """
        super().__init__(
            verbosity=verbosity,
            checkpoint_file=checkpoint_file,
            **kwargs,
        )
        self.spec = spec
        self.item_name = spec.item_name
        self.msg_fmt_instructions = msg_fmt_instructions or ""

        # State tracking
        self.has_seen_system_prompt: bool = False
        self.wants_to_end_conversation: bool = False

    def reset(self) -> None:
        """
        Reset the policy to its initial state.
        """
        super().reset()
        self.wants_to_end_conversation = False
        self.has_seen_system_prompt = False

    def insert_user_msg(self, user_response: str) -> None:
        """
        Insert a user response into the conversation history without getting a response from the policy.
        """
        pass

    ######## MAIN METHODS ##########

    def _generate_response(
        self, user_response: Optional[str]
    ) -> Tuple[str, int, float, bool]:
        """
        Generate a response to the user message.
        Calls _generate_message and returns the result.

        Args:
            user_response: The user's response message

        Returns:
            Tuple[str, int, float, bool]: (message, token_cost, runtime_cost, wants_to_end_conversation)
        """
        assistant_msg, token_cost, runtime_cost, wants_to_end_conversation = (
            self._generate_message(user_response)
        )
        # Store wants_to_end_conversation as an extra attribute
        self.wants_to_end_conversation = wants_to_end_conversation
        return assistant_msg, token_cost, runtime_cost, wants_to_end_conversation

    def _generate_message(
        self, user_response: Optional[str] = None
    ) -> Tuple[str, int, float, bool]:
        """
        Generate the next message in the conversation.
        Returns:
            Tuple[str, int, float, bool]:
                - str: The next message in the conversation
                - int: The token cost of the response
                - float: The runtime cost of the response
                - bool: Whether the assistant wants to end the conversation

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_final_predictions(
        self,
        k: int,
        retrieval_function=None,
        execution_max_per_retrieval: Optional[int] = None,
        execution_max_queries: Optional[int] = None,
        execution_global_max: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Get the final predictions from the policy after elicitation.
        This method should produce x_1, ..., x_k (in ranked order) for evaluation.

        Args:
            k: Number of items to return in ranked order
            retrieval_function: The query function to use (if needed for predictions)
            execution_max_per_retrieval: Maximum items per query during execution. If None and execution_global_max is set, items will be split across queries.
            execution_max_queries: Maximum number of queries to execute (default: None, no limit)
            execution_global_max: Maximum total items that can be retrieved across all query calls (default: None, no limit). If set and execution_max_per_retrieval is None, items will be split across queries.

        Returns:
            Tuple[List[str], Dict[str, Any]]: A tuple of (ranked_item_ids, metadata)
                - ranked_item_ids: A list of item IDs in ranked order (x_1, ..., x_k)
                - metadata: A dictionary containing prediction metadata (e.g., search_queries, tool_calls)

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_final_report(
        self,
        items: Dict[str, str],
        use_item_jsons: bool = False,
    ) -> Dict[str, str]:
        """
        Ask the policy to write a final report over a set of candidate items.

        Returns a mapping from item id (str) to per-item report text (str).

        use_item_jsons: Hint for agentic implementations so the final-report prompt
            matches JSON vs plain-text item formatting (ignored by this default).

        Default implementation is a safe fallback that echoes the input
        representations. Policies that maintain a LangChain agent executor can
        override this to use an agentic helper.
        """
        return {str(iid): str(text) for iid, text in items.items()}

    ######## CHECKPOINTING ##########

    def _get_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get the current state including policy-specific state.
        """
        state = super()._get_state(*args, **kwargs)
        if hasattr(self, "agent_executor"):
            state["agent_executor_state"] = self.agent_executor.get_state()
        if self.spec is not None:
            state["specification_state"] = self.spec.get_state()
        if hasattr(self, "wants_to_end_conversation"):
            state["wants_to_end_conversation"] = self.wants_to_end_conversation
        return state

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get additional state to include in checkpoint (spec_state, wants_to_end_conversation).
        """
        state = super()._get_checkpoint_state()
        if self.spec is not None:
            state["specification_state"] = self.spec.get_state()
        if hasattr(self, "wants_to_end_conversation"):
            state["wants_to_end_conversation"] = self.wants_to_end_conversation
        return state

    def _extract_checkpoint_state(
        self, checkpoint_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract policy-specific state from checkpoint data.
        """
        state = super()._extract_checkpoint_state(checkpoint_data)
        if "specification_state" in checkpoint_data:
            state["specification_state"] = checkpoint_data["specification_state"]
        if "wants_to_end_conversation" in checkpoint_data:
            state["wants_to_end_conversation"] = checkpoint_data[
                "wants_to_end_conversation"
            ]
        return state

    def _restore_additional_state(
        self,
        specification_state: Optional[Dict[str, Any]] = None,
        wants_to_end_conversation: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Restore policy-specific state (spec_state, wants_to_end_conversation).
        """
        super()._restore_additional_state(**kwargs)
        if specification_state is not None and self.spec is not None:
            self.spec.load_state(specification_state)
        if wants_to_end_conversation is not None:
            self.wants_to_end_conversation = wants_to_end_conversation
