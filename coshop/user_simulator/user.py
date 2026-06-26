"""Abstract base class for user simulators.

Defines the interface shared by all user simulator implementations:
- :meth:`UserSimulator._generate_response`: produce the next turn given the assistant's message.
- :meth:`UserSimulator.rank_items`: rank a list of item IDs from best to worst.
- :meth:`UserSimulator.set_budget_tracker`: attach a :class:`~coshop.evaluation.budget.BudgetTracker`.

Checkpoint / state serialisation is handled via :class:`~coshop.utils.agent.Agent`
with an extra ``specification_state`` key added by this class.
"""

from typing import Tuple, Literal, Optional, Dict, Any, List

from ..data.dataset import Specification
from ..utils.agent import Agent


class UserSimulator(Agent):
    """Abstract base class for all user simulators.

    A user simulator plays the role of the shopper in a conversational
    recommendation episode.  Concrete subclasses must implement
    :meth:`_generate_response` (produce the next utterance given the agent's
    message) and :meth:`rank_items` (rank candidate items from best to worst).

    The simulator holds a :class:`~coshop.data.dataset.Specification` which
    encodes the user's ground-truth preferences (``xstar``, z-variants, utility
    function, etc.) and exposes ``item_name`` and ``simulator_persona`` for
    prompt construction.
    """

    def __init__(
        self,
        spec: Optional[Specification] = None,
        verbosity: Literal[0, 1, 2] = 0,
        checkpoint_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the user simulator.

        Args:
            spec: The specification containing reward / validation function (can be passed as positional or keyword arg)
            verbosity: Whether to print verbose output. 0: print nothing, 1: print function outputs but not prompts, 2: print everything
            checkpoint_file: File to save checkpoints
            **kwargs: Additional keyword arguments
        """
        # Allow spec to be passed as keyword argument for multiple inheritance compatibility
        if spec is None:
            spec = kwargs.pop("spec", None)
        if spec is None:
            raise TypeError(
                "UserSimulator.__init__() missing required argument: 'spec'"
            )

        super().__init__(verbosity=verbosity, checkpoint_file=checkpoint_file, **kwargs)

        # Task information
        self.spec = spec
        self.item_name = spec.item_name
        self.simulator_persona = getattr(spec, "simulator_persona", "")

    ######## MAIN METHODS ##########

    def set_budget_tracker(self, budget_tracker) -> None:
        """
        Set the budget tracker for simulators that support budget-aware behavior.
        No-op by default; subclasses (e.g. ZstarSingleTurnBudgeted) override to store it.
        """
        pass

    def _generate_response(
        self, assistant_msg: Optional[str]
    ) -> Tuple[str, int, float]:
        """
        Process a message from the assistant and generate a response.

        Args:
            assistant_msg: The message from the assistant

        Returns:
            Tuple[str, int, float]: A tuple containing:
                - str: The user's response
                - int: The token cost of the response
                - float: The runtime cost of the response

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get the current state including user simulator-specific state (spec_state).
        """
        state = super()._get_state(*args, **kwargs)
        if self.spec is not None:
            state["specification_state"] = self.spec.get_state()
        return state

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get additional state to include in checkpoint (spec_state).
        """
        state = super()._get_checkpoint_state()
        if self.spec is not None:
            state["specification_state"] = self.spec.get_state()
        return state

    def _extract_checkpoint_state(
        self, checkpoint_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract user simulator-specific state from checkpoint data.
        """
        state = super()._extract_checkpoint_state(checkpoint_data)
        if "specification_state" in checkpoint_data:
            state["specification_state"] = checkpoint_data["specification_state"]
        return state

    def _restore_additional_state(
        self, specification_state: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """
        Restore user simulator-specific state (spec_state).
        """
        super()._restore_additional_state(**kwargs)
        if specification_state is not None and self.spec is not None:
            self.spec.load_state(specification_state)

    def rank_items(
        self,
        items: Dict[str, str],
        mode: Literal["rank", "agentic", "parser"] = "agentic",
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Rank a provided mapping of item ID -> item text from best to worst,
        according to the simulator's notion of user preferences.

        Subclasses must implement this; the base class only defines the
        interface so that downstream evaluation code can call
        simulator.rank_items(...) in a uniform way.

        Returns:
            A ``(ranked_ids, metadata)`` tuple where ``ranked_ids`` orders the
            keys of ``items`` from best to worst and ``metadata`` holds
            mode-specific details (e.g. scores, known features).
        """
        raise NotImplementedError(
            "Subclasses must implement rank_items for reranking evaluation"
        )

    def rank_items_initial_state(
        self,
        items: Dict[str, str],
        mode: Literal["rank", "agentic", "parser"] = "agentic",
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Rank a provided mapping of item ID -> item text using the simulator's
        initial state (i.e., before any interaction has taken place).

        By default, this falls back to the current-state ranking behavior
        so existing simulators continue to work without modification.
        Subclasses can override this to provide true initial-state ranking.
        """
        return self.rank_items(items, mode=mode, **kwargs)
