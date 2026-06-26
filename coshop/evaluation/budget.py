"""Budget tracking for elicitation episodes.

Tracks per-episode consumption across five dimensions (turns, runtime, tokens,
questions, unique items) and signals when any configured limit is exceeded.
"""

from typing import Tuple, Optional, Dict, Any


class BudgetTracker:
    """Tracks multiple budget dimensions and signals when any limit is exhausted.

    Each budget dimension is optional — pass ``None`` (the default) to impose no
    limit on that dimension.  Budgets are checked in ``is_exhausted()``, which
    returns a ``(bool, reason_str)`` pair.

    Runtime and token costs are *weighted* before accumulation to allow
    asymmetric penalization of policy vs. user simulator costs:

    - ``weighted_runtime = policy_budget_weight * policy_runtime + user_budget_weight * user_runtime``
    - ``weighted_tokens  = policy_budget_weight * policy_tokens  + user_budget_weight * user_tokens``

    Set a weight to ``0.0`` to ignore that component entirely (e.g. exclude user
    simulator costs from the token budget).

    Attributes:
        budget_turns: Maximum number of conversation turns allowed.
        budget_runtime: Maximum weighted wall-clock seconds allowed.
        budget_tokens: Maximum weighted token count allowed.
        budget_questions: Maximum clarifying questions the user simulator may ask.
        budget_unique_items: Maximum distinct catalog items the agent may surface.
        policy_budget_weight: Multiplier applied to policy-side runtime/token costs.
        user_budget_weight: Multiplier applied to user-simulator runtime/token costs.
        current_turns: Turns consumed so far.
        current_runtime: Weighted seconds consumed so far.
        current_tokens: Weighted tokens consumed so far (policy + user).
        current_policy_tokens: Raw policy token count (unweighted).
        current_user_tokens: Raw user-simulator token count (unweighted).
        current_questions: Clarifying questions asked so far.
        current_unique_items: Distinct catalog items surfaced so far.
    """

    def __init__(
        self,
        budget_turns: Optional[int] = None,
        budget_runtime: Optional[float] = None,
        budget_tokens: Optional[int] = None,
        budget_questions: Optional[int] = None,
        budget_unique_items: Optional[int] = None,
        policy_budget_weight: float = 1.0,
        user_budget_weight: float = 1.0,
    ):
        """Initialize a BudgetTracker.

        Args:
            budget_turns: Maximum conversation turns before exhaustion.  ``None``
                means no turn limit.
            budget_runtime: Maximum weighted wall-clock seconds before exhaustion.
                ``None`` means no runtime limit.
            budget_tokens: Maximum weighted tokens before exhaustion.  ``None``
                means no token limit.
            budget_questions: Maximum clarifying questions the user simulator may
                ask.  Checked externally via ``get_remaining_questions()``; does
                *not* trigger ``is_exhausted()``.  ``None`` means no limit.
            budget_unique_items: Maximum distinct catalog items the agent may
                surface.  Checked externally via ``get_remaining_unique_items()``;
                does *not* trigger ``is_exhausted()``.  ``None`` means no limit.
            policy_budget_weight: Multiplier for policy-side runtime/token costs.
                Defaults to ``1.0``.
            user_budget_weight: Multiplier for user-simulator runtime/token costs.
                Defaults to ``1.0``.
        """
        self.budget_turns = budget_turns
        self.budget_runtime = budget_runtime
        self.budget_tokens = budget_tokens
        self.budget_questions = budget_questions
        self.budget_unique_items = budget_unique_items
        self.policy_budget_weight = policy_budget_weight
        self.user_budget_weight = user_budget_weight

        # Track current usage
        self.current_turns = 0
        self.current_runtime = 0.0
        self.current_policy_tokens = 0
        self.current_user_tokens = 0
        self.current_tokens = 0
        self.current_questions = 0
        self.current_unique_items = 0

        # Detailed token breakdown across all LLM calls (conversation +
        # final-prediction). Set via set_policy_token_breakdown /
        # set_user_token_breakdown. These are reporting-only and do not feed
        # the budget exhaustion check.
        self.policy_token_breakdown: Dict[str, int] = {
            "input": 0,
            "input_cached": 0,
            "output": 0,
            "reasoning": 0,
        }
        self.user_token_breakdown: Dict[str, int] = {
            "input": 0,
            "input_cached": 0,
            "output": 0,
            "reasoning": 0,
        }

    def set_policy_token_breakdown(self, breakdown: Dict[str, int]):
        """Overwrite policy token breakdown with the agent's cumulative counts."""
        for k in self.policy_token_breakdown:
            self.policy_token_breakdown[k] = int(breakdown.get(k, 0) or 0)

    def set_user_token_breakdown(self, breakdown: Dict[str, int]):
        """Overwrite user token breakdown with the simulator's cumulative counts."""
        for k in self.user_token_breakdown:
            self.user_token_breakdown[k] = int(breakdown.get(k, 0) or 0)

    def add_turn(self):
        """Increment the turn counter by one."""
        self.current_turns += 1

    def add_questions(self, count: int):
        """Add to the clarifying-question counter.

        Args:
            count: Number of questions asked in the current turn.
        """
        self.current_questions += count

    def add_unique_items(self, count: int):
        """Add to the unique-items-surfaced counter.

        Args:
            count: Number of new distinct catalog items shown in the current turn.
        """
        self.current_unique_items += count

    def get_remaining_questions(self) -> Optional[int]:
        """Return the number of clarifying questions still permitted.

        Returns:
            Remaining question budget (≥ 0), or ``None`` if no limit is set.
        """
        if self.budget_questions is None:
            return None
        return max(0, self.budget_questions - self.current_questions)

    def get_remaining_unique_items(self) -> Optional[int]:
        """Return the number of additional unique items that may still be surfaced.

        Returns:
            Remaining unique-items budget (≥ 0), or ``None`` if no limit is set.
        """
        if self.budget_unique_items is None:
            return None
        return max(0, self.budget_unique_items - self.current_unique_items)

    def add_runtime(self, policy_runtime: float, user_runtime: float):
        """Accumulate weighted wall-clock runtime.

        Args:
            policy_runtime: Seconds spent in the policy (agent) this turn.
            user_runtime: Seconds spent in the user simulator this turn.
        """
        weighted_runtime = (
            self.policy_budget_weight * policy_runtime
            + self.user_budget_weight * user_runtime
        )
        self.current_runtime += weighted_runtime

    def add_tokens(self, policy_tokens: int, user_tokens: int):
        """Accumulate weighted token usage.

        Internally tracks raw policy and user token counts separately, then
        recomputes ``current_tokens`` as:
        ``policy_budget_weight * total_policy_tokens + user_budget_weight * total_user_tokens``.

        Args:
            policy_tokens: Tokens used by the policy this turn.
            user_tokens: Tokens used by the user simulator this turn.
        """
        self.current_policy_tokens += policy_tokens
        self.current_user_tokens += user_tokens
        weighted_tokens = (
            self.policy_budget_weight * self.current_policy_tokens
            + self.user_budget_weight * self.current_user_tokens
        )
        self.current_tokens = weighted_tokens

    def is_exhausted(self) -> Tuple[bool, str]:
        """Check whether any hard budget limit has been reached.

        Checks turns, runtime, and tokens in that order.  Questions and
        unique-items budgets are *soft* limits surfaced via
        ``get_remaining_questions()`` / ``get_remaining_unique_items()`` and
        are not checked here.

        Returns:
            A ``(exhausted, reason)`` tuple where ``exhausted`` is ``True`` if
            any limit is reached and ``reason`` is one of
            ``"turns_exhausted"``, ``"runtime_exhausted"``,
            ``"tokens_exhausted"``, or ``""`` (not exhausted).
        """
        if self.budget_turns is not None and self.current_turns >= self.budget_turns:
            return True, "turns_exhausted"
        if (
            self.budget_runtime is not None
            and self.current_runtime >= self.budget_runtime
        ):
            return True, "runtime_exhausted"
        if self.budget_tokens is not None and self.current_tokens >= self.budget_tokens:
            return True, "tokens_exhausted"
        return False, ""

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of current usage and remaining budget for all dimensions.

        Returns:
            A dict with keys ``"turns"``, ``"runtime"``, ``"tokens"``,
            ``"questions"``, and ``"unique_items"``.  Each value is a dict with
            ``"current"``, ``"budget"`` (limit or ``None``), and ``"remaining"``
            (remaining capacity or ``None`` if unlimited) sub-keys.  The
            ``"tokens"`` entry additionally includes ``"policy_tokens"`` and
            ``"user_tokens"`` (raw unweighted counts) plus ``"policy_breakdown"``
            and ``"user_breakdown"`` (per-bucket input/input_cached/output/
            reasoning splits, populated via ``set_*_token_breakdown``).
        """
        return {
            "turns": {
                "current": self.current_turns,
                "budget": self.budget_turns,
                "remaining": (
                    self.budget_turns - self.current_turns
                    if self.budget_turns is not None
                    else None
                ),
            },
            "runtime": {
                "current": self.current_runtime,
                "budget": self.budget_runtime,
                "remaining": (
                    self.budget_runtime - self.current_runtime
                    if self.budget_runtime is not None
                    else None
                ),
            },
            "tokens": {
                "current": self.current_tokens,
                "budget": self.budget_tokens,
                "remaining": (
                    self.budget_tokens - self.current_tokens
                    if self.budget_tokens is not None
                    else None
                ),
                "policy_tokens": self.current_policy_tokens,
                "user_tokens": self.current_user_tokens,
                "policy_breakdown": dict(self.policy_token_breakdown),
                "user_breakdown": dict(self.user_token_breakdown),
            },
            "questions": {
                "current": self.current_questions,
                "budget": self.budget_questions,
                "remaining": self.get_remaining_questions(),
            },
            "unique_items": {
                "current": self.current_unique_items,
                "budget": self.budget_unique_items,
                "remaining": self.get_remaining_unique_items(),
            },
        }
