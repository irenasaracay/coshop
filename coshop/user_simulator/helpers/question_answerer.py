"""
QuestionAnswerer: answers clarifying questions one-by-one via LOTUS sem_map.

Given current preference context (z), a single question, and a hint about which
feature(s) matter, generates a short shopper-style answer. Used by ExpertUser
to build the full response to multiple clarifying questions by stitching per-question
answers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import lotus

from .feature_utils import normalize_feature_name
from .message_parser import ClarifyingQuestion
from ...utils.lotus import configure_lotus, sem_map_with_retries
from ...utils.misc import print_debug


# Prompt for a single clarifying question (placeholders: z_context, question, feature_hint).
# One row per question; LOTUS fills from DataFrame columns.
SINGLE_QUESTION_PROMPT = """You are the SHOPPER (customer) answering a question from the customer service agent.

Your background: {simulator_persona}

Your current preferences (use only these to answer):
{z_context}
You cannot relax these preferences. They are strict.

Make sure you provide the preference for this feature when answering the question: {feature_hint}
Try not to directly quote this and paraphrase it instead.

Question from the agent: {question}

Answer in 1-2 sentences. Be brief and conversational, but not too confident (e.g., 'Red would be nice' rather than 'I want red and nothing else.'). If your preferences above do not contain the answer, say "I don't know yet" or "I'm open to anything". Do not invent details."""


class QuestionAnswerer:
    """
    Answers each clarifying question via a separate LOTUS sem_map call, then
    stitches the results into one response.
    """

    def __init__(
        self,
        model_name: str = "gpt-5-nano",
        model_kwargs: Optional[dict] = None,
        true_features: Optional[Dict[str, str]] = None,
        simulator_persona: str = "",
        verbosity: int = 0,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs.setdefault("temperature", 0.0)
        self._true_features = true_features or {}
        self._simulator_persona = simulator_persona
        self.verbosity = verbosity
        self._lm = configure_lotus(model_name, self.model_kwargs)
        self._question_prompt_history: List[Dict[str, Any]] = []

    @property
    def question_prompt_history(self) -> List[Dict[str, Any]]:
        """History of (question, prompt, response) for each clarifying-question answer."""
        return self._question_prompt_history

    def clear_question_prompt_history(self) -> None:
        """Clear the question prompt history (e.g. on simulator reset)."""
        self._question_prompt_history = []

    def answer_questions(
        self,
        questions: List[ClarifyingQuestion],
        z_context: str,
        feature_values: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, int, float]:
        """
        Answer each question with a separate LOTUS call, then stitch into one response.

        Args:
            questions: List of clarifying questions (each has .question and .relevant_columns).
            z_context: Current preference context (z) for the shopper.
            feature_values: Optional mapping column_name -> value to show in the feature hint.

        Returns:
            (stitched_response, token_cost, runtime_cost). Token/runtime from LOTUS
            are not available here, so returns 0, 0.0 for cost.
        """
        if not questions:
            return "", 0, 0.0

        # Build one row per question: z_context, question text, human-readable feature hint (name + value)
        rows = []
        for q in questions:
            feature_hint = self._feature_hint(q.relevant_columns, feature_values or {})
            rows.append(
                {
                    "z_context": z_context,
                    "question": q.question,
                    "feature_hint": feature_hint,
                    "simulator_persona": self._simulator_persona,
                }
            )

        df = pd.DataFrame(rows)
        lotus.settings.configure(lm=self._lm)
        try:
            mapped = sem_map_with_retries(
                df,
                SINGLE_QUESTION_PROMPT,
                validation_fn=lambda x: isinstance(x, str) and len(x.strip()) > 0,
            )
        except Exception as e:
            if self.verbosity > 0:
                print_debug(
                    f"QuestionAnswerer LOTUS sem_map failed: {e}",
                    "QuestionAnswerer.answer_questions",
                )
            # No fallback: omit these questions from the response
            return "", 0, 0.0

        # Stitch answers in order and record prompt history
        parts = []
        for i, q in enumerate(questions):
            row = rows[i]
            prompt = SINGLE_QUESTION_PROMPT.format(**row)
            if i < len(mapped) and "_map" in mapped.columns:
                raw = mapped.iloc[i].get("_map", "")
                ans = raw.strip() if isinstance(raw, str) else ""
                self._question_prompt_history.append(
                    {
                        "question": q.question,
                        "prompt": prompt,
                        "response": ans,
                    }
                )
                if ans:
                    parts.append(f"*{q.question}*: {ans}")

        return "\n\n".join(parts), 0, 0.0

    def _feature_hint(
        self,
        relevant_columns: List[str],
        feature_values: Optional[Dict[str, str]] = None,
    ) -> str:
        """Human-readable hint: column name (and value when provided) for this question."""
        if not relevant_columns:
            return "(n/a? Probably an unanswerable question that I should responsd, 'I don't really care' or 'I don't know yet' to. Double check.)"
        vals = feature_values or {}
        parts = []
        for col in relevant_columns:
            if col not in vals:
                continue
            desc = normalize_feature_name(self._true_features.get(col, col))
            parts.append(f"{desc}: {vals[col]}")

        if not parts:
            return "(n/a? Probably an unanswerable question that I should responsd, 'I don't really care' or 'I don't know yet' to. Double check.)"
        
        return "; ".join(parts)
