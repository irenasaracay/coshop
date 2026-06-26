"""
Evaluate a shopping agent on the CoShop benchmark.

Example usage:
    python evaluate_agent.py \
        --dataset movielens \
        --policy raw_llm \
        --policy_model claude-haiku-4-5 \
        --simulator expert_user \
        --simulator_model claude-haiku-4-5 \
        --retrieval_type BM25 \
        --output_dir results/ \
        --spec_indices 0 1 2
"""

import argparse
import gc
import itertools
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import tqdm

from coshop.data import get_dataset
from coshop.evaluation.budget import BudgetTracker
from coshop.evaluation.metrics import compute_evaluation_metrics
from coshop.tools.catalog_retrieval import get_hard_filter_retrieval_tool, get_retrieval_tool
from coshop.tools.retrieval.retrieval import get_retrieval_fn
from coshop.tools.reflect import get_reflect_tool
from coshop.user_simulator import get_simulator
from example_agents import get_policy, POLICIES
from example_agents.conversational import (
    MSG_FMT_INSTRUCTIONS,
    MSG_FMT_STRUCTURED_DIALOG_ACTIONS,
    POLICY_MSG_FMT_INSTRUCTIONS_ITEM_JSON,
)
from example_agents.langchain_prediction_utils import _extract_seen_ids_from_tool_calls


# ---------------------------------------------------------------------------
# Elicitation loop
# ---------------------------------------------------------------------------


def run_elicitation(
    *,
    simulator_obj,
    policy_obj,
    budget_tracker: BudgetTracker,
    allow_policy_end: bool,
    verbosity: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the elicitation conversation loop. The user (simulator) always speaks first.

    Returns:
        (end_reason, budget_metrics)
    """
    if hasattr(simulator_obj, "set_budget_tracker"):
        simulator_obj.set_budget_tracker(budget_tracker)

    prev_policy_tokens = 0
    prev_user_tokens = 0

    # User opens
    user_response = simulator_obj()
    if verbosity > 0:
        print(f"[USER] {user_response}")
        print("-" * 60)

    user_history = simulator_obj.get_conversation_history()
    if user_history:
        last = user_history[-1]
        prev_user_tokens = last.get("token_cost", 0) or 0
        budget_tracker.add_tokens(0, prev_user_tokens)
        budget_tracker.add_runtime(0.0, last.get("runtime_cost", 0.0) or 0.0)

    try:
        while True:
            exhausted, reason = budget_tracker.is_exhausted()
            if exhausted:
                return f"budget_exhausted:{reason}", budget_tracker.get_metrics()

            assistant_msg = policy_obj(user_response)
            if allow_policy_end and policy_obj.wants_to_end_conversation:
                return "policy_end", budget_tracker.get_metrics()

            policy_history = policy_obj.get_conversation_history()
            if policy_history:
                last = policy_history[-1]
                policy_tokens_total = last.get("token_cost", 0) or 0
                policy_runtime = last.get("runtime_cost", 0.0) or 0.0
            else:
                policy_tokens_total = 0
                policy_runtime = 0.0

            policy_tokens_inc = policy_tokens_total - prev_policy_tokens
            prev_policy_tokens = policy_tokens_total

            exhausted, reason = budget_tracker.is_exhausted()
            if exhausted:
                return f"budget_exhausted:{reason}", budget_tracker.get_metrics()

            user_response = simulator_obj(assistant_msg)
            if verbosity > 0:
                print(f"[AGENT] {assistant_msg}")
                print(f"[USER]  {user_response}")
                print("-" * 60)

            user_history = simulator_obj.get_conversation_history()
            if user_history:
                last = user_history[-1]
                user_tokens_total = last.get("token_cost", 0) or 0
                user_runtime = last.get("runtime_cost", 0.0) or 0.0
            else:
                user_tokens_total = 0
                user_runtime = 0.0

            user_tokens_inc = user_tokens_total - prev_user_tokens
            prev_user_tokens = user_tokens_total

            budget_tracker.add_turn()
            budget_tracker.add_tokens(policy_tokens_inc, user_tokens_inc)
            budget_tracker.add_runtime(policy_runtime, user_runtime)

            exhausted, reason = budget_tracker.is_exhausted()
            if exhausted:
                return f"budget_exhausted:{reason}", budget_tracker.get_metrics()

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"error:{e}", budget_tracker.get_metrics()


# ---------------------------------------------------------------------------
# Per-spec processing
# ---------------------------------------------------------------------------


def _sanitize_ids(
    ranked_ids: List[str], catalog, k: int, rng: random.Random
) -> List[str]:
    """Keep only valid catalog IDs, pad with random items if needed."""
    valid = [i for i in ranked_ids if i in catalog.index][:k]
    if len(valid) < k:
        pool = [str(i) for i in catalog.index if str(i) not in set(valid)]
        rng.shuffle(pool)
        valid += pool[: k - len(valid)]
    return valid


def _rerank_and_evaluate(
    items, *, simulator, spec, catalog, k, seen_item_ids, cache_dir
):
    """
    Have the user simulator re-rank a slate of candidate items (id -> text) and
    score the resulting order. Returns a PredictedEvaluation, or None if the
    simulator rerank fails.
    """
    if not items:
        return None
    try:
        ranked, _meta = simulator.rank_items(items, mode="agentic")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[WARN] simulator rerank failed: {e}")
        return None

    valid_ids = set(items)
    ranked = [i for i in ranked if i in valid_ids][:k]
    evaluation = compute_evaluation_metrics(
        spec=spec,
        ranked_item_ids=ranked,
        catalog=catalog,
        execution_max_per_retrieval=None,
        execution_max_queries=None,
        execution_global_max=None,
        k=k,
        seen_item_ids=seen_item_ids,
        cache_dir=cache_dir,
    )
    return evaluation.predicted


def process_spec(spec_index: int, args, dataset) -> bool:
    """Evaluate a single specification. Returns True on success."""
    try:
        spec = dataset[spec_index]

        # ---- retrieval function ----
        retrieval_kwargs = dict(args.retrieval_kwargs)
        retrieval_kwargs.setdefault("dataset_name", args.dataset)
        # Standard RAG condition: LLM eval-expression prefilter, scored by the
        # simulator model (matches example_experiment_launchers/launch_standard_rag_agent.sh).
        retrieval_kwargs.setdefault("prefilter", True)
        retrieval_kwargs.setdefault("eval_expression_model_name", args.simulator_model)
        # Retrieval is always done on the unrestricted representation (matching the
        # vector-search server, which embeds the unrestricted text); the restricted
        # display representation is only used to render returned items. Without this,
        # BM25 would index the restricted representation while VectorSearch searches
        # the unrestricted one — an inconsistency between the two backends.
        retrieval_kwargs.setdefault(
            "retrieval_representation",
            getattr(dataset, "representation_unrestricted", dataset.representation),
        )
        dataset_version = getattr(dataset, "version", None)
        if dataset_version is not None:
            retrieval_kwargs.setdefault("version", dataset_version)
        if args.vector_search_api_url:
            retrieval_kwargs.setdefault("vector_search_api_url", args.vector_search_api_url)
        # Bare retrieval fn used for final predictions. Per-call / global item
        # limits are enforced by get_final_predictions (via the `m` argument) and
        # by the elicitation tool wrappers, not by the retrieval fn constructor,
        # so only pass arguments BM25/VectorSearch actually accept. In particular
        # the eval-expression prefilter needs `eval_expression_columns`.
        retrieval_fn = get_retrieval_fn(
            retrieval_name=args.retrieval_type,
            catalog=dataset.catalog,
            representation=dataset.representation,
            eval_expression_columns=dataset.filterable_features,
            **retrieval_kwargs,
        )

        # ---- tools ----
        policy_tools = []
        if args.retrieval_access:
            if args.hard_filter_retrieval_access:
                policy_tools.append(
                    get_hard_filter_retrieval_tool(
                        retrieval_name=args.retrieval_type,
                        catalog=dataset.catalog,
                        filterable_features=dataset.filterable_features,
                        max_items_limit=args.elicitation_max_per_retrieval,
                        execution_global_max=args.elicitation_global_max,
                        representation=dataset.representation,
                        **retrieval_kwargs,
                    )
                )
            else:
                policy_tools.append(
                    get_retrieval_tool(
                        retrieval_name=args.retrieval_type,
                        catalog=dataset.catalog,
                        representation=dataset.representation,
                        max_items_limit=args.elicitation_max_per_retrieval,
                        execution_global_max=args.elicitation_global_max,
                        **retrieval_kwargs,
                    )
                )
        if args.thought_tool_access:
            policy_tools.append(get_reflect_tool())

        # ---- simulator ----
        simulator_kwargs = dict(args.simulator_kwargs)
        simulator_kwargs.setdefault("dataset", dataset)
        # The oracle (full_spec_user) surfaces a single-turn z* preference
        # description; which z-variant (z0/zs/zse/zstar) is set via z_condition.
        if args.simulator == "full_spec_user":
            simulator_kwargs.setdefault("z_condition", args.z_condition)
        else:
            # Standard CoPref user defaults (match launch_standard_rag_agent.sh):
            # non-proactive, reveals a few features per turn, stops once x* is
            # identifiable, and uses JSON item representations.
            simulator_kwargs.setdefault("early_stop_on_xstar", False)
            simulator_kwargs.setdefault("max_features_per_turn", 5)
            simulator_kwargs.setdefault("proactive_user", False)
            # Item-JSON formatting is controlled by a single flag for both the
            # policy and the simulator so the two cannot drift apart. Warn and
            # override if the simulator_kwargs tried to set it independently.
            sim_use_jsons = simulator_kwargs.get("use_item_jsons")
            if sim_use_jsons is not None and bool(sim_use_jsons) != args.policy_formats_items_as_json:
                print(
                    f"[WARN] simulator_kwargs use_item_jsons={sim_use_jsons} conflicts "
                    f"with --policy_formats_items_as_json={args.policy_formats_items_as_json}; "
                    "using --policy_formats_items_as_json for both."
                )
            simulator_kwargs["use_item_jsons"] = args.policy_formats_items_as_json
            simulator_kwargs["use_structured_actions"] = args.use_structured_actions
            simulator_kwargs.setdefault("parser_reasoning_effort", "medium")
        simulator_model_kwargs: Dict[str, Any] = {"seed": args.seed}
        if args.simulator_vllm_url:
            simulator_model_kwargs["model_provider"] = "vllm"
            simulator_model_kwargs["vllm_api_url"] = args.simulator_vllm_url
        simulator = get_simulator(
            args.simulator,
            spec=spec,
            model_name=args.simulator_model,
            model_kwargs=simulator_model_kwargs,
            verbosity=args.verbosity,
            **simulator_kwargs,
        )

        # ---- policy ----
        is_baseline = args.policy in ("random_baseline", "popularity_baseline")
        # Oracle simulator (full_spec_user) reveals z* in a single turn instead
        # of a multi-turn elicitation conversation (mirrors eval_z_conditions).
        is_oracle = args.simulator == "full_spec_user"

        policy_kwargs = dict(args.policy_kwargs)
        if args.policy == "popularity_baseline":
            if hasattr(dataset, "popularity_df"):
                policy_kwargs.setdefault("popularity_df", dataset.popularity_df)
            policy_kwargs.setdefault("catalog", dataset.catalog)
            policy_kwargs.setdefault("representation", dataset.representation)

        if args.budget_turns is not None:
            policy_kwargs.setdefault("budget_turns", args.budget_turns)
        if args.budget_questions is not None:
            policy_kwargs.setdefault("budget_questions", args.budget_questions)
        if args.budget_unique_items is not None:
            policy_kwargs.setdefault("budget_unique_items", args.budget_unique_items)

        # Choose how the policy formats items in its messages. By default it
        # wraps item ids in <item>...</item> tags (MSG_FMT_INSTRUCTIONS); with
        # --policy_formats_items_as_json it instead emits items as JSON objects.
        if args.use_structured_actions:
            msg_fmt = MSG_FMT_STRUCTURED_DIALOG_ACTIONS
        elif args.policy_formats_items_as_json:
            msg_fmt = POLICY_MSG_FMT_INSTRUCTIONS_ITEM_JSON
        else:
            msg_fmt = MSG_FMT_INSTRUCTIONS

        policy_model_kwargs: Dict[str, Any] = {"seed": args.seed}
        if args.policy_vllm_url:
            policy_model_kwargs["model_provider"] = "vllm"
            policy_model_kwargs["vllm_api_url"] = args.policy_vllm_url

        policy = get_policy(
            args.policy,
            spec=spec,
            model_name=args.policy_model,
            model_kwargs=policy_model_kwargs,
            verbosity=args.verbosity,
            actions=policy_tools,
            msg_fmt_instructions=msg_fmt,
            **policy_kwargs,
        )

        # ---- elicitation ----
        budget_tracker = BudgetTracker(
            budget_turns=args.budget_turns,
            budget_tokens=args.budget_tokens,
            budget_questions=args.budget_questions,
            budget_unique_items=args.budget_unique_items,
        )

        if is_baseline:
            end_reason = "baseline_policy"
            budget_metrics = budget_tracker.get_metrics()
        elif is_oracle:
            # Insert the oracle's single-turn z* response into the policy's
            # conversation history and skip the elicitation loop. The exact
            # z-variant (z0/zs/zse/zstar) is controlled by the simulator's
            # z_condition. Mirrors eval_z_conditions.py.
            policy.insert_user_msg(simulator())
            end_reason = "oracle_simulator"
            budget_metrics = budget_tracker.get_metrics()
        else:
            end_reason, budget_metrics = run_elicitation(
                simulator_obj=simulator,
                policy_obj=policy,
                budget_tracker=budget_tracker,
                allow_policy_end=args.allow_policy_end,
                verbosity=args.verbosity,
            )

        # ---- collect seen ids from conversation tool calls ----
        conversation_seen_ids: List[str] = []
        for turn in policy.get_conversation_history():
            for action in turn.get("actions", []):
                conversation_seen_ids.extend(
                    _extract_seen_ids_from_tool_calls(action.get("tool_calls", []))
                )

        # ---- final predictions ----
        ranked_ids, pred_meta = policy.get_final_predictions(
            k=args.k,
            retrieval_function=retrieval_fn,
            execution_max_per_retrieval=args.execution_max_per_retrieval,
            execution_max_queries=args.execution_max_queries,
            execution_global_max=args.execution_global_max,
            prediction_summarize_after=args.prediction_summarize_after,
        )

        # Accumulate seen ids from final search
        research_seen_ids = set(conversation_seen_ids)
        if isinstance(pred_meta, dict):
            research_seen_ids.update(pred_meta.get("seen_ids", []))
            for tc in pred_meta.get("tool_calls", []):
                research_seen_ids.update(_extract_seen_ids_from_tool_calls([tc]))
            research_seen_ids.update(str(i) for i in pred_meta.get("item_ids", []))

        ranked_ids = _sanitize_ids(
            ranked_ids, dataset.catalog, args.k, random.Random(args.seed + 1)
        )

        # ---- evaluate ----
        evaluation = compute_evaluation_metrics(
            spec=spec,
            ranked_item_ids=ranked_ids,
            catalog=dataset.catalog,
            execution_max_per_retrieval=args.execution_max_per_retrieval,
            execution_max_queries=args.execution_max_queries,
            execution_global_max=args.execution_global_max,
            k=args.k,
            seen_item_ids=list(research_seen_ids),
            cache_dir=args.cache_dir,
        )

        # ---- simulator preference-state metadata ----
        # Capture the simulator's initial preference state and how it evolved
        # per turn (z_end_of_turn + features revealed each turn, via the feature
        # tracker). Must be captured BEFORE the team-accuracy branches below,
        # since branch 3 reveals all features and would otherwise pollute this.
        simulator_state_history = (
            simulator.get_state_history()
            if hasattr(simulator, "get_state_history")
            else None
        )
        initial_preference_state = {
            "initial_z_context": getattr(simulator, "_initial_z_context", None),
            "initial_known_feature_names": getattr(
                simulator, "_initial_known_feature_names", None
            ),
        }

        # ---- team-accuracy reranking (3 branches) ----
        # After the agent emits its k predictions, have the user simulator
        # re-rank those items under three conditions and score each by
        # recall_at_1 (the "team accuracy"):
        #   1. standard                 -> agent-written reports
        #   2. + oracle reports         -> true dataset item text
        #   3. + oracle reports + full  -> true text, all features revealed
        # Shuffle the slate once (shared across branches) so the simulator does
        # not just receive the agent's own ranking as input-order bias.
        slate_ids = [i for i in ranked_ids if i in dataset.catalog.index]
        random.Random(args.seed + 2).shuffle(slate_ids)
        oracle_items = {
            i: dataset.representation.row_to_str(dataset.catalog.loc[i])
            for i in slate_ids
        }
        rerank_seen_ids = list(research_seen_ids)

        # Branch 1: standard (agent-written reports)
        try:
            report_items_raw = policy.get_final_report(
                oracle_items, use_item_jsons=args.policy_formats_items_as_json
            )
            # Re-impose the shared shuffled slate order so the simulator receives
            # items in the same order as branches 2 & 3 (not the agent's ranking).
            report_items = {i: report_items_raw[i] for i in slate_ids if i in report_items_raw}
            report_items.update({i: report_items_raw[i] for i in report_items_raw if i not in report_items})
        except Exception as e:
            print(f"[WARN] policy.get_final_report failed: {e}")
            report_items = None
        pred_eval_standard = (
            _rerank_and_evaluate(
                report_items,
                simulator=simulator,
                spec=spec,
                catalog=dataset.catalog,
                k=args.k,
                seen_item_ids=rerank_seen_ids,
                cache_dir=args.cache_dir,
            )
            if report_items
            else None
        )

        # Branch 2: + oracle reports (true dataset item text)
        pred_eval_oracle = _rerank_and_evaluate(
            oracle_items,
            simulator=simulator,
            spec=spec,
            catalog=dataset.catalog,
            k=args.k,
            seen_item_ids=rerank_seen_ids,
            cache_dir=args.cache_dir,
        )

        # Branch 3: + oracle reports + full user preference state.
        # Runs last because revealing all features mutates simulator state.
        all_columns = list(simulator.features_star.columns)
        simulator.feature_tracker.reveal_features(
            all_columns, categories=["search", "experience", "credence"]
        )
        pred_eval_full = _rerank_and_evaluate(
            oracle_items,
            simulator=simulator,
            spec=spec,
            catalog=dataset.catalog,
            k=args.k,
            seen_item_ids=rerank_seen_ids,
            cache_dir=args.cache_dir,
        )

        team_branches = [
            ("standard", pred_eval_standard),
            ("+ oracle reports", pred_eval_oracle),
            ("+ oracle reports + full user preference state", pred_eval_full),
        ]
        print("\n==== Team accuracy (recall_at_1) ====")
        for name, pe in team_branches:
            acc = pe.recall_at_1 if pe is not None else None
            print(f"[TEAM ACC] {name}: recall_at_1 = {acc}")

        # ---- sync token breakdowns ----
        # Pull each agent's cumulative per-LLM-call token breakdown (conversation
        # + final prediction/report) into the budget tracker, then re-snapshot
        # budget_metrics so the saved file carries the input/cached/output/
        # reasoning splits under budget_metrics["tokens"]["policy_breakdown"]/
        # ["user_breakdown"].
        if hasattr(policy, "cumulative_token_breakdown"):
            budget_tracker.set_policy_token_breakdown(policy.cumulative_token_breakdown)
        if hasattr(simulator, "cumulative_token_breakdown"):
            budget_tracker.set_user_token_breakdown(simulator.cumulative_token_breakdown)
        budget_metrics = budget_tracker.get_metrics()

        # ---- save ----
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_spec{spec_index}_{args.policy}_k{args.k}.json",
        )
        result = {
            "config": {
                "dataset": args.dataset,
                "spec_index": spec_index,
                "policy": args.policy,
                "policy_model": args.policy_model,
                "simulator": args.simulator,
                "simulator_model": args.simulator_model,
                "z_condition": args.z_condition,
                "retrieval_type": args.retrieval_type,
                "k": args.k,
                "seed": args.seed,
            },
            "end_reason": end_reason,
            "budget_metrics": budget_metrics,
            # The agent's own final top-k ranking is available (with scores) in
            # evaluation.predicted.predicted_ids.
            "agent_evaluation": asdict(evaluation),
            "team_evaluation": {
                "standard": asdict(pred_eval_standard)
                if pred_eval_standard is not None
                else None,
                "oracle_reports": asdict(pred_eval_oracle)
                if pred_eval_oracle is not None
                else None,
                "oracle_reports_full_pref_state": asdict(pred_eval_full)
                if pred_eval_full is not None
                else None,
            },
            # Simulator's initial preference state and per-turn evolution
            # (z_end_of_turn + features revealed each turn, via feature tracker).
            "initial_preference_state": initial_preference_state,
            "turns": [
                {
                    "user_msg": u.get("msg") if u else None,
                    "user_actions": u.get("actions", []) if u else [],
                    "user_token_cost": u.get("token_cost") if u else None,
                    "user_runtime_cost": u.get("runtime_cost") if u else None,
                    "policy_msg": p.get("msg") if p else None,
                    "policy_actions": p.get("actions", []) if p else [],
                    "policy_token_cost": p.get("token_cost") if p else None,
                    "policy_runtime_cost": p.get("runtime_cost") if p else None,
                }
                for u, p in itertools.zip_longest(
                    simulator.get_conversation_history(),
                    policy.get_conversation_history(),
                )
            ],
            # The candidate slate the simulator re-ranked, plus the item text
            # shown in each condition: agent-written reports (standard) vs.
            # the true dataset item text (oracle conditions).
            "rerank_slate_ids": slate_ids,
            "agent_report_text": report_items,
            "oracle_item_text": oracle_items,
            "simulator_state_history": simulator_state_history,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        if args.verbosity > 0:
            print(f"Saved results to {output_path}")
            print(f"End reason: {end_reason}")
            print(f"Top-{args.k} predictions: {ranked_ids}")

        del spec, simulator, policy
        gc.collect()
        return True

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[ERROR] spec {spec_index}: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _ParseKwargs(argparse.Action):
    """Parse key=value pairs into a dict."""

    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest, {}) or {}
        for item in values or []:
            k, _, v = item.partition("=")
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
            if v == "True":
                v = True
            elif v == "False":
                v = False
            d[k] = v
        setattr(namespace, self.dest, d)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CoShop shopping agent")

    # Dataset
    parser.add_argument(
        "--dataset", default="movielens", choices=["movielens", "goodreads", "hm"]
    )
    parser.add_argument(
        "--spec_indices",
        nargs="*",
        type=int,
        default=None,
        help="Spec indices to run (omit or pass -1 for all specs, indices 0-99)",
    )
    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Dataset version (default v2)",
    )
    parser.add_argument(
        "--dataset_kwargs",
        nargs="*",
        action=_ParseKwargs,
        default={},
        help="Extra key=value kwargs forwarded to the dataset constructor",
    )

    # Retrieval
    parser.add_argument(
        "--retrieval_type",
        default="VectorSearch",
        help="Retrieval backend: BM25 or VectorSearch",
    )
    parser.add_argument("--retrieval_kwargs", nargs="*", action=_ParseKwargs, default={})
    parser.add_argument(
        "--retrieval_access",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Give the policy a catalog-retrieval tool during elicitation",
    )
    parser.add_argument(
        "--hard_filter_retrieval_access",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Use the column-filtering variant of the retrieval tool",
    )
    parser.add_argument(
        "--thought_tool_access",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Give the policy a reflect (scratchpad) tool",
    )

    # Policy
    parser.add_argument("--policy", default="copref_aware_llm", choices=POLICIES)
    parser.add_argument("--policy_model", default="claude-haiku-4-5")
    parser.add_argument("--policy_kwargs", nargs="*", action=_ParseKwargs, default={})
    parser.add_argument(
        "--use_structured_actions",
        action="store_true",
        help="Use structured dialog-action format (ASK_QUESTION / SHOW_ITEM_FOR_FEEDBACK)",
    )
    parser.add_argument(
        "--policy_formats_items_as_json",
        type=lambda x: x.lower() != "false",
        default=True,
        help=(
            "Use JSON item formatting for BOTH the policy and the simulator (kept "
            "coupled). The policy emits items as JSON objects (instead of "
            "<item>...</item> tags) and uses JSON item text in the final report, and "
            "the simulator's use_item_jsons is forced to match. Default True. "
            "Ignored for the policy message format with --use_structured_actions."
        ),
    )

    # Simulator
    parser.add_argument(
        "--simulator",
        default="copref_user",
        choices=["expert_user", "copref_user", "full_spec_user"],
    )
    parser.add_argument("--simulator_model", default="claude-haiku-4-5")
    parser.add_argument(
        "--simulator_kwargs", nargs="*", action=_ParseKwargs, default={}
    )
    parser.add_argument(
        "--z_condition",
        default="zstar",
        choices=["z0", "zs", "zse", "zstar"],
        help=(
            "Preference-context variant the oracle simulator (full_spec_user) "
            "reveals in its single turn: z0 (base), zs (+search), zse "
            "(+experience), zstar (all features). Ignored by other simulators."
        ),
    )

    # Budget (elicitation)
    parser.add_argument(
        "--budget_turns",
        type=int,
        default=5,
        help="Max (user, agent) turn pairs during elicitation",
    )
    parser.add_argument("--budget_tokens", type=int, default=None)
    parser.add_argument(
        "--budget_questions",
        type=int,
        default=20,
        help="Max clarifying questions the policy may ask",
    )
    parser.add_argument("--budget_unique_items", type=int, default=10)
    parser.add_argument(
        "--allow_policy_end",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Let the policy end the conversation early (<END_CONVERSATION>)",
    )
    parser.add_argument("--elicitation_max_per_retrieval", type=int, default=None)
    parser.add_argument("--elicitation_global_max", type=int, default=50)

    # Final-prediction search budget
    parser.add_argument("--k", type=int, default=5, help="Top-k items to predict")
    parser.add_argument("--execution_max_per_retrieval", type=int, default=None)
    parser.add_argument("--execution_max_queries", type=int, default=None)
    parser.add_argument("--execution_global_max", type=int, default=250)
    parser.add_argument(
        "--prediction_summarize_after",
        type=int,
        default=2,
        help=(
            "If set, the final-prediction agent summarizes its inherited conversation "
            "history up front and arms per-step mid-rollout compression after this many "
            "fresh tool results. None (default) keeps the full inherited context."
        ),
    )

    # vLLM / external API URLs
    parser.add_argument(
        "--policy_vllm_url",
        default=None,
        help="vLLM base URL for the policy model (sets model_provider=vllm)",
    )
    parser.add_argument(
        "--simulator_vllm_url",
        default=None,
        help="vLLM base URL for the simulator model (sets model_provider=vllm)",
    )
    parser.add_argument(
        "--vector_search_api_url",
        default=None,
        help="Vector search API URL (overrides VECTOR_SEARCH_API_URL env var)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument(
        "--cache_dir", default=None, help="Directory for scoring cache (optional)"
    )

    args = parser.parse_args()

    # Propagate vector search URL to env so all downstream code picks it up
    if args.vector_search_api_url:
        os.environ["VECTOR_SEARCH_API_URL"] = args.vector_search_api_url

    # Load dataset
    dataset_kwargs = dict(args.dataset_kwargs)
    dataset_kwargs.setdefault("max_xstar", 1)
    dataset_kwargs.setdefault("version", args.version)
    dataset = get_dataset(args.dataset, **dataset_kwargs)

    # Determine spec indices
    if args.spec_indices is None or args.spec_indices == [-1]:
        indices = list(range(dataset.fixed_length))
    else:
        indices = args.spec_indices

    print(
        f"Evaluating {len(indices)} spec(s) on {args.dataset} | policy={args.policy} | simulator={args.simulator}"
    )

    results = []
    for idx in tqdm.tqdm(indices, desc="Specs"):
        ok = process_spec(idx, args, dataset)
        results.append(ok)

    successful = sum(results)
    print(f"\nDone: {successful}/{len(results)} specs succeeded.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
