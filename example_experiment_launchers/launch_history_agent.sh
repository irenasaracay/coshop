#!/bin/bash
#
# History-access intervention against the co-preference (SEC) user.
#
# Identical to launch_standard_rag_agent.sh, except the policy is the
# history-aware agent (copref_aware_history_llm), which additionally sees the
# user's prior rating history. This tests whether history access lets agents
# anticipate a user's SEC split or infer preferences without asking.
# Results are written to results/<dataset>/natural_copref_history/.

set -e
trap 'echo "Interrupted. Exiting..."; exit 130' INT TERM

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# ---------------------------------------------------------------------------
# Models. Defaults run against the OpenAI API (set OPENAI_API_KEY).
# Point *_VLLM_URL at a vLLM server to use a local model instead.
# ---------------------------------------------------------------------------
POLICY_MODEL="gpt-4.1-mini"
SIMULATOR_MODEL="gpt-4.1-mini"
POLICY_VLLM_URL=""
SIMULATOR_VLLM_URL=""

# Vector-search server (start it with ./launch_vector_search_server.sh).
VECTOR_SEARCH_API_URL="http://localhost:3004"

# ---------------------------------------------------------------------------
# Evaluation grid.
# ---------------------------------------------------------------------------
DATASETS="hm movielens goodreads"
SPEC_INDICES=$(seq 0 99 | tr '\n' ' ')
SEED=0
K=5
OUTPUT_ROOT="$REPO_ROOT/results"

for dataset in $DATASETS; do
    output_dir="$OUTPUT_ROOT/$dataset/history"
    mkdir -p "$output_dir"

    cmd=(python evaluate_agent.py
        --dataset "$dataset"
        --version v2
        --spec_indices $SPEC_INDICES
        --seed "$SEED"
        --k "$K"
        --output_dir "$output_dir"
        --verbosity 1

        # Retrieval: vector search with the LLM eval-expression hard filter.
        --retrieval_type VectorSearch
        --retrieval_access True
        --thought_tool_access True
        --retrieval_kwargs prefilter=True eval_expression_model_name="$SIMULATOR_MODEL"
        --vector_search_api_url "$VECTOR_SEARCH_API_URL"

        # Policy: co-preference (SEC) aware + user rating-history access.
        --policy copref_aware_history_llm
        --policy_model "$POLICY_MODEL"
        --allow_policy_end True
        --policy_formats_items_as_json True

        # User: CoPrefUser
        --simulator copref_user
        --simulator_model "$SIMULATOR_MODEL"

        # Elicitation budget.
        --budget_turns 5
        --budget_questions 20
        --budget_unique_items 10
        --elicitation_global_max 50

        # Final-prediction budget.
        --execution_global_max 250
        --prediction_summarize_after 2
    )

    [ -n "$POLICY_VLLM_URL" ]    && cmd+=(--policy_vllm_url "$POLICY_VLLM_URL")
    [ -n "$SIMULATOR_VLLM_URL" ] && cmd+=(--simulator_vllm_url "$SIMULATOR_VLLM_URL")

    echo "=== $dataset / natural_copref_history -> $output_dir ==="
    "${cmd[@]}" || echo "ERROR: run failed for $dataset/natural_copref_history, continuing..."
done

echo "History-access co-preference (SEC) sweep completed!"
