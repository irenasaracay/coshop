#!/bin/bash
#
# Structured-dialog-action elicitation against the co-preference (SEC) user.
#
# Same co-preference setup as the natural condition, but the policy communicates
# via the structured dialog-action format (ASK_QUESTION / SHOW_ITEM_FOR_FEEDBACK)
# instead of free-form natural language. Results are written to
# results/<dataset>/structured_copref/.

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
    output_dir="$OUTPUT_ROOT/$dataset/structured"
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

        # Policy: co-preference (SEC) aware, structured dialog actions, may end early.
        --policy copref_aware_llm
        --policy_model "$POLICY_MODEL"
        --use_structured_actions
        --allow_policy_end True
        --policy_formats_items_as_json False

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

    echo "=== $dataset / structured_copref -> $output_dir ==="
    "${cmd[@]}" || echo "ERROR: run failed for $dataset/structured_copref, continuing..."
done

echo "Structured co-preference (SEC) sweep completed!"
