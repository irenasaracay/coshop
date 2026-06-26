#!/bin/bash
#
# Full-knowledge (z*) upper bound.
#
# Instead of a multi-turn elicitation conversation, the full-specification user
# reveals its complete preference context (z_condition=zstar) in a single turn,
# and a plain raw_llm policy makes its predictions from that. This is the
# no-elicitation-cost ceiling the interactive conditions are compared against.
# Results are written to results/<dataset>/full_knowledge/.

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
    output_dir="$OUTPUT_ROOT/$dataset/fully_specified"
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

        # Policy: plain raw LLM (no elicitation conversation).
        --policy raw_llm
        --policy_model "$POLICY_MODEL"
        --policy_formats_items_as_json True

        # User: full-specification oracle, reveals the complete z* context in
        # a single turn (no elicitation loop runs).
        --simulator full_spec_user
        --simulator_model "$SIMULATOR_MODEL"
        --z_condition zstar

        # Final-prediction budget.
        --budget_turns 10
        --execution_global_max 250
    )

    [ -n "$POLICY_VLLM_URL" ]    && cmd+=(--policy_vllm_url "$POLICY_VLLM_URL")
    [ -n "$SIMULATOR_VLLM_URL" ] && cmd+=(--simulator_vllm_url "$SIMULATOR_VLLM_URL")

    echo "=== $dataset / full_knowledge -> $output_dir ==="
    "${cmd[@]}" || echo "ERROR: run failed for $dataset/full_knowledge, continuing..."
done

echo "Full-knowledge (z*) sweep completed!"
