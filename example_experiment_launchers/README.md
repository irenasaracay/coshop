# Example launchers

Self-contained launcher scripts that drive `evaluate_agent.py` for the main
evaluation conditions in the CoShop benchmark. Each script is standalone (every
config is spelled out inline near the top) and sweeps the three datasets
(`hm`, `movielens`, `goodreads`), writing results to
`results/<dataset>/<condition>/`.

| Script | Condition | Simulator | Policy | Paper |
| --- | --- | --- | --- | --- |
| `launch_standard_rag_agent.sh` | Standard RAG agent vs. CoPref (SEC) users — the headline benchmark | `copref_user` | `copref_aware_llm` | §5.2 "Team accuracy reveals human-facing failures" (Table 2, RAG Agents) |
| `launch_search_features_only.sh` | All features made search features (`F_s = F`); preferences are merely retrieved | `expert_user` | `copref_aware_llm` | §5.1 "Agents excel at search but fail to resolve underspecification" (Figure 4A, *search features only*) |
| `launch_fully_specified.sh` | Full preference set `φ(x*)` revealed upfront, no elicitation — execution-only ceiling | `full_spec_user` | `raw_llm` | §5.1 (Figure 4A, *fully-specified*) |
| `launch_history_agent.sh` | Policy additionally sees the user's prior rating history — knowledge intervention | `copref_user` | `copref_aware_history_llm` | §5.3 "Interventions on agent knowledge and communication" (Table 2, history access; Figure 4C) |
| `launch_override_item_descs.sh` | Agent item descriptions overridden to mention all features — communication intervention | `copref_user` | `copref_aware_llm` | §5.3 "Interventions on agent knowledge and communication" (Table 2, override item descriptions; Figure 4C) |
| `launch_structured_agent.sh` | Policy emits structured dialog actions (`--use_structured_actions`) instead of free-form text | `copref_user` | `copref_aware_llm` | Appendix "structured dialog actions" |

## Prerequisites

1. Install the package: `pip install -e .` from `clean_repo/elicitation/`.
2. Start the vector-search server (used by all launchers):
   ```bash
   EMBEDDING_API_URL=http://localhost:8000 ./launch_vector_search_server.sh
   ```
3. Provide model credentials. By default the launchers call the OpenAI API
   (`gpt-4.1-mini`), so set `OPENAI_API_KEY`.

## Running

```bash
./launch_standard_rag_agent.sh
```

To run a quick smoke test, switch to local vLLM models, or change the dataset
grid, edit the config block at the top of the script (`DATASETS`,
`SPEC_INDICES`, `SEED`, `K`, `OUTPUT_ROOT`, `POLICY_MODEL`, `SIMULATOR_MODEL`,
`POLICY_VLLM_URL`, `SIMULATOR_VLLM_URL`, `VECTOR_SEARCH_API_URL`). For example,
to use locally hosted vLLM models instead of a hosted API, set:

```bash
POLICY_MODEL="Qwen/Qwen3.5-27B-FP8"
POLICY_VLLM_URL="http://localhost:8001/v1"
SIMULATOR_MODEL="openai/gpt-oss-120b"
SIMULATOR_VLLM_URL="http://localhost:8000/v1"
```
