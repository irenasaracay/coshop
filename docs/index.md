# coshop

**coshop** is a benchmark for evaluating conversational shopping assistant agents that must elicit latent user preferences through dialogue and retrieve the correct item(s) from a catalog.

## Subpackages

| Package | Description |
|---------|-------------|
| [`coshop.data`](api/data/datasets.md) | Dataset loading, item representation, and utility scoring |
| [`coshop.evaluation`](api/evaluation/metrics.md) | NDCG-based metrics and per-turn budget tracking |
| [`coshop.tools`](api/tools/catalog_retrieval.md) | LangChain-compatible retrieval and history-search tools |
| [`coshop.user_simulator`](api/user_simulator/user.md) | Simulated users that reveal preferences progressively |
| [`coshop.utils`](api/utils/custom_types.md) | Shared data types, agent base classes, and helper utilities |
