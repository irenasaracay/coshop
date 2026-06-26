"""coshop: Conversational Shopping benchmark package.

``coshop`` is a benchmark for evaluating shopping assistant agents that must
elicit latent user preferences through conversation and then retrieve the
correct item(s) from a catalog.

Subpackages:
    data: Dataset loading, item representation, and utility scoring.
    evaluation: NDCG-based metrics and per-turn budget tracking.
    tools: LangChain-compatible retrieval and history-search tools.
    user_simulator: Simulated users that reveal preferences progressively.
    utils: Shared data types, agent base classes, and helper utilities.
"""

__version__ = "0.1.0"
