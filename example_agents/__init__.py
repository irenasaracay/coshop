from .conversational import (
    RawLLM,
    CoPrefAwareLLM,
    CoPrefAwareHistoryLLM,
)
from .baseline_policies import (
    RandomBaselinePolicy,
    PopularityBaselinePolicy,
)


POLICIES = [
    "raw_llm",
    "copref_aware_llm",
    "copref_aware_history_llm",
    "random_baseline",
    "popularity_baseline",
]


def get_policy(policy_name: str, **kwargs):
    if policy_name == "raw_llm":
        return RawLLM(**kwargs)
    elif policy_name == "copref_aware_llm":
        return CoPrefAwareLLM(**kwargs)
    elif policy_name == "copref_aware_history_llm":
        return CoPrefAwareHistoryLLM(**kwargs)
    elif policy_name == "random_baseline":
        return RandomBaselinePolicy(**kwargs)
    elif policy_name == "popularity_baseline":
        return PopularityBaselinePolicy(**kwargs)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")