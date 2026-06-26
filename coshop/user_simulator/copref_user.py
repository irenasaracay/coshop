"""
CoPrefUser: ExpertUser with search/experience/credence
feature partition. The only difference from ExpertUser is that the
FeatureTracker is initialized with non-empty experience and/or credence
feature lists.

The SEC split itself is produced at the dataset layer (loaded from each
dataset's ``sec_split`` asset JSON in data/*/data.py) and stored on each
Specification as spec.sec_split. This class just forwards that split
into ExpertUser's FeatureTracker, with an optional random override.
"""

from __future__ import annotations

import json
import random as _random
from pathlib import Path
from typing import List, Union

from .expert_user import ExpertUser

# Fallback SEC fractions when data/dataset_sec_means.json is missing (e.g. before running compute_dataset_stats.py).
# Each tuple is (search_frac, experience_frac, credence_frac) normalised to sum to 1.
SEC_FRAC_DEFAULTS_DEV: dict[str, tuple[float, float, float]] = {
    "hm": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "goodreads": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "movielens": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}

# Path to SEC means JSON produced by data/compute_dataset_stats.py (dev split means).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SEC_MEANS_PATH = _PROJECT_ROOT / "data" / "dataset_sec_means.json"


def _load_sec_frac_defaults_dev() -> dict[str, tuple[float, float, float]]:
    """Load per-dataset SEC fractions from data/dataset_sec_means.json (dev split means). Fall back to SEC_FRAC_DEFAULTS_DEV."""
    out = dict(SEC_FRAC_DEFAULTS_DEV)
    if not _SEC_MEANS_PATH.exists():
        return out
    try:
        with open(_SEC_MEANS_PATH, encoding="utf-8") as f:
            sec_means = json.load(f)
    except Exception:
        return out
    for ds_name, splits in sec_means.items():
        dev = splits.get("dev") if isinstance(splits, dict) else None
        if not dev or not isinstance(dev, dict):
            continue
        s = float(dev.get("n_search", 0) or 0)
        e = float(dev.get("n_experience", 0) or 0)
        c = float(dev.get("n_credence", 0) or 0)
        total = s + e + c
        if total <= 0:
            continue
        out[ds_name] = (s / total, e / total, c / total)
    return out


class CoPrefUser(ExpertUser):
    """
    ExpertUser with a real search/experience/credence partition on features.

    The only behavioral difference is the FeatureTracker initialization:
    - search_features / experience_features / credence_features are set from
      the spec's sec_split.

    All revelation logic (questions -> search/credence, feedback -> search/experience,
    credence explanations) is handled by FeatureTracker in the parent.
    """

    def __init__(
        self,
        spec,
        seed: int = 0,
        *args,
        randomly_split_sec: bool = False,
        search_frac: Union[float, bool] = False,
        experience_frac: Union[float, bool] = False,
        credence_frac: Union[float, bool] = False,
        **kwargs,
    ):
        # Columns visible to the simulator for this spec.
        columns: List[str] = list(spec.xstar_simulator_view.columns)

        if randomly_split_sec:
            # Resolve fractions: if any is False, use per-dataset dev defaults from dataset_sec_means.json.
            ds_name = spec.dataset_name
            if search_frac is False or experience_frac is False or credence_frac is False:
                defaults = _load_sec_frac_defaults_dev().get(
                    ds_name, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
                )
                s_frac = defaults[0] if search_frac is False else float(search_frac)
                e_frac = defaults[1] if experience_frac is False else float(experience_frac)
                c_frac = defaults[2] if credence_frac is False else float(credence_frac)
            else:
                s_frac = float(search_frac)
                e_frac = float(experience_frac)
                c_frac = float(credence_frac)
            # Reproducible random split using per-instance RNG.
            rng = _random.Random(seed)
            shuffled = columns[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            total = s_frac + e_frac + c_frac
            if total <= 0:
                s_frac = e_frac = c_frac = 1.0 / 3.0
                total = 1.0
            s = int(round(n * (s_frac / total)))
            e = int(round(n * (e_frac / total)))
            c = n - s - e
            if c < 0:
                c = 0
            sec_split = {
                "search": shuffled[:s],
                "experience": shuffled[s: s + e],
                "credence": shuffled[s + e: s + e + c],
            }
        else:
            sec_split = spec.sec_split

        super().__init__(
            spec=spec,
            seed=seed,
            search_features=sec_split.get("search", []),
            experience_features=sec_split.get("experience", []),
            credence_features=sec_split.get("credence", []),
            *args,
            **kwargs,
        )
