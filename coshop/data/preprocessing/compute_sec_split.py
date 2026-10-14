"""
Feature Classification by Preference Novelty
=============================================

Overview
--------
For a given user i and target item x*, we classify each feature f into one of
three categories reflecting how "known" the user's preference for x*'s value
was prior to the purchase:

    - Search:     the user has a well-established prior preference for this value.
                  They likely knew they wanted it and could have found it by
                  filtering or searching directly.

    - Experience: the user has some exposure to this value but no strong
                  directional preference. Familiarity is uncertain.

    - Credence:   the user has never encountered this value in their purchase
                  history, or the feature is structurally unobserved. The
                  preference could only have been formed by experiencing x*
                  itself, or requires domain knowledge to evaluate.

This taxonomy loosely follows the search / experience / credence goods
distinction in information economics (Nelson 1970, Darby & Karni 1973):
search attributes are verifiable before purchase, experience attributes are
learned through consumption, and credence attributes may never be fully
verifiable. Here we adapt the terminology to describe not product attributes
per se, but the user's epistemic relationship to each attribute value at the
time of purchase.

Scoring
-------
Features are scored using a base-rate-adjusted surprise metric. For feature f
with target value v = x*_f, the score is the pointwise mutual information
(PMI) between the user's empirical purchase distribution and the catalog
marginal:

    s(f, i, x*) = log2[ p_catalog(f=v) / p_user(f=v) ]
                = -log2 p_user(f=v) + log2 p_catalog(f=v)

This measures how much more surprised we should be to see value v under the
user's history than under the catalog prior. The catalog correction is
critical: without it, a user who frequently buys a rare material (e.g.
viscose) would appear to have an "unknown" preference for it, simply because
it is rare in the world. The PMI removes this artifact by anchoring surprise
to the base rate.

Interpretation of thresholds
-----------------------------
The score has two thresholds that partition features into the three
categories:

    s < EXPERIENCE_THRESHOLD (= -1):
        The user buys this value at more than twice the catalog rate, which is
        strong evidence of a settled prior preference. Classified as Search.

    EXPERIENCE_THRESHOLD <= s < CREDENCE_THRESHOLD (= 1):
        The user selects this value within a factor of two of the catalog
        rate. No strong preference signal in either direction. Classified as
        Experience.

    s >= CREDENCE_THRESHOLD (= 1), or value never seen:
        The user buys this value at most half as often as the catalog offers
        it, or has never purchased it at all. In the never-seen case, the
        score is undefined (log 0) and we assign a large sentinel value to
        place it at the top of the credence ranking. Classified as Credence.

Edge cases
----------
Three edge cases are handled before scoring:

    1. x*_f is missing or OPEN_TO_ANYTHING: the target item does not have a
       meaningful value for this feature (structural N/A). Excluded entirely.

    2. v never appears in the user's history: p_user(f=v) = 0, so the score
       is undefined. Rather than smooth, we assign SENTINEL_VALUE and classify
       directly as Credence. This is conceptually clean: a value the user has
       never encountered is maximally novel, regardless of its catalog rate.

    3. Feature is almost always missing in the catalog
       (p_catalog(OPEN_TO_ANYTHING) >= MISSINGNESS_THRESHOLD = 0.95): the
       feature is structurally unobserved for nearly all items, making the
       catalog marginal unreliable as a base rate. Classified as Credence.

Manual overrides
----------------
Some features are overridden regardless of their score:

    ALWAYS_SEARCH_FEATURES: features like product_type that are definitionally
        known before purchase -- the user selected this category explicitly.

    ALWAYS_CREDENCE_FEATURES: features that require external knowledge or
        institutional context to evaluate (e.g. publisher, production company,
        Bechdel test result), regardless of purchase history.

Usage
-----
    python -m coshop.data.preprocessing.compute_sec_split --dataset all
    python -m coshop.data.preprocessing.compute_sec_split --dataset hm --start 100 --end 200
"""

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..dataset import OPEN_TO_ANYTHING, Dataset
from ._common import (
    add_common_args,
    load_dataset,
    output_path,
    read_json,
    resolve_datasets,
    todo_indexes,
    user_indexes,
    write_json,
)

CREDENCE_THRESHOLD = 1
EXPERIENCE_THRESHOLD = -1
MISSINGNESS_THRESHOLD = 0.95
SENTINEL_VALUE = 99999

ALWAYS_SEARCH_FEATURES = {
    "hm": [
        "product_type_name",
        "gender",
        "age",
    ],
    "movielens": [
        "genres",
    ],
    "goodreads": [
        "genres",
    ],
}

ALWAYS_CREDENCE_FEATURES = {
    "hm": [
      "vegan",
      "is_sustainable",
      "is_woven_fabric",
      "has_yoke_panel",
      "material",
      "waist_treatment",
      "functional_finish",
      "toxin_free",
      "bodice_style"
    ],
    "movielens": [
      "national_film_registry",
      "award_profile",
      "imdb_top_list",
      "budget",
      "format_resolution",
      "revenue",
      "production_companies",
      "belongs_to_collection",
      "bechdel_test"
    ],
    "goodreads": [
      "technology_level",
      "plot_devices",
      "award_status",
      "fantasy_subgenre",
      "book_club_pick",
      "mystery_subgenre",
      "nonfiction_subtype",
      "magic_system",
      "series_name",
      "author_name",
      "publisher",
      "paranormal_creatures",
      "narrative_devices"
    ],
}


def rank_features_for_user(
    xstar_simulator_view: pd.DataFrame,
    historical_ids: List[str],
    simulator_catalog: pd.DataFrame,
    catalog_marginals: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, float]:
    """Score each feature of the target item; higher = more novel to the user.

    ``catalog_marginals`` caches the per-column catalog value distribution
    across users (it does not depend on the user).
    """
    if catalog_marginals is None:
        catalog_marginals = {}
    # Note: this excludes features for which the target value is missing
    historical_sim_view = simulator_catalog.loc[historical_ids]
    feature_to_score = {}
    for c in xstar_simulator_view.columns:
        tgt = xstar_simulator_view.iloc[0][c]
        # skip missing or open to anything
        if pd.isna(tgt) or tgt == OPEN_TO_ANYTHING:
            continue
        puser = historical_sim_view[c].value_counts(dropna=False) / len(
            historical_sim_view
        )
        # user has never seen this value before -> auto sort to credence
        if tgt not in puser or puser[tgt] == 0:
            # use large sentinel value
            feature_to_score[c] = SENTINEL_VALUE
            continue
        if c not in catalog_marginals:
            catalog_marginals[c] = simulator_catalog[c].value_counts(dropna=False) / len(
                simulator_catalog
            )
        pcatalog = catalog_marginals[c]
        # feature is extremely rare in the catalog -> auto sort to credence
        if OPEN_TO_ANYTHING in pcatalog and pcatalog[OPEN_TO_ANYTHING] >= MISSINGNESS_THRESHOLD:
            feature_to_score[c] = SENTINEL_VALUE
            continue
        surprise = -np.log2(puser[tgt]) + np.log2(pcatalog[tgt])
        feature_to_score[c] = surprise.item()

    return dict(sorted(feature_to_score.items(), key=lambda x: x[1], reverse=True))


def classify_features(
    scores: Dict[str, float], dataset_name: str
) -> Dict[str, List[str]]:
    """Partition scored features into search / experience / credence."""
    always_search = ALWAYS_SEARCH_FEATURES[dataset_name]
    always_credence = ALWAYS_CREDENCE_FEATURES[dataset_name]
    search_features = [
        f
        for f, s in scores.items()
        if (s < EXPERIENCE_THRESHOLD or f in always_search)
        and f not in always_credence
    ]
    experience_features = [
        f
        for f, s in scores.items()
        if EXPERIENCE_THRESHOLD <= s < CREDENCE_THRESHOLD
        and f not in always_search
        and f not in always_credence
    ]
    credence_features = [
        f
        for f, s in scores.items()
        if (s >= CREDENCE_THRESHOLD and f not in always_search)
        or f in always_credence
    ]
    return {
        "search": search_features,
        "experience": experience_features,
        "credence": credence_features,
    }


def compute_sec_split_for_user(
    ds: Dataset, index: int, catalog_marginals: Optional[Dict[str, pd.Series]] = None
) -> Dict[str, List[str]]:
    user = ds.load_user_raw(index)
    scores = rank_features_for_user(
        user["xstar_simulator_view"],
        user["historical_ids"],
        ds.simulator_catalog,
        catalog_marginals,
    )
    return classify_features(scores, ds.dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-user SEC (search/experience/credence) splits."
    )
    add_common_args(parser)
    args = parser.parse_args()

    for name in resolve_datasets(args):
        ds = load_dataset(name, args)
        out = output_path(ds, "sec_split_extended", args)
        existing = read_json(out)
        todo = todo_indexes(user_indexes(ds, args), existing, args, out)
        catalog_marginals: Dict[str, pd.Series] = {}
        for i in todo:
            existing[str(i)] = compute_sec_split_for_user(ds, i, catalog_marginals)
            print(f"{name}_{i}", {k: len(v) for k, v in existing[str(i)].items()})
        write_json(out, existing)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
