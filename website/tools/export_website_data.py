"""Export a small sample of real CoShop specs to JSON for the project website.

Reads the static ``*_v2`` dataset asset files directly (no heavy ``coshop``
loaders, no embedding server, no catalog downloads) and writes
``website/data/samples.json``, which the static site reads client-side to
visualize SEC splits and preference stacks.

The preference stack is rendered as layered feature sets:
``S1 (initial state) -> +search -> +experience -> +credence``, anchored by the
user's initial natural-language query (z0). The output is committed; this is a
one-time generator, not a per-visit pipeline.

    python website/tools/export_website_data.py
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # clean_repo/elicitation
ASSETS = ROOT / "coshop" / "data" / "{domain}" / "assets"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "samples.json"

SPECS_PER_DOMAIN = 3

# Optional explicit sample spec indices per domain (defaults to the first
# SPECS_PER_DOMAIN). goodreads[0] is swapped off a romance title.
SPEC_INDICES = {"goodreads": [3, 1, 2]}

DOMAIN_META = {
    "hm": {
        "label": "H&M",
        "domain": "Fashion / e-commerce",
        "stats": {
            "Test users": "100",
            "Catalog size": "37,570",
            "Initial state size": "3.92 (6.05)",
            "Number of search features": "6.11 (6.84)",
            "Number of experience features": "21.47 (7.33)",
            "Number of credence features": "9.11 (7.42)",
        },
    },
    "movielens": {
        "label": "MovieLens",
        "domain": "Movie recommendation",
        "stats": {
            "Test users": "100",
            "Catalog size": "26,637",
            "Initial state size": "1.67 (1.20)",
            "Number of search features": "8.60 (16.74)",
            "Number of experience features": "70.20 (12.72)",
            "Number of credence features": "8.79 (2.35)",
        },
    },
    "goodreads": {
        "label": "Goodreads",
        "domain": "Book recommendation",
        "stats": {
            "Test users": "100",
            "Catalog size": "39,050",
            "Initial state size": "3.76 (11.08)",
            "Number of search features": "7.76 (11.92)",
            "Number of experience features": "66.98 (65.22)",
            "Number of credence features": "13.50 (70.79)",
        },
    },
}

# Cap features per SEC bucket so the JSON / viz stays small and legible.
MAX_FEATURES_PER_BUCKET = 12

# Human-readable name column for the target item, per domain.
TARGET_NAME_COL = {"hm": "prod_name", "movielens": "title", "goodreads": "title"}


def parse_list(value):
    """Parse a stringified python list (or pass through an actual list)."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        out = ast.literal_eval(value)
        return list(out) if isinstance(out, (list, tuple)) else [out]
    except (ValueError, SyntaxError):
        return [value]


def format_value(value):
    """Render a target-item feature value compactly for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    # Lists are often stored as stringified python lists, e.g. "['Comedy','Drama']".
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        parsed = parse_list(value)
        if parsed:
            return ", ".join(str(p) for p in parsed)[:120]
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    # Booleans stored as strings.
    if text.lower() in ("true", "false"):
        return "yes" if text.lower() == "true" else "no"
    return text[:120] if text else None


def feature_entries(names, descriptions, target_row):
    """Map raw feature names to {name, label, value} dicts, capped for legibility."""
    entries = []
    for name in names[:MAX_FEATURES_PER_BUCKET]:
        value = None
        if target_row is not None and name in target_row.index:
            value = format_value(target_row[name])
        entries.append(
            {"name": name, "label": descriptions.get(name, name), "value": value}
        )
    return entries


def initial_known_for_spec(ikf, idx):
    """Initial known features (S1) and z0 query for a spec.

    Rows in initial_known_features_v2.csv are keyed by ``user_idx`` (the spec
    index) and are NOT in row order, so we must match on user_idx — mirroring
    Dataset.get_initial_known_features_for_spec. ``tag:foo`` entries resolve to
    the bare tag column name.
    """
    rows = ikf[ikf["user_idx"] == idx]
    if rows.empty:
        return [], ""
    r = rows.iloc[0]
    raw = parse_list(r.get("feature_names"))
    feats = [
        n[len("tag:"):].strip() if isinstance(n, str) and n.startswith("tag:") else n
        for n in raw
    ]
    query = r.get("query")
    query = "" if query is None or pd.isna(query) else str(query)
    return feats, query


def target_id_for_spec(base, idx):
    """First id on the first line of {idx}_items.txt is the target item x*."""
    items_path = base / "transactions_v2" / f"{idx}_items.txt"
    if not items_path.exists():
        return None
    first_line = items_path.read_text().splitlines()[0]
    return first_line.split(",")[0].strip()


# How many features to show in the SEC-distribution panel (None = all features).
SEC_DIST_FEATURES = None


def sec_distribution(sec, descriptions):
    """Across all users, how often is each feature search / experience / credence?

    Returns the most-mixed, frequently-appearing features so the panel highlights
    that the same feature is split differently across users (personalized SEC).
    """
    counts = {}
    for buckets in sec.values():
        for bucket in ("search", "experience", "credence"):
            for feat in buckets.get(bucket, []):
                c = counts.setdefault(feat, {"search": 0, "experience": 0, "credence": 0})
                c[bucket] += 1

    rows = []
    for feat, c in counts.items():
        total = c["search"] + c["experience"] + c["credence"]
        if total == 0:
            continue
        # "Mixedness": how far from being a single category (0 = always one bucket).
        frac = [c[b] / total for b in ("search", "experience", "credence")]
        mixed = 1.0 - max(frac)
        rows.append((feat, c, total, mixed))

    # Prefer features that appear often and are split across categories.
    rows.sort(key=lambda r: (r[3] * min(r[2], 100), r[2]), reverse=True)

    out = []
    selected = rows if SEC_DIST_FEATURES is None else rows[:SEC_DIST_FEATURES]
    for feat, c, total, _ in selected:
        out.append(
            {
                "name": feat,
                "label": descriptions.get(feat, feat),
                "n": total,
                "search": round(c["search"] / total, 3),
                "experience": round(c["experience"] / total, 3),
                "credence": round(c["credence"] / total, 3),
            }
        )
    return out


def export_domain(domain):
    base = Path(str(ASSETS).format(domain=domain))
    sec = json.loads((base / "sec_split_v2.json").read_text())
    descriptions = json.loads((base / "feature_descriptions_v2.json").read_text())
    ikf = pd.read_csv(base / "initial_known_features_v2.csv")

    if domain in SPEC_INDICES:
        indices = [str(i) for i in SPEC_INDICES[domain]]
    else:
        indices = sorted(sec.keys(), key=lambda k: int(k))[:SPECS_PER_DOMAIN]

    # Only read the catalog columns we actually need, keyed by id (string).
    needed = set()
    for key in indices:
        for bucket in ("search", "experience", "credence"):
            needed.update(sec[key].get(bucket, []))
    for key in indices:
        s1f, _ = initial_known_for_spec(ikf, int(key))
        needed.update(s1f)

    header = pd.read_csv(base / "catalog_v2.csv", nrows=0).columns
    name_col = TARGET_NAME_COL.get(domain)
    extra = [name_col] if name_col in header else []
    usecols = ["id"] + extra + [c for c in needed if c in header and c != name_col]
    catalog = pd.read_csv(
        base / "catalog_v2.csv", usecols=usecols, index_col="id", low_memory=False
    )
    catalog.index = catalog.index.astype(str)

    specs = []
    for key in indices:
        idx = int(key)
        buckets = sec[key]
        s1_features, z0_query = initial_known_for_spec(ikf, idx)

        target_id = target_id_for_spec(base, idx)
        target_row = (
            catalog.loc[target_id]
            if target_id is not None and target_id in catalog.index
            else None
        )

        # Dedup: features already in the initial state shouldn't repeat in search
        # (the initial state S1 is a subset of the search features).
        s1_set = set(s1_features)
        search_names = [f for f in buckets.get("search", []) if f not in s1_set]

        target_name = None
        if target_row is not None and name_col and name_col in target_row.index:
            target_name = format_value(target_row[name_col])

        specs.append(
            {
                "spec_index": idx,
                "initial_query": z0_query,
                "target": {"id": target_id, "name": target_name},
                "initial_state": feature_entries(s1_features, descriptions, target_row),
                "sec_split": {
                    "search": feature_entries(search_names, descriptions, target_row),
                    "experience": feature_entries(buckets.get("experience", []), descriptions, target_row),
                    "credence": feature_entries(buckets.get("credence", []), descriptions, target_row),
                },
            }
        )
    return {"specs": specs, "sec_distribution": sec_distribution(sec, descriptions)}


def main():
    payload = {"domains": []}
    for domain, meta in DOMAIN_META.items():
        print(f"Exporting {domain} ...")
        payload["domains"].append(
            {"name": domain, **meta, **export_domain(domain)}
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    size = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
