# Per-user asset generators

Every benchmark user is one real user from the source dataset. Beyond the
transaction files in `coshop/data/<dataset>/assets/transactions_v2/`, each user
needs three per-user assets. This package generates them.

The assets for both splits are already shipped, so you only need these scripts
to rebuild them or to add more users.

| Step | Module | Output (extended split) | Needs |
|---|---|---|---|
| 1 | `compute_sec_split` | `sec_split_v2_extended.json` | CPU only |
| 2 | `initial_known_features` | `initial_known_features_v2_extended.csv` | gpt-oss-120b vLLM + vector-search server |
| 3 | `generate_baseline_queries` | `baseline_queries_v2_extended.json` | gpt-oss-120b vLLM |

Run them in this order: S1 is drawn from the user's search features, and the
baseline query is written from z0, which is built from S1.

## Splits

- **`test`** (default): users 0–99, the benchmark test set reported in the
  paper. Its assets are `sec_split_v2.json`, `initial_known_features_v2.csv`
  and `baseline_queries_v2.json`. The generators never write to these files.
- **`extended`**: every other user in the transactions directory, keyed by
  file index: H&M 100–993 (894 users), MovieLens 100–999 (900), Goodreads
  100–999 (900). Its assets live in the `*_extended` files named in
  `v2_config.yml`, and the dataset loader merges them in. The extended split
  doesn't support `subset_fraction`.

| | H&M | MovieLens | Goodreads |
|---|---|---|---|
| Catalog items | 37,570 | 26,637 | 39,050 |
| Test users | 100 (0–99) | 100 (0–99) | 100 (0–99) |
| Extended users | 894 (100–993) | 900 (100–999) | 900 (100–999) |
| Median history length (test / extended) | 49.5 / 46 | 139.5 / 145 | 9 / 9 |

**Getting the files.** Each dataset's assets come as two zips on Google
Drive, downloaded into the installed package's `assets/` directory on first
use: the main zip (catalog and the test split; `assets_file_id` in
`<dataset>/data.py`) and a small extended zip (`transactions_v2/` for users
100+ and the three `*_extended` files; `extended_assets_file_id`), fetched
only when you load `split="extended"`. Each zip has its own sentinel
(`.download_complete`, `.download_complete_extended`). To rebuild the extended
zip after adding users, zip those files with the same relative paths.

```python
from coshop.data import get_dataset

ds = get_dataset("hm", split="extended")
spec = ds[150]   # user 150
```

```bash
python evaluate_agent.py --dataset hm --split extended --spec_indices 100 101 ...
```

## Running

All three take `--dataset {hm,movielens,goodreads,all}`, `--start` (default 100),
`--end` (default: all users) and `--version` (default v2). They resume: users
that already have output are skipped unless you pass `--overwrite`. `--out`
writes somewhere else, e.g. to regenerate test-set users into a scratch file
for comparison.

```bash
# 1. SEC split (offline, a few minutes)
python -m coshop.data.preprocessing.compute_sec_split --dataset all

# 2. Initial known features (S1)
python -m coshop.data.preprocessing.initial_known_features --dataset all \
    --vllm-url http://localhost:3005 --vector-search-url http://localhost:3004

# 3. Baseline queries (only used by the random and popularity baselines)
python -m coshop.data.preprocessing.generate_baseline_queries --dataset all \
    --vllm-url http://localhost:3005
```

Start the vector-search server with `launch_vector_search_server.sh`.

**Speed.** In step 2 the query writing is batched, but each round's vector searches
(each with an LLM prefilter call) are independent. `--retrieval-workers` (default 16)
runs them concurrently; this changes speed, not results. Running one process per
dataset parallelizes further.

**Cost.** Step 2 runs about |S1| + 1 rounds per user. Each round makes one LM
call to write the query and one vector search; the search's prefilter makes
another LM call. Expect roughly 2(|S1| + 1) gpt-oss-120b calls per user (about
10 for H&M and Goodreads, about 5 for MovieLens). Step 3 makes one call per user.

## How the assets are built

**SEC split** (`compute_sec_split`). For each feature with a value on x*, the
score is s = log2 p_catalog(v) − log2 p_user(v). Here p_user is the frequency
of x*'s value v in the user's history, and p_catalog its frequency over the
simulator view of the full catalog.
- **Search:** s < −1, or a feature in `ALWAYS_SEARCH_FEATURES`.
- **Experience:** −1 ≤ s < 1.
- **Credence:** s ≥ 1, a value the user has never bought, a feature that is
  ≥ 95% missing in the catalog, or a feature in `ALWAYS_CREDENCE_FEATURES`.

Deterministic. The MovieLens `year` label ("1980s or 1990s") is drawn from a
private RNG seeded with the dataset `seed`, default 0 (see below).

**Initial known features, S1** (`initial_known_features`). Start from the
always-included feature: `product_type_name` for H&M, `genres` otherwise. Then
repeat until x* is in the top K=250:
1. gpt-oss-120b (temperature 0) writes one search query from the known
   attributes;
2. vector search runs the query;
3. if x* isn't found, add one random search feature with a value on x*.

The saved set is the one *before* x* became retrievable, so `recall` is always
0. Each user has its own random generator, seeded with `f"{seed}:{user_idx}"`.
LM and vector-search outputs aren't bit-reproducible.

**Baseline queries** (`generate_baseline_queries`). gpt-oss-120b (temperature 0)
writes one search query from `str(spec.z0)`.

## Where the users come from

The transaction files were built upstream from the raw source datasets. Those
extraction scripts are not part of this package.

- **Target and history.** A user's interactions are sorted newest first. The
  targets (`*_items.txt` line 1) are the most recent valid item plus later items
  in the same group (product type, or genres), up to 5. Line 2 holds 5
  validation items. Only the first id on line 1 is used as x*. History is
  everything that isn't a target, restricted to catalog items.
- **H&M.** Customers with at least 30 purchases, in customer-id order. v2 is
  v1 with 6 users removed (v1 indices 8, 18, 35, 38, 46, 49) and renumbered, so
  v1 and v2 indices differ from 8 on.
- **MovieLens.** ML-20M users with at least 50 ratings of 4 or more on at least
  5 distinct days, in user-id order.
- **Goodreads.** Users with at least 5 read books. Users whose first target is
  tagged sex/erotic/steamy/dark-romance were removed. Users 0–89 and 202–999
  come from the current extraction. **Users 90–201 come from an earlier version
  of the extraction:** 1 target (all books read on the last date), at least 15
  history items, and a 2-line `*_items.txt` with `last_read_date` in the user
  JSON. This includes test users 90–99 and extended users 100–201.
- **Catalog.** Every extended user's targets and history are already in
  `catalog_v2`, so adding users needs no new catalog rows or feature extraction.
- **MovieLens year labels.** The simulator sees a movie's year as one of the two
  20-year windows that contain it ("1970s or 1980s" or "1980s or 1990s"). The
  window is drawn with a private `random.Random(seed)` in catalog row order.
  `seed` is a dataset argument (default 0), and `evaluate_agent.py` passes its
  `--seed`. For seeds 0, 1 and 2 this reproduces the labels of the camera-ready
  MovieLens runs, which seeded the global RNG with the run seed right before
  loading. The generated per-user files use seed 0.
