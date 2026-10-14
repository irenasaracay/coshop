"""Generators for the per-user dataset assets.

Each benchmark user needs three assets on top of their transaction files:

1. :mod:`~coshop.data.preprocessing.compute_sec_split` — the per-user
   search/experience/credence feature split. Offline and deterministic.
2. :mod:`~coshop.data.preprocessing.initial_known_features` — the S1 feature
   set the simulated user knows at the start. Needs a gpt-oss-120b vLLM server
   and the vector-search server.
3. :mod:`~coshop.data.preprocessing.generate_baseline_queries` — the z0 baseline
   search query. Needs a gpt-oss-120b vLLM server.

Run them in that order. By default they fill in the extended split (users
``TEST_SET_SIZE`` and up) and write to the ``*_extended`` asset files; the
test-set asset files are never modified. See ``README.md`` in this directory.
"""
