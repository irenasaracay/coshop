"""Core dataset abstractions for the coshop benchmark.

This module defines two primary classes:

* :class:`Specification` — a single benchmark task, containing the ground-truth
  target item(s) (``xstar``), the layered preference descriptions (z-variants),
  the SEC feature split, and utility scoring functions.
* :class:`Dataset` — an abstract, lazily-loaded collection of
  :class:`Specification` objects backed by a shared item catalog.

Helper function :func:`build_z_variants_for_spec` constructs the z-variant
strings used by simulated users and baseline evaluations.
"""

from typing import (
    List,
    Dict,
    Optional,
    Any,
    Iterator,
    Union,
    Set,
    Tuple,
)
import ast
import random
import json
import os
from pathlib import Path
import pandas as pd
from PIL import Image

from ..utils.misc import (
    download_file_from_google_drive,
)
from .utility import (
    ColumnMatchingUtilityFunction,
    ServerCosinePercentileUtilityFunction,
    ColumnMatchWithServerCosinePercentileUtilityFunction,
)
from .representation import ParagraphRepresentation

ROOT_DIR = Path(__file__).parent  # coshop/data/

# Number of benchmark specs exposed per dataset (indices 0..TEST_SET_SIZE-1).
TEST_SET_SIZE = 100

# Placeholder used to fill NA in simulator/target views (e.g. hm/data.py fillna).
OPEN_TO_ANYTHING = "open to anything"


class Specification:
    """A single benchmark task within a :class:`Dataset`.

    A Specification bundles everything needed to run one elicitation episode:
    the ground-truth target item(s) (``xstar``), layered user-preference
    descriptions (z-variants), the SEC feature partition, historical user data,
    pre-computed baseline queries, and callable utility functions.

    Attributes:
        dataset_name: Identifier of the parent dataset (e.g. ``"hm"``).
        index: Position of this spec within the dataset.
        item_name: Human-readable singular noun for catalog items
            (e.g. ``"clothing item"``).
        z0: Preference description from initially known features only.
        zs: Preference description after all *search* features are revealed.
        zse: Preference description after search + *experience* features are
            revealed.
        zstar: Full ground-truth preference description (search + experience +
            credence features).
        xstar: List of ground-truth item IDs.
        xstar_series: DataFrame of the ground-truth items (full catalog rows).
        xstar_simulator_view: DataFrame of the ground-truth items as seen by
            the simulator (vagueified, hidden features removed).
        sec_split: Mapping ``{"search": [...], "experience": [...],
            "credence": [...]}`` partitioning feature columns by SEC category.
        initial_known_features: Columns the simulated user knows at the start
            of the conversation (a subset of search features).
        available_features: Columns in ``xstar_simulator_view`` that have at
            least one non-NA value.
        simulator_persona: Free-text profile string passed to the user
            simulator (purchase history summary, demographics, etc.).
        historical_data: Mapping from item ID to text representation of the
            user's past interactions.
        historical_ids: Ordered list of item IDs in the user's history.
        historical_df: DataFrame of historical items with an added
            ``user_rating_of_5`` column.
        baseline_queries: List of vector-search query strings used as baselines.
        z0_baseline_query: Vector-search query derived from ``z0``.
        zstar_baseline_query: Vector-search query derived from ``zstar``.
        z0_baseline_ids: Top-k item IDs retrieved using ``z0_baseline_query``.
        zstar_baseline_ids: Top-k item IDs retrieved using
            ``zstar_baseline_query``.
        ustar: Combined (column + embedding) utility function.
        column_ustar: Column-matching-only utility function.
        embedding_ustar: Embedding-based utility function (``None`` if the
            vector search server is unavailable).
    """

    def __init__(
        self,
        dataset_name: str,
        index: str,
        version: Optional[str] = None,
        item_name: Optional[str] = None,
        z0: Optional[str] = None,
        zstar: Optional[str] = None,
        baseline_queries: Optional[List[str]] = None,
        name: Optional[str] = None,
        state_files: Optional[List[str]] = None,
        files_to_clean: Optional[List[str]] = None,
        # Fixed-spec parameters
        xstar: Optional[List[str]] = None,
        xstar_series: Optional[pd.DataFrame] = None,
        xstar_simulator_view: Optional[pd.DataFrame] = None,
        sec_split: Optional[Dict[str, List[str]]] = None,
        initial_known_features: Optional[List[str]] = None,
        simulator_persona: Optional[str] = None,
        historical_data: Optional[Dict[str, str]] = None,
        historical_ids: Optional[List[str]] = None,
        historical_ratings: Optional[List[float]] = None,
        catalog: Optional[pd.DataFrame] = None,
        column_descriptions: Optional[Dict[str, str]] = None,
        special_match_functions: Optional[List[str]] = None,
        representation: Optional[Any] = None,
        z0_baseline_query: Optional[str] = None,
        zstar_baseline_query: Optional[str] = None,
        z0_baseline_ids: Optional[List[str]] = None,
        zstar_baseline_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise a Specification.

        When ``xstar_simulator_view`` is provided the constructor automatically
        builds all z-variants via :func:`build_z_variants_for_spec` and wires
        up both a column-matching and an embedding-based utility function.

        Args:
            dataset_name: Identifier of the parent dataset.
            index: Position index of this spec in the dataset.
            item_name: Singular noun for catalog items (e.g. ``"book"``).
            z0: Pre-built z0 string (used when ``xstar_simulator_view`` is
                ``None``).
            zstar: Pre-built zstar string (used when ``xstar_simulator_view``
                is ``None``).
            baseline_queries: List of query strings for vector-search baselines.
            name: Optional human-readable name for this spec.
            state_files: Paths to files whose content should be captured and
                restored by :meth:`get_state` / :meth:`load_state`.
            files_to_clean: Paths to temporary files deleted on ``__del__``.
            xstar: List of ground-truth item IDs.
            xstar_series: Full catalog rows of the ground-truth items.
            xstar_simulator_view: Simulator-visible rows of the ground-truth
                items (vagueified, hidden features removed).
            sec_split: SEC partition mapping
                ``{"search": [...], "experience": [...], "credence": [...]}``.
            initial_known_features: Columns already known to the user at the
                start of the conversation.
            simulator_persona: Free-text user profile passed to the simulator.
            historical_data: Item-ID → text-representation mapping of past
                user interactions.
            historical_ids: Ordered item IDs in the user's history.
            historical_ratings: Per-item ratings (parallel to
                ``historical_ids``), on a 0–5 scale.
            catalog: Full item catalog DataFrame (used to build
                ``historical_df`` and utility functions).
            column_descriptions: Mapping from column name to human-readable
                feature description (shown to the simulator). May be renamed
                to ``feature_descriptions`` in a future release.
            special_match_functions: Column-level match-function overrides
                passed to :class:`~coshop.data.utility.ColumnMatchingUtilityFunction`.
            representation: Item representation object (unused by the
                constructor; stored on caller request).
            z0_baseline_query: Pre-computed vector-search query from z0.
            zstar_baseline_query: Pre-computed vector-search query from zstar.
            z0_baseline_ids: Pre-retrieved top-k IDs for ``z0_baseline_query``.
            zstar_baseline_ids: Pre-retrieved top-k IDs for
                ``zstar_baseline_query``.
            **kwargs: Additional key-value pairs stored as instance attributes.
        """
        self.dataset_name = dataset_name
        self.version = version
        self.index = index
        self.item_name = item_name
        self.baseline_queries = baseline_queries or []
        self.name = name
        self.state_files = state_files
        self.files_to_clean = files_to_clean

        if xstar_simulator_view is not None:
            # Build z variants from the simulator view
            z_variants = build_z_variants_for_spec(
                target_items_simulator_view=xstar_simulator_view,
                sec_split=sec_split,
                initial_known_features=initial_known_features,
                column_descriptions=column_descriptions,
                item_name=item_name,
            )
            self.z0 = z_variants["z0"]
            self.zs = z_variants["zs"]
            self.zse = z_variants["zse"]
            self.zstar = z_variants["zstar"]

            assert xstar is not None, (
                "xstar should be set to the id(s) of the ground truth item(s)"
            )
            assert xstar_series is not None, (
                "xstar_series should be set to the dataframe of the ground truth item(s)"
            )
            assert isinstance(xstar, list), "xstar must be a list of item ids"
            assert isinstance(xstar_series, pd.DataFrame), (
                "xstar_series must be a DataFrame"
            )

            self.xstar = xstar
            self.xstar_series = xstar_series
            self.xstar_simulator_view = xstar_simulator_view

            non_null_columns = [
                col
                for col in xstar_simulator_view.columns
                if (xstar_simulator_view[col] != OPEN_TO_ANYTHING).any()
            ]
            self.available_features = non_null_columns

            # SEC partition over columns in xstar_simulator_view; built by dataset-specific code.
            # Expected format: {"search": [...], "experience": [...], "credence": [...]}.
            self.sec_split: Dict[str, List[str]] = sec_split or {}
            self.initial_known_features = initial_known_features
            self.simulator_persona = simulator_persona

            # Historical user data. Restrict to items present in the (possibly
            # subsetted) catalog so downstream .loc lookups never raise; with the
            # full catalog this is a no-op. historical_ids and historical_ratings
            # are filtered together to stay aligned.
            self.historical_data = historical_data
            catalog_index_set = set(catalog.index)
            _pairs = [
                (hid, hr)
                for hid, hr in zip(historical_ids or [], historical_ratings or [])
                if hid in catalog_index_set
            ]
            kept_ids = [hid for hid, _ in _pairs]
            kept_ratings = [hr for _, hr in _pairs]
            self.historical_ids = kept_ids
            self.historical_df = catalog.loc[kept_ids]
            self.historical_df["user_rating_of_5"] = pd.Series(
                kept_ratings, index=kept_ids
            )

            # Baseline queries and retrieved ids from vector search (if available)
            self.z0_baseline_query: Optional[str] = z0_baseline_query
            self.zstar_baseline_query: Optional[str] = zstar_baseline_query
            self.z0_baseline_ids: List[str] = z0_baseline_ids or []
            self.zstar_baseline_ids: List[str] = zstar_baseline_ids or []

            # Set up ustars
            base_column_ustar = ColumnMatchingUtilityFunction(
                xstar=xstar_series,
                catalog=catalog,
                cols_to_compare=non_null_columns,
                special_match_functions=special_match_functions,
            )
            try:
                base_embedding_ustar = ServerCosinePercentileUtilityFunction(
                    xstar=xstar,
                    dataset_name=self.dataset_name,
                    catalog=catalog,
                    version=self.version,
                )
                self.ustar = ColumnMatchWithServerCosinePercentileUtilityFunction(
                    column_matching_uf=base_column_ustar,
                    server_cosine_uf=base_embedding_ustar,
                )
                # Expose column-only and server-only scorers as lambdas
                self.column_ustar = base_column_ustar
                self.embedding_ustar = base_embedding_ustar
            except Exception:
                # Fallback: only column-based ustar is available
                self.ustar = base_column_ustar
                self.column_ustar = base_column_ustar
                self.embedding_ustar = None
        else:
            self.z0 = z0
            self.zstar = zstar

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"Specification(name={self.name})"

    ################ state ################

    def get_state(self) -> Dict[str, Any]:
        """Serialise the current tool state for checkpointing.

        Reads the content of every path in :attr:`state_files` and returns a
        dict that can be passed to :meth:`load_state` to restore the same
        state later.

        Returns:
            A dict ``{"filenames": [...], "file_contents": [...]}`` where each
            entry in ``file_contents`` is the file's text or ``None`` if the
            file does not yet exist.  Returns ``{}`` if :attr:`state_files` is
            ``None``.
        """
        if self.state_files is None:
            return {}

        def _read(path):
            if not os.path.exists(path):
                return None
            return open(path, "r").read()

        return {
            "file_contents": [_read(f) for f in self.state_files],
            "filenames": self.state_files,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore tool state from a previously serialised snapshot.

        Writes each file listed in ``state["filenames"]`` with the
        corresponding content from ``state["file_contents"]``.  Entries whose
        content is ``None`` are skipped.

        Args:
            state: A dict as returned by :meth:`get_state`.
        """
        self.state_files = state.get("filenames", [])
        for f, contents in zip(self.state_files, state.get("file_contents", [])):
            if contents is None:
                continue
            with open(f, "w") as f:
                f.write(contents)

    def __del__(self) -> None:
        # Clean up files
        if hasattr(self, "files_to_clean") and self.files_to_clean is not None:
            for file in self.files_to_clean:
                try:
                    os.remove(file)
                except Exception:
                    pass


#########################################################

# Asset download settings
DOWNLOAD_SETTINGS = {
    "chunk_size": 8192,
    "timeout": 600,  # 10 minutes
}


class Dataset:
    """Abstract base class for a coshop benchmark dataset.

    A Dataset is an iterable collection of :class:`Specification` objects
    backed by a shared item catalog.  Specs are loaded lazily by default;
    call :meth:`load_specs` to pre-load a batch.

    Subclasses must implement:
        * :attr:`dataset_name` (property)
        * :attr:`dataset_description` (property)
        * :attr:`item_name` (property)
        * :attr:`special_match_functions` (property)
        * :meth:`_load_specs`
        * :meth:`get_image_fn`
        * :meth:`render_item_fn`
    """

    def __init__(
        self,
        max_xstar: Optional[int] = 1,
        restrict_representation_to_simulator_columns: bool = True,
        max_search_features: Optional[int] = None,
        dropout_extra_search: bool = True,
        true_features: Optional[Dict[str, str]] = None,
        hidden_features: Optional[List[str]] = None,
        filterable_features: Optional[List[str]] = None,
        catalog: Optional[pd.DataFrame] = None,
        popularity_df: Optional[pd.DataFrame] = None,
        identifying_cols: Optional[List[str]] = None,
    ) -> None:
        """Initialise a Dataset.

        Args:
            max_xstar: Maximum number of acceptable ground-truth items per
                spec.  Specs with more ground-truth items are discarded.
                Defaults to ``1``.
            restrict_representation_to_simulator_columns: When ``True``
                (default), the item representation shown to the agent only
                includes columns visible to the simulator (i.e. excluding
                ``hidden_features``).  Set to ``False`` to expose all catalog
                columns.
            max_search_features: Optional cap on the number of *search*-
                category features available per spec.  Works together with
                ``dropout_extra_search`` to modify the effective SEC split.
                ``None`` disables the cap.
            dropout_extra_search: Controls what happens to search features
                that exceed ``max_search_features``.  When ``True`` (default)
                the extra features are masked to ``OPEN_TO_ANYTHING`` in the
                simulator's view (i.e. dropped entirely).  When ``False`` they
                are re-categorised as *experience* features instead.
            true_features: Mapping from column name to human-readable feature
                description used by the item representation.  May be renamed
                to ``feature_descriptions`` in a future release.
            hidden_features: Column names that should not be visible to the
                user simulator.  These columns are excluded from the simulator
                catalog and the restricted item representation.
            filterable_features: Column names on which the agent's retrieval
                tools may apply hard equality filters.
            catalog: Pre-loaded catalog DataFrame.  Subclasses normally pass
                this after loading from disk.
            popularity_df: Optional DataFrame mapping item IDs to popularity
                scores, used by the popularity-ordered retrieval mode.
            identifying_cols: Columns (beyond ``"id"``) that uniquely identify
                an item and should always appear in representations (e.g.
                ``["title", "artist"]``).
        """
        self.max_xstar = max_xstar

        # Optional cap on the number of search features the simulator can use per spec.
        self._max_search_features = max_search_features
        self._dropout_extra_search = dropout_extra_search

        self.true_features = true_features
        self.filterable_features = filterable_features

        self.catalog = catalog
        self.popularity_df = popularity_df

        # Simulator view of catalog (visible features only, tags exploded, vagueification applied).
        self.simulator_catalog = self._build_simulator_catalog(
            hidden_features=hidden_features
        )
        
        self.identifying_cols = identifying_cols or []
        if "id" not in self.identifying_cols:
            self.identifying_cols.insert(0, "id")
        
        
        self.representation_restricted = ParagraphRepresentation(
            feature_descriptions=true_features,
            restricted_columns=(
                self.identifying_cols + list(self.simulator_catalog.columns)
            )
        )
        self.representation_unrestricted = ParagraphRepresentation(
            feature_descriptions=true_features,
        )
        self.representation = (
            self.representation_restricted
            if restrict_representation_to_simulator_columns
            else self.representation_unrestricted
        )

    def _vagueify_simulator_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dataset-specific vagueification to the simulator catalog.

        Intentionally reduces precision of feature values to approximate what
        a real user would know or remember (e.g. rounding prices to the
        nearest $10, replacing exact counts with coarse buckets).  Override
        in subclasses.  The default implementation is a no-op.

        Args:
            df: Subset of the catalog with hidden features already removed.

        Returns:
            The vagueified DataFrame.
        """
        return df

    def _build_simulator_catalog(
        self, hidden_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Build the simulator's restricted view of the catalog.

        Removes ``hidden_features`` columns then applies
        :meth:`_vagueify_simulator_catalog`.  The resulting columns correspond
        to the keys used in the SEC split.

        Args:
            hidden_features: Column names to exclude from the simulator view.

        Returns:
            A DataFrame with hidden columns removed and vagueification applied.
        """
        if self.catalog is None or self.catalog.empty:
            return pd.DataFrame()

        visible = [c for c in self.catalog.columns if c not in hidden_features]
        if not visible:
            return pd.DataFrame(index=self.catalog.index)
        df = self.catalog[visible].copy()
        df = self._vagueify_simulator_catalog(df)
        return df

    def _load_vector_search_baselines(self) -> Dict[str, Dict[str, Any]]:
        """
        Load vector-search-based baseline queries and top-k ids for fixed specs, if available.

        Expected JSON format (per dataset):
            {
              "0": {
                "z0":   {"query": "<baseline query from z0>",   "top_k_ids": ["id1", ..., "id6"]},
                "zstar":{"query": "<baseline query from zstar>","top_k_ids": ["id1", ..., "id6"]}
              },
              ...
            }
        """
        filename = (getattr(self, "_config", None) or {}).get(
            "baseline_queries", "baseline_queries.json"
        )
        json_path = ROOT_DIR / self.dataset_name / "assets" / filename
        if not json_path.exists():
            return {}
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def get_vector_search_baselines_for_spec(
        self, index: int
    ) -> Dict[str, Any]:
        """Return pre-computed vector-search baseline data for a spec.

        Lazily loads ``data/<dataset_name>/assets/baseline_queries.json`` on
        first call and caches the result.

        Args:
            index: Spec index (zero-based within the dataset split).

        Returns:
            A dict with keys ``"z0"`` and ``"zstar"``, each mapping to
            ``{"query": str, "top_k_ids": [str, ...]}``.  Returns ``{}`` if
            no baseline data is available for the given index.
        """
        cache_attr = "_vector_search_baselines_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, self._load_vector_search_baselines())
        all_data: Dict[str, Any] = getattr(self, cache_attr)
        return all_data.get(str(index), {})

    def _apply_max_search_features_for_spec(
        self,
        *,
        index: int,
        sec_split: Dict[str, List[str]],
        target_items_simulator_view: pd.DataFrame,
        initial_known_features: Optional[List[str]],
    ) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        Optionally cap the number of search features used for a fixed spec.

        All initial-known search features are always kept. Remaining search
        features are randomly subsampled so that total search features do not exceed self._max_search_features.

        Behavior for the extra (dropped) search features:
          - If self._dropout_extra_search is True:
              * They are removed from the SEC split entirely, and their values
                are set to NA in the simulator's view of the target items.
          - If self._dropout_extra_search is False:
              * They are moved from search to experience in the SEC split, and
                their values remain visible to the simulator as experience
                features.
        """
        if self._max_search_features is None:
            return sec_split, target_items_simulator_view

        search_cols = list(sec_split.get("search", []))
        if not search_cols:
            return sec_split, target_items_simulator_view

        limit = self._max_search_features
        if limit is None or limit >= len(search_cols):
            return sec_split, target_items_simulator_view

        # Initial-known features are always kept (filtered to search-only here).
        initial_known_features = initial_known_features or []
        initial_search = [c for c in initial_known_features if c in search_cols]

        # Remaining search features that are not initial-known.
        remaining_search = [c for c in search_cols if c not in initial_search]

        if not remaining_search and len(initial_search) <= limit:
            return sec_split, target_items_simulator_view

        # Number of extra (non-initial) search features we are allowed to keep.
        max_extra_to_keep = max(0, limit - len(initial_search))
        if max_extra_to_keep <= 0:
            extra_kept: List[str] = []
        else:
            rng = random.Random(index)

            # Prefer search features whose target values are not all the NA
            # placeholder OPEN_TO_ANYTHING in the simulator view.
            non_na_remaining: List[str] = []
            na_like_remaining: List[str] = []
            for col in remaining_search:
                if col in target_items_simulator_view.columns:
                    col_values = target_items_simulator_view[col]
                    # Treat a feature as non-NA if it has at least one value
                    # that is not the OPEN_TO_ANYTHING placeholder.
                    has_non_na = (col_values != OPEN_TO_ANYTHING).any()
                else:
                    has_non_na = False

                if has_non_na:
                    non_na_remaining.append(col)
                else:
                    na_like_remaining.append(col)

            if len(remaining_search) <= max_extra_to_keep:
                extra_kept = remaining_search
            else:
                # First sample from non-NA features where possible, then
                # backfill with NA-like features if we still need more.
                if len(non_na_remaining) >= max_extra_to_keep:
                    extra_kept = rng.sample(non_na_remaining, max_extra_to_keep)
                else:
                    extra_kept = list(non_na_remaining)
                    remaining_slots = max_extra_to_keep - len(extra_kept)
                    if na_like_remaining and remaining_slots > 0:
                        extra_kept.extend(
                            rng.sample(
                                na_like_remaining,
                                min(remaining_slots, len(na_like_remaining)),
                            )
                        )

        kept_search_set = set(initial_search) | set(extra_kept)
        kept_search = [c for c in search_cols if c in kept_search_set]
        extra_search = [c for c in search_cols if c not in kept_search_set]

        if not extra_search:
            return sec_split, target_items_simulator_view

        adjusted_view = target_items_simulator_view.copy()

        if self._dropout_extra_search:
            # Drop extra search features entirely from simulator view for this spec.
            # Simulator catalogs use OPEN_TO_ANYTHING instead of NA, so mirror that here.
            for col in extra_search:
                if col in adjusted_view.columns:
                    adjusted_view[col] = OPEN_TO_ANYTHING
            # Do not modify sec_split: extra search features remain categorized as search.
            return sec_split, adjusted_view
        else:
            # Convert extra search features into experience features.
            # Copy SEC split lists so we do not mutate callers' dictionaries.
            new_sec_split: Dict[str, List[str]] = {
                "search": list(kept_search),
                "experience": list(sec_split.get("experience", [])),
                "credence": list(sec_split.get("credence", [])),
            }
            experience_cols = new_sec_split.setdefault("experience", [])
            for col in extra_search:
                if col not in experience_cols:
                    experience_cols.append(col)

            return new_sec_split, adjusted_view

    def get_initial_known_features_for_spec(
        self, index: int
    ) -> Optional[List[str]]:
        """
        Load initial known features for a fixed spec from
        data/<dataset_name>/assets/incremental_search_recall_results_new.csv.
        Same logic as in expert_user._load_initial_features_from_incremental:
        resolve feature_names (direct column names and "tag:foo") to the
        visible simulator columns (including exploded tag column names).
        """
        inc_df = self.initial_known_features
        row = inc_df[inc_df.get("user_idx") == index]
        if row.empty:
            return []
        raw = row.iloc[0].get("feature_names")
        if not isinstance(raw, str):
            return []
        try:
            names = ast.literal_eval(raw)
        except Exception:
            return []
        if not isinstance(names, list):
            return []

        # Visible columns: simulator_catalog; if "tags" present, add exploded tag names (non-hidden).
        visible_cols = list(self.simulator_catalog.columns)
        selected: Set[str] = set()
        for n in names:
            if isinstance(n, str) and n in visible_cols:
                selected.add(n)
        for n in names:
            if not isinstance(n, str) or not n.startswith("tag:"):
                continue
            tag = n[len("tag:") :].strip()
            if tag and tag in visible_cols:
                selected.add(tag)
        if not selected:
            return []

        return [c for c in visible_cols if c in selected]

    def get_image_fn(
        self, id: str, return_image_url: bool = False
    ) -> Union[str, Image.Image]:
        """Return the image for a catalog item.

        Args:
            id: Catalog item ID.
            return_image_url: When ``True``, return a base64-encoded JPEG
                string (or a URL string) instead of a PIL Image.

        Returns:
            A PIL :class:`~PIL.Image.Image` when ``return_image_url`` is
            ``False``, or a base64/URL string otherwise.
        """
        raise NotImplementedError

    def render_item_fn(
        self,
        id: str,
        show_features: bool = True,
        *,
        return_html: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Render a catalog item for display.

        Args:
            id: Catalog item ID.
            show_features: Whether to include feature text alongside the image.
            return_html: When ``True``, return an HTML-serialisable payload
                dict ``{"title": str, "body_html": str, "image_url": str}``
                suitable for embedding in an iframe instead of rendering
                directly in Streamlit.
            **kwargs: Additional keyword arguments passed to the underlying
                render implementation.

        Returns:
            Streamlit render output, or an HTML payload dict when
            ``return_html=True``.
        """
        raise NotImplementedError

    ################ collection-level methods ################

    @property
    def dataset_name(self) -> str:
        """Programmatic identifier for this dataset (e.g. ``"hm"``).

        Used to route API requests and locate dataset-specific asset files.
        """
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """One-sentence description of the dataset shown in documentation and UIs."""
        raise NotImplementedError

    @property
    def item_name(self) -> str:
        """Singular noun for items in this catalog (e.g. ``"clothing item"``,
        ``"movie"``, ``"book"``).

        Used to build natural-language prompts for the user simulator.
        """
        raise NotImplementedError

    @property
    def special_match_functions(self) -> Dict[str, str]:
        """Column-level match-function overrides for the utility scorer.

        Returns a dict mapping column name to the name of the match function it
        requires (e.g. ``"jaccard_similarity"`` for multi-valued tags, or a
        numeric comparison for price ranges).  Columns not listed use exact
        match.  Passed to
        :class:`~coshop.data.utility.ColumnMatchingUtilityFunction`.
        """
        raise NotImplementedError

    @property
    def assets_file_id(self) -> str:
        return None

    def __repr__(self) -> str:
        return f"Dataset(name={self.dataset_name}, specs={len(self.specs)})"

    def __del__(self) -> None:
        if hasattr(self, "specs"):
            for spec in self.specs.values():
                if hasattr(spec, "__del__"):
                    spec.__del__()

    def _finish_init(self) -> None:
        """Validate required attributes and build the spec index.

        Must be called by subclasses at the end of their ``__init__`` after
        setting ``self.length``, ``self.catalog``, and ``self.representation``.
        Initialises ``self.specs`` as a dict mapping spec index → ``None``
        (populated lazily on first access).

        Raises:
            ValueError: If any required attribute is ``None``.
        """
        for attr in [
            "length",
            "catalog",
            "representation",
        ]:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} is not set")

        # Store the total fixed length before applying the test-set cap
        self._total_length = self.length

        # Test set: use first 100 (or all if fewer than 100 are available),
        # keyed by their actual file indexes (0-99).
        self.length = min(TEST_SET_SIZE, self._total_length)
        self.specs = {i: None for i in range(self.length)}

    def load_specs(
        self, indexes: Optional[List[int]] = None, reload: bool = False
    ) -> None:
        """Pre-load a batch of specs into memory.

        Args:
            indexes: List of spec indices to load.  Defaults to all indices in
                the current split.
            reload: When ``True``, reload specs that are already cached.
                Defaults to ``False``.

        Raises:
            ValueError: If any requested index does not belong to the current
                dataset split.
        """
        if indexes is None:
            print(f"Loading all {self.length} specs")
            indexes = list(self.specs.keys())

        if any(i not in self.specs for i in indexes):
            raise ValueError(f"Indexes {indexes} not found in dataset")

        if not reload:
            # remove already loaded specs
            indexes = [i for i in indexes if self.specs[i] is None]

        # Use indexes directly (they're already the actual file indexes)
        loaded_specs = self._load_specs(indexes=indexes)
        self.specs.update(loaded_specs)

    def _load_specs(self, **kwargs: Any) -> Dict[int, Specification]:
        """Load and return a dict of :class:`Specification` objects.

        Must be implemented by each dataset subclass.  Called by
        :meth:`load_specs` with the list of indices that need to be loaded.

        Args:
            **kwargs: At minimum ``indexes: List[int]`` is passed.  Subclasses
                may accept additional keyword arguments.

        Returns:
            A dict mapping spec index → :class:`Specification`.
        """
        raise NotImplementedError

    def _ensure_assets_available(self) -> None:
        """Ensure dataset assets are downloaded and available.

        Assets are hosted on Google Drive and downloaded on first use. A
        ``.download_complete`` sentinel is written into the assets directory
        once a download finishes successfully, so an interrupted/partial
        download is never mistaken for a complete one and is re-attempted on
        the next instantiation. In an editable/source checkout where assets are
        already present, the sentinel is created up front so no download fires.
        """
        if self.assets_file_id is None:
            # no assets to download for this dataset
            return

        assets_dir = ROOT_DIR / self.dataset_name / "assets"
        marker = assets_dir / ".download_complete"

        # Already have a verified-complete set of assets.
        if marker.exists():
            return

        # Source checkout: assets shipped alongside the code. Mark complete and
        # skip the download.
        config_present = (assets_dir / f"{self.version}_config.yml").exists() or any(
            assets_dir.glob("catalog_v*.csv")
        ) if assets_dir.exists() else False
        if config_present:
            marker.touch()
            return

        # Otherwise, download from Google Drive. Only mark complete once the
        # download+unzip actually succeeds, so a failed/partial attempt is
        # re-tried on the next instantiation instead of being cached as done.
        print(f"Downloading assets for {self.dataset_name}")
        try:
            ok = download_file_from_google_drive(
                self.assets_file_id, str(assets_dir), unzip=True, **DOWNLOAD_SETTINGS
            )
        except Exception as e:
            ok = False
            print(f"Download failed for {self.dataset_name}: {e}")
        if not ok:
            raise RuntimeError(
                f"Failed to download assets for {self.dataset_name!r} "
                f"(Drive file id {self.assets_file_id!r}). Ensure the file's "
                "sharing is set to 'Anyone with the link' and is reachable."
            )
        marker.touch()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key: Union[str, int], load_on_demand: bool = True) -> Specification:
        if isinstance(key, int):
            ix = key
        else:
            key = str(key)
            assert key.startswith(""), "Key must be an int or start with ''"
            ix = int(key.split("_")[1])

        assert ix in self.specs, (
            f"Index {ix} is out of bounds for test fixed specs. "
            f"Test set only includes 0 to {self.length - 1}"
        )

        spec = self.specs.get(ix, None)
        if spec is None and load_on_demand:
            self.load_specs(indexes=[ix])
            spec = self.specs[ix]
        return spec

    def __iter__(self) -> Iterator[Specification]:
        return iter(list(self.specs.values()))


################## HELPER FUNCTIONS ################


def build_z_variants_for_spec(
    target_items_simulator_view: pd.DataFrame,
    sec_split: Dict[str, List[str]],
    initial_known_features: Optional[List[str]],
    column_descriptions: Dict[str, str],
    item_name: str,
) -> Dict[str, str]:
    """Build the four z-variant preference descriptions for a spec.

    The z-variants are natural-language strings that describe the user's
    preferences at increasing levels of completeness, following the SEC
    (Search / Experience / Credence) classification of item features:

    * ``z0``   — only the initially known features (a subset of search features).
    * ``zs``   — z0 plus all remaining *search* features.
    * ``zse``  — zs plus all *experience* features.
    * ``zstar`` — full ground-truth preferences: search + experience + credence.

    The strings are built by progressively revealing features to a
    :class:`~coshop.user_simulator.helpers.feature_tracker.FeatureTracker` and calling
    :meth:`~coshop.user_simulator.helpers.feature_tracker.FeatureTracker.get_known_context`.

    Args:
        target_items_simulator_view: DataFrame of the ground-truth items as
            seen by the simulator (one row per xstar item, vagueified).
        sec_split: Mapping ``{"search": [...], "experience": [...],
            "credence": [...]}`` partitioning columns by SEC category.
        initial_known_features: Columns the simulated user knows before the
            conversation begins (must be a subset of search features).
        column_descriptions: Mapping from column name to human-readable
            feature description used to format the preference text.
        item_name: Singular noun for catalog items (e.g. ``"book"``), used
            when generating the preference text.

    Returns:
        A dict with keys ``"z0"``, ``"zs"``, ``"zse"``, and ``"zstar"``,
        each mapping to a formatted preference description string.
    """
    # Lazy import to avoid circular dependency: FeatureTracker imports this module.
    from ..user_simulator.helpers.feature_tracker import FeatureTracker  # type: ignore

    search_cols = sec_split.get("search", []) or []
    experience_cols = sec_split.get("experience", []) or []
    credence_cols = sec_split.get("credence", []) or []

    ft = FeatureTracker(
        target_df=target_items_simulator_view,
        search_features=search_cols,
        experience_features=experience_cols,
        credence_features=credence_cols,
        max_features_to_reveal=None,
        verbosity=0,
        column_descriptions=column_descriptions,
        item_name=item_name,
    )

    # Step 1: initial-known (search) features -> z0
    if initial_known_features:
        # categories=None: reveal regardless of SEC category, matching hm logic.
        ft.reveal_features(initial_known_features, categories=None)  # type: ignore[arg-type]
    z0 = ft.get_known_context(drop_na_vals=True)

    # Step 2: reveal all remaining search features -> zs
    remaining_search_cols = [f.column_name for f in ft.unknown_search_features]
    if remaining_search_cols:
        ft.reveal_features(remaining_search_cols, categories=["search"])  # type: ignore[arg-type]
    zs = ft.get_known_context(drop_na_vals=True)

    # Step 3: reveal all remaining experience features -> zse
    remaining_experience_cols = [
        f.column_name for f in ft.experience_features if not f.known
    ]
    if remaining_experience_cols:
        ft.reveal_features(remaining_experience_cols, categories=["experience"])  # type: ignore[arg-type]
    zse = ft.get_known_context(drop_na_vals=True)

    # Step 4: reveal any remaining (typically credence) features -> zstar
    remaining_cols = [f.column_name for f in ft.unknown_features]
    if remaining_cols:
        ft.reveal_features(remaining_cols, categories=None)  # type: ignore[arg-type]
    zstar = ft.get_known_context(drop_na_vals=True)

    return {"z0": z0, "zs": zs, "zse": zse, "zstar": zstar}
