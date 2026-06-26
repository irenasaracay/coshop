"""Goodreads book dataset for the coshop benchmark.

Builds benchmark tasks from Goodreads user rating histories.  The held-out
book(s) become xstar; earlier highly-rated books form the historical context.

The catalog contains structured book metadata (genres, author, ratings,
series info, thematic tags, etc.) merged with Goodreads review snippets.
Cover images are fetched lazily from their stored URLs at render time.

Per-version configuration (catalog/transactions paths, hidden-from-simulator
columns, filterable features, special match functions, and the
``restrict_representation_to_simulator_columns`` default) lives in
``v1_config.yml`` / ``v2_config.yml`` alongside this module. Pass
``version="v1"`` or ``version="v2"`` (default ``"v2"``) at construction to
select between them.
"""

from typing import List, Optional, Dict, Any, Union, Tuple, Callable
import os
import pandas as pd
import json
import glob
import io
import math
from PIL import Image
from ..dataset import (
    Dataset,
    Specification,
)
from ..representation import representation_text_to_html, Representation
import html as _html

from collections import Counter
from ...utils.misc import is_same_image

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))



class GoodreadsDataset(Dataset):
    """Goodreads book recommendation dataset.

    Catalog: ~10k books with structured metadata (genres, author, series,
    rating statistics, thematic tags, tropes, etc.).  Tasks: each task is
    built from one user's reading/rating history.

    Assets (catalog, transactions) are downloaded from Google Drive on first
    instantiation if not already present in ``data/goodreads/assets/``.
    A stock placeholder image (``assets/stock_image.png``) is used to detect
    and filter out uninformative cover images fetched from stored URLs.
    """

    @property
    def dataset_name(self) -> str:
        return "goodreads"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **get book recommendations based on your preferences.**"

    @property
    def item_name(self) -> str:
        return "book"
    
    @property
    def special_match_functions(self) -> Dict[str, str]:
        return dict(self._config.get("special_match_functions") or {})

    @property
    def assets_file_id(self) -> str:
        return "1m2VCWp2C9GHbUzTFqcKZeLVm-KwSsL5c"

    def _vagueify_simulator_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce precision of simulator-visible feature values.

        Applied transforms:

        * **genres** — truncated to the first 3 genres.
        * **series_works_count / series_position** — mapped to coarse buckets
          (e.g. ``"4-10"``, ``"mid-series"``).
        * **average_rating / ratings_count / text_reviews_count /
          publication_year / num_pages** — rounded and formatted with
          inequality prefixes (``>=`` / ``<=``).
        * **NA values** — filled with ``"open to anything"``.
        """
        out = df.copy()
        # Genres: subset to first 3 when the list is longer than 3
        if "genres" in out.columns:
            out["genres"] = out["genres"].apply(
                lambda v: (
                    ", ".join([g.strip() for g in str(v).split(",") if g.strip()][:3])
                    if pd.notna(v) and str(v).strip()
                    else v
                )
            )
        agg_features = [
            "series_works_count",
            "series_position",
            "average_rating",
            "ratings_count",
            "text_reviews_count",
            "publication_year",
            "num_pages",
            "is_in_series",
            "is_ebook",
        ]
        for col in agg_features:
            if col not in out.columns:
                continue
            out[col] = out[col].apply(
                lambda v, c=col: _apply_goodreads_feature_aggregation(
                    c, v, include_inequalities=True
                )
            )
        out = out.fillna("open to anything")
        return out

    def __init__(
        self,
        indexes: Optional[List[int]] = None,
        subset_fraction: Optional[float] = None,
        version: str = "v2",
        **kwargs,
    ) -> None:
        """Initialise the Goodreads dataset.

        Args:
            indexes: Optional list of spec indices to pre-load.  ``None``
                loads specs lazily.
            subset_fraction: Fraction of the catalog to retain.  xstar and selected historical items are always
                kept.  ``None`` uses the full catalog.
            **kwargs: Forwarded to :class:`~coshop.data.dataset.Dataset`.
        """
        from .. import load_dataset_config
        self._config = load_dataset_config(DATASET_ROOT, version)
        self.version = self._config.get("version", version)
        # Download assets (catalog, transactions, ...) on first use if needed.
        self._ensure_assets_available()
        assets_dir = f"{DATASET_ROOT}/assets"
        # Load catalog
        catalog_df = pd.read_csv(
            f"{assets_dir}/{self._config['catalog']}", index_col="id"
        )
        catalog_df["genres"] = catalog_df["genres"].fillna("")
        catalog_df["is_ebook"] = catalog_df["is_ebook"].replace(False, pd.NA)

        # add in synthetic tags as all having 10 votes
        catalog_df["synthetic_tags"] = catalog_df["synthetic_tags"].apply(
            lambda x: {k: 10 for k in eval(x)}
        )
        catalog_df["tags"] = catalog_df.apply(
            lambda row: json.dumps({**eval(row["tags"]), **row["synthetic_tags"]}),
            axis=1,
        )
        catalog_df = catalog_df.drop(columns=["synthetic_tags"])

        # Binary feature: is_in_series (True if book is part of a series)
        if "series_works_count" in catalog_df.columns:
            catalog_df["is_in_series"] = (
                catalog_df["series_works_count"].fillna(0).astype(float) > 0
            )
        elif "series_name" in catalog_df.columns:
            sn = catalog_df["series_name"].fillna("").astype(str).str.strip()
            catalog_df["is_in_series"] = sn != ""
        else:
            catalog_df["is_in_series"] = False

        # Int feature: position in series
        catalog_df["series_position"] = (
            catalog_df["title"]
            .str.extract(r"\((.*?)#(\d+)\)", expand=False)[1]
            .astype(float)
        )

        # Ensure catalog index (IDs) are strings
        catalog_df.index = catalog_df.index.astype(str)
        if subset_fraction is not None:
            from ..subset_catalog import subset_catalog_from_transactions
            from ..dataset import TEST_SET_SIZE
            catalog_df = subset_catalog_from_transactions(
                catalog_df,
                subset_fraction,
                self.transactions_dir,
                test_set_size=TEST_SET_SIZE,
                user_suffix="_user.json",
                history_extractor=lambda info: [
                    str(r.get("id", ""))
                    for r in info.get("previous_ratings", [])
                    if isinstance(r, dict) and r.get("id") and r.get("rating")
                ],
            )
        self.catalog = catalog_df

        # Extract popularity column to create popularity_df
        popularity_df = None
        if "ratings_count" in catalog_df.columns:
            popularity_df = pd.DataFrame(
                {"popularity": catalog_df["ratings_count"]}, index=catalog_df.index
            )
            popularity_df.index = popularity_df.index.astype(str)
            popularity_df = popularity_df.loc[self.catalog.index]

        # Initial known features
        initial_known_features_path = (
            f"{assets_dir}/{self._config['initial_known_features']}"
        )
        if os.path.exists(initial_known_features_path):
            initial_known_features = pd.read_csv(initial_known_features_path)
            self.initial_known_features = initial_known_features
        else:
            self.initial_known_features = pd.DataFrame(
                columns=["user_idx", "feature_names"]
            )

        self.sec_split = json.load(open(f"{assets_dir}/{self._config['sec_split']}"))

        # Load true features
        true_features = json.load(
            open(f"{assets_dir}/{self._config['feature_descriptions']}")
        )

        # Load the stock image to compare against
        self._stock_image = Image.open(f"{assets_dir}/stock_image.png")

        kwargs.setdefault(
            "restrict_representation_to_simulator_columns",
            self._config.get("restrict_representation_to_simulator_columns", True),
        )
        super().__init__(
            true_features=true_features,
            catalog=catalog_df,
            popularity_df=popularity_df,
            filterable_features=list(self._config["filterable_features"]),
            hidden_features=list(self._config["hidden_from_simulator"]),
            identifying_cols=["id", "title"],
            **kwargs,
        )
        self.render_item_fn = make_render_item_fn(
            self.catalog, self.get_image_fn, self.representation
        )

        # Load the fixed information
        self.length = len(
            glob.glob(f"{self.transactions_dir}/*_items.txt")
        )

        # All subclasses must have these attributes set
        self._finish_init()

        if indexes is not None:
            self.load_specs(indexes=indexes)

    def _load_specs(
        self, indexes: Optional[List[int]] = None
    ) -> Dict[int, Specification]:
        if indexes is None:
            return {}
        # Load the transaction data
        specs = {}
        for ix in indexes:
            user_info, xstars = self.load_transaction_data(ix)
            # DataFrame of target items (multiple possible)
            xstars = [str(x) for x in xstars]
            # If max_xstar is set, limit to that many items; otherwise keep all
            if self.max_xstar is not None:
                xstars = xstars[: self.max_xstar]

            target_items = self.catalog.loc[xstars]
            target_items_simulator_view = self.simulator_catalog.loc[xstars]
            sec_split = self.sec_split[str(ix)]

            # Use FeatureTracker to build z-variants so they match get_known_context.
            # Use the same search/experience/credence split as the simulator.
            initial_known_features = self.get_initial_known_features_for_spec(ix)
            sec_split, target_items_simulator_view = (
                self._apply_max_search_features_for_spec(
                    index=ix,
                    sec_split=sec_split,
                    target_items_simulator_view=target_items_simulator_view,
                    initial_known_features=initial_known_features,
                )
            )
            historical_data, historical_ids, historical_ratings = self.build_historical_data(user_info)

            # Vector-search baseline queries and ids (if available)
            vs_baselines = self.get_vector_search_baselines_for_spec(ix)
            z0_baseline = (
                vs_baselines.get("z0", {}) if isinstance(vs_baselines, dict) else {}
            )
            zstar_baseline = (
                vs_baselines.get("zstar", {}) if isinstance(vs_baselines, dict) else {}
            )

            spec = Specification(
                dataset_name=self.dataset_name,
                version=self.version,
                index=f"{ix}",
                item_name=self.item_name,
                xstar=xstars,
                xstar_series=target_items,
                xstar_simulator_view=target_items_simulator_view,
                sec_split=sec_split,
                initial_known_features=initial_known_features,
                column_descriptions=self.true_features,
                name=f"goodreads_{ix}",
                baseline_queries=[z0_baseline.get("query")],
                simulator_persona=self.get_simulator_profile(
                    user_info, include_historical_item_list=False
                ),
                historical_data=historical_data,
                historical_ids=historical_ids,
                historical_ratings=historical_ratings,
                special_match_functions=self.special_match_functions,
                catalog=self.catalog,
                representation=self.representation,
                z0_baseline_query=z0_baseline.get("query"),
                zstar_baseline_query=zstar_baseline.get("query"),
                z0_baseline_ids=z0_baseline.get("top_k_ids") or [],
                zstar_baseline_ids=zstar_baseline.get("top_k_ids") or [],
            )
            specs[ix] = spec
        return specs

    def get_image_fn(
        self, id: str, return_image_url: bool = False, max_px: int = 256
    ) -> Union[str, Image.Image]:
        """Return the cover image for a book.

        Fetches the image from the ``image_url`` stored in the catalog.
        Images that are pixel-identical to the stock placeholder
        (``assets/stock_image.png``) are filtered out and treated as missing.

        Args:
            id: Catalog item ID string.
            return_image_url: When ``True``, return the raw URL string without
                fetching the image.
            max_px: Maximum thumbnail dimension in pixels.

        Returns:
            A URL string, a PIL Image, or ``""`` / ``None`` when unavailable.
        """
        try:
            # Get image_url from catalog
            if id not in self.catalog.index:
                return ""
            image_url = self.catalog.loc[id].get("image_url", "")
            if pd.isna(image_url) or not image_url:
                return ""

            if return_image_url:
                return image_url
            else:
                # Could fetch and return PIL Image here if needed
                from PIL import Image
                import requests

                try:
                    response = requests.get(image_url, timeout=5)
                    if response.status_code == 200:
                        img = Image.open(io.BytesIO(response.content)).convert("RGB")

                        # check if the image is very close to the stock image
                        if is_same_image(img, self._stock_image):
                            return None

                        img.thumbnail((max_px, max_px))
                        return img
                except Exception:
                    pass
                return None
        except Exception:
            return ""

    @property
    def transactions_dir(self) -> str:
        return os.path.join(DATASET_ROOT, "assets", self._config["transactions_dir"])

    def load_transaction_data(self, example_idx) -> Tuple[dict, List[str]]:
        """Load raw transaction files for one benchmark task.

        Args:
            example_idx: Spec index (integer).

        Returns:
            A tuple ``(user_info, target_ids)`` where ``user_info`` is the
            parsed JSON from ``{idx}_user.json`` and ``target_ids`` is a list
            of book ID strings from ``{idx}_items.txt``.
        """
        user_file = os.path.join(self.transactions_dir, f"{example_idx}_user.json")
        with open(user_file, "r") as f:
            user_info = json.load(f)

        items_file = os.path.join(self.transactions_dir, f"{example_idx}_items.txt")
        with open(items_file, "r") as f:
            lines = f.read().splitlines()
        if len(lines) == 0:
            raise ValueError(f"Empty items file: {items_file}")

        # Line 1 is comma-separated list of target ids
        target_ids = [s.strip() for s in lines[0].split(",") if s.strip()]
        return user_info, [str(x) for x in target_ids]

    def build_historical_data(
        self, user_info: dict, filter_rating_threshold: float = None
    ) -> Tuple[Dict[str, str], List[str], List[float]]:
        """Build historical-data structures from a user's book rating history.

        Each entry in ``historical_data`` includes the rating and, when
        available, the user's review snippet.

        Args:
            user_info: Parsed user JSON from :meth:`load_transaction_data`.
            filter_rating_threshold: When set, only books with rating
                ``>= filter_rating_threshold`` appear in ``historical_ids``.

        Returns:
            A tuple ``(historical_data, historical_ids, historical_ratings)``
            where ``historical_data`` maps book ID to a text like
            ``"I rated it 4/5.0 stars. My review: `...`"``.
        """
        historical_data: Dict[str, str] = {}
        historical_ids: List[str] = []
        historical_ratings: List[float] = []
        previous_ratings = user_info.get("previous_ratings", [])
        for rating_entry in previous_ratings:
            if isinstance(rating_entry, dict):
                book_id = str(rating_entry.get("id", ""))
                rating = rating_entry.get("rating", "")
                review = rating_entry.get("review_text", "")
                if book_id and rating and book_id in self.catalog.index:
                    desc_parts = [f"I rated it {rating}/5.0 stars"]
                    if review and pd.notna(review):
                        desc_parts.append(f"My review: `{review}`")
                    historical_data[book_id] = ". ".join(desc_parts)

                    if (
                        filter_rating_threshold is None
                        or float(rating) >= filter_rating_threshold
                    ):
                        historical_ids.append(book_id)
                        historical_ratings.append(float(rating))
        return historical_data, historical_ids, historical_ratings

    def get_simulator_profile(
        self,
        user_info: dict,
        max_historical_ratings: int = 5,
        include_historical_item_list: bool = True,
    ) -> str:
        """Build the simulator persona string for one user.

        Includes total rating count, high-rating fraction, top genres and
        authors, and optionally an inline list of recently-rated books with
        review snippets.

        Args:
            user_info: Parsed user JSON from :meth:`load_transaction_data`.
            max_historical_ratings: Maximum books listed inline when
                ``include_historical_item_list=True``.  Defaults to ``5``.
            include_historical_item_list: When ``False``, omits the per-book
                list (for simulators using the history search tool).

        Returns:
            A free-text persona string passed to the user simulator.
        """
        catalog = self.catalog
        parts = []

        previous_ratings = user_info.get("previous_ratings", [])
        if previous_ratings:
            high_ratings = [
                r
                for r in previous_ratings
                if isinstance(r, dict) and r.get("rating", 0) >= 4.0
            ]
            parts.append(
                f"I have rated {len(previous_ratings)} books previously, with {len(high_ratings)} books rated 4.0 or higher."
            )

            try:
                genre_counts = Counter(
                    [
                        g
                        for r in previous_ratings
                        for g in (
                            r.get("genres", "").split(", ") if r.get("genres") else []
                        )
                    ]
                )
                author_counts = Counter(
                    [
                        r.get("author_name", "")
                        for r in previous_ratings
                        if r.get("author_name")
                    ]
                )
                parts.append(
                    f"My most frequent genres are: {', '.join([g for g, _ in genre_counts.most_common(3)])}"
                )
                parts.append(
                    f"My most frequent authors are: {', '.join([a for a, _ in author_counts.most_common(3)])}"
                )
            except Exception:
                pass

            if include_historical_item_list:
                parts.append("\nBooks I have recently rated:")
            for rating_entry in (
                previous_ratings[:max_historical_ratings]
                if include_historical_item_list
                else []
            ):
                if isinstance(rating_entry, dict):
                    book_id = str(rating_entry.get("id", ""))
                    rating = rating_entry.get("rating", "")
                    review = rating_entry.get("review_text", "")
                    if book_id in catalog.index:
                        item_row = catalog.loc[book_id]
                        book_title = item_row.get("title", f"Book {book_id}")
                        author = item_row.get("author_name", "")
                        genres = item_row.get("genres", "")
                        average_rating = item_row.get("average_rating", "")

                        book_desc = f"- {book_title}"
                        if author and pd.notna(author):
                            book_desc += f" by {author}"
                        if genres:
                            genre_list = [
                                g.strip() for g in str(genres).split(",") if g.strip()
                            ]
                            book_desc += f" [{', '.join(genre_list[:2])}]"  # Limit to 2 genres
                        if pd.notna(average_rating):
                            book_desc += f" [Avg rating: {average_rating:.2f}/5.0]"
                        book_desc += f" [My rating: {rating}/5.0 stars]"
                        parts.append(book_desc)

                        description = item_row.get("description", "")
                        if description and pd.notna(description):
                            parts.append(f"\t{description}")

                        if review and pd.notna(review):
                            parts.append(f"\tMy review: `{review}`")
                    else:
                        parts.append(f"- Book {book_id} - My rating: {rating}/5.0 stars")

        if parts:
            text = "\n".join(parts)
        else:
            text = "I am looking for book recommendations."

        return text


def _series_position_to_bucket(value: Any) -> str:
    """Map series_position to display bucket: 1st in the series or mid-series."""
    try:
        pos = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if pos <= 1:
        return "1st in the series"
    return "mid-series (not 1st in series, could be any other position)"


def _series_works_count_to_bucket(value: Any) -> str:
    """Map series_works_count to display bucket: 1, 2, 3, 4-10, 10+."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if 2 <= n <= 3:
        return "2-3"
    if 4 <= n <= 10:
        return "4-10"
    return "10+"


def _format_goodreads_feature_value(feature: str, value: Any) -> str:
    """Format a single feature value for books (used by constraint message builders).
    Numeric fields are vague-ified (rounded) to avoid leaking exact values.
    """
    if feature == "series_works_count":
        return _series_works_count_to_bucket(value)
    if feature == "series_position":
        return _series_position_to_bucket(value)
    if feature == "average_rating":
        try:
            rating = float(value)
            rating_rounded = math.floor(rating * 2) / 2
            return f">= {rating_rounded:.2f}"
        except Exception:
            return str(value)
    if feature in ["ratings_count", "text_reviews_count"]:
        try:
            num_ratings = float(value)
            num_ratings_rounded = math.floor(num_ratings / 100) * 100
            return f">= {num_ratings_rounded:,}"
        except Exception:
            return str(value)
    if feature == "text_reviews_count":
        try:
            num_reviews = float(value)
            num_reviews_rounded = math.floor(num_reviews / 100) * 100
            return f">= {num_reviews_rounded:,}"
        except Exception:
            return str(value)
    if feature == "publication_year":
        try:
            year = float(value)
            # Round down to century (units of 100)
            year_rounded = math.floor(year / 100) * 100
            return f">= {int(year_rounded)}"
        except Exception:
            return str(value)
    if feature == "num_pages":
        try:
            pages = float(value)
            # Round up to nearest 100
            pages_rounded = math.ceil(pages / 100) * 100
            return f"<= {int(pages_rounded)}"
        except Exception:
            return str(value)
    if feature == "is_in_series":
        if value is None or pd.isna(value):
            return "open to anything"
        return "Yes" if value else "No"
    if feature == "is_ebook":
        if value is None or pd.isna(value):
            return "open to anything"
        return "Yes" if value else "No"
    return str(value)


def _apply_goodreads_feature_aggregation(
    feature: str,
    value: Any,
    include_inequalities: bool = False,
) -> Any:
    """
    Apply Goodreads-specific aggregation (rounding) to a feature value.
    This matches the logic in _format_goodreads_feature_value.

    Args:
        feature: The feature name
        value: The raw feature value

    Returns:
        The aggregated/rounded value (as a string for numeric values, or the original type)
    """
    if pd.isna(value) or value is None:
        return value

    # When requested (e.g., for simulator catalog vague-ification), reuse the
    # human-readable formatting that already includes >= / <= symbols.
    if include_inequalities:
        return _format_goodreads_feature_value(feature, value)

    try:
        if feature == "series_works_count":
            return _series_works_count_to_bucket(value)
        if feature == "series_position":
            return _series_position_to_bucket(value)
        if feature == "average_rating":
            rating = float(value)
            rating_rounded = math.floor(rating * 2) / 2
            return str(rating_rounded)
        elif feature in ["ratings_count", "text_reviews_count"]:
            num_ratings = float(value)
            num_ratings_rounded = math.floor(num_ratings / 100) * 100
            return str(int(num_ratings_rounded))
        elif feature == "text_reviews_count":
            num_reviews = float(value)
            num_reviews_rounded = math.floor(num_reviews / 100) * 100
            return str(int(num_reviews_rounded))
        elif feature == "publication_year":
            year = float(value)
            year_rounded = math.floor(year / 100) * 100
            return str(int(year_rounded))
        elif feature == "num_pages":
            pages = float(value)
            pages_rounded = math.ceil(pages / 100) * 100
            return str(int(pages_rounded))
        elif feature == "is_in_series":
            return "Yes" if value else "No"
        elif feature == "is_ebook":
            return "Yes" if value else "No"
    except (ValueError, TypeError):
        pass

    return value


def make_render_item_fn(
    catalog: pd.DataFrame,
    get_image_fn: Callable[..., Any],
    representation: Representation,
) -> Callable[..., Any]:
    """Create the ``render_item_fn`` closure for Goodreads books.

    Args:
        catalog: Full catalog DataFrame (index = book IDs).
        get_image_fn: Callable mapping a book ID to a cover URL or PIL Image.
        representation: Item representation for feature text formatting.

    Returns:
        A render function (same contract as :func:`coshop.data.hm.data.make_render_item_fn`).
    """

    def esc(x: Any) -> str:
        return _html.escape("" if x is None else str(x))

    def render_item_fn(
        id: str,
        show_features: bool = True,
        width: int = 200,
        *,
        return_html: bool = False,
        image_thumb_px: int = 256,
        **_: Any,
    ) -> Any:
        try:
            item = catalog.loc[str(id)]
        except (KeyError, IndexError):
            payload = {
                "title": f"Book {id}",
                "body_html": "<div style='opacity:0.85;'>Book not found in catalog.</div>",
                "image_url": "",
            }
            if return_html:
                return payload
            import streamlit as st
            st.markdown(
                f"<div><div style='font-weight:600;margin-bottom:6px;'>{esc(payload['title'])}</div>{payload['body_html']}</div>",
                unsafe_allow_html=True,
            )
            return None

        title = item.get("title", "Unknown Book")
        image_url = get_image_fn(str(item.name), return_image_url=True, max_px=image_thumb_px)

        if show_features:
            rep_text = representation.row_to_str(item)
            body_html = representation_text_to_html(rep_text)
        else:
            body_html = ""

        payload = {
            "title": str(title),
            "body_html": body_html,
            "image_url": image_url,
        }

        if return_html:
            return payload

        import streamlit as st
        img_html = (
            f"<img src='{esc(image_url)}' style='width:{int(width)}px;height:auto;display:block;margin:0 auto 8px auto;'/>"
            if image_url
            else "<div style='font-style:italic;opacity:0.8;margin:6px 0 8px;'>Cover not available</div>"
        )
        full_html = (
            "<div>"
            f"<div style='font-weight:600;font-size:1.1em;margin-bottom:6px;'>{esc(payload['title'])}</div>"
            f"{img_html}"
            f"<div style='opacity:0.9;'>{payload['body_html']}</div>"
            "</div>"
        )
        st.markdown(full_html, unsafe_allow_html=True)
        return None

    return render_item_fn