"""MovieLens movie dataset for the coshop benchmark.

Builds benchmark tasks from MovieLens user rating histories.  The held-out
movie(s) a user rated most recently become xstar; earlier highly-rated movies
form the historical context shown to the simulator.

The catalog is sourced from TMDB metadata merged with MovieLens ratings.
Poster images are fetched lazily from TMDB at render time.

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
import random
from PIL import Image
from ..dataset import (
    Dataset,
    Specification,
)
from ..representation import Representation, representation_text_to_html
import html as _html
from collections import Counter

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

# TMDB base URL for poster images
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"

# ISO 639-1 (2-letter) code to language name for original_language display
_ISO639_1_TO_NAME = {
    "ab": "Abkhazian",
    "aa": "Afar",
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "as": "Assamese",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "bm": "Bambara",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bi": "Bislama",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "dv": "Divehi",
    "nl": "Dutch",
    "dz": "Dzongkha",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "ee": "Ewe",
    "fo": "Faroese",
    "fj": "Fijian",
    "fi": "Finnish",
    "fr": "French",
    "ff": "Fulah",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gn": "Guaraní",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hu": "Hungarian",
    "is": "Icelandic",
    "io": "Ido",
    "ig": "Igbo",
    "id": "Indonesian",
    "ia": "Interlingua",
    "iu": "Inuktitut",
    "ik": "Inupiaq",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "kl": "Kalaallisut",
    "kn": "Kannada",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "kk": "Kazakh",
    "km": "Khmer",
    "ki": "Kikuyu",
    "rw": "Kinyarwanda",
    "ky": "Kyrgyz",
    "kv": "Komi",
    "kg": "Kongo",
    "ko": "Korean",
    "ku": "Kurdish",
    "kj": "Kuanyama",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lg": "Ganda",
    "li": "Limburgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lu": "Luba-Katanga",
    "lv": "Latvian",
    "gv": "Manx",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mh": "Marshallese",
    "mn": "Mongolian",
    "na": "Nauru",
    "nv": "Navajo",
    "nd": "North Ndebele",
    "ne": "Nepali",
    "ng": "Ndonga",
    "no": "Norwegian",
    "nb": "Norwegian Bokmål",
    "nn": "Norwegian Nynorsk",
    "ii": "Sichuan Yi",
    "oc": "Occitan",
    "oj": "Ojibwa",
    "or": "Odia",
    "om": "Oromo",
    "os": "Ossetic",
    "pi": "Pali",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "qu": "Quechua",
    "ro": "Romanian",
    "rm": "Romansh",
    "rn": "Rundi",
    "ru": "Russian",
    "se": "Northern Sami",
    "sm": "Samoan",
    "sg": "Sango",
    "sr": "Serbian",
    "gd": "Scottish Gaelic",
    "sn": "Shona",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "st": "Southern Sotho",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "ss": "Swati",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ty": "Tahitian",
    "tg": "Tajik",
    "ta": "Tamil",
    "tt": "Tatar",
    "te": "Telugu",
    "th": "Thai",
    "bo": "Tibetan",
    "ti": "Tigrinya",
    "to": "Tongan",
    "ts": "Tsonga",
    "tn": "Tswana",
    "tr": "Turkish",
    "tk": "Turkmen",
    "tw": "Twi",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "ve": "Venda",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "cy": "Welsh",
    "wo": "Wolof",
    "fy": "Western Frisian",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "za": "Zhuang",
    "zu": "Zulu",
}


def _iso639_1_to_language_name(code: str):
    """Map ISO 639-1 (2-letter) code to language name; return code unchanged if unknown."""
    if not code or not isinstance(code, str):
        return code
    key = code.strip().lower()[:2]
    return _ISO639_1_TO_NAME.get(key, code)

INITIAL_FEATURES = ["adult", "genres", "spoken_languages"]



class MovieLensDataset(Dataset):
    """MovieLens movie recommendation dataset.

    Catalog: ~9k movies with structured TMDB metadata (genres, year, runtime,
    ratings, cast tags, etc.).  Tasks: each task is built from one user's
    rating history; a held-out high-rated movie becomes xstar and prior
    high-rated movies form the historical context.

    Assets (catalog, transactions) are downloaded from Google Drive on first
    instantiation if not already present in ``data/movielens/assets/``.
    """

    @property
    def dataset_name(self) -> str:
        return "movielens"

    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **get movie recommendations based on your preferences.**"

    @property
    def item_name(self) -> str:
        return "movie"

    @property
    def special_match_functions(self) -> Dict[str, str]:
        return dict(self._config.get("special_match_functions") or {})

    @property
    def assets_file_id(self) -> str:
        return "1Voy2hJReKdY3-G4FVsL5U_gPXLO__XlR"

    def _vagueify_simulator_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce precision of simulator-visible feature values.

        Applied transforms (matching the logic in
        :func:`_apply_movielens_feature_aggregation`):

        * **original_language** — ISO 639-1 code mapped to full language name.
        * **popularity** — raw score replaced with a percentile-based label
          (e.g. ``"very popular"``, ``"obscure"``).
        * **vote_average / vote_count / budget / revenue / runtime / year** —
          rounded and prefixed with ``>=`` or ``<=`` inequality symbols.
        * **NA values** — filled with ``"open to anything"``.
        """
        out = df.copy()
        # original_language: ISO 639-1 code -> language name
        if "original_language" in out.columns:
            out["original_language"] = out["original_language"].apply(
                lambda v: _iso639_1_to_language_name(v) if pd.notna(v) else v
            )
        # Popularity: percentile-based label (e.g. "very popular", "obscure")
        if "popularity" in out.columns:
            pop_series = out["popularity"].dropna()
            out["popularity"] = out["popularity"].apply(
                lambda v: _popularity_percentile_label(v, pop_series)
            )
        agg_features = [
            "vote_average",
            "vote_count",
            "budget",
            "revenue",
            "runtime",
            "year",
            "adult",
        ]
        for col in agg_features:
            if col not in out.columns:
                continue
            out[col] = out[col].apply(
                lambda v, c=col: _apply_movielens_feature_aggregation(
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
        """Initialise the MovieLens dataset.

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
        catalog_df["genres"] = catalog_df["genres"].apply(lambda x: ", ".join(eval(x)))
        catalog_df["spoken_languages"] = catalog_df["spoken_languages"].apply(
            lambda x: ", ".join(eval(x))
        )
        catalog_df["production_companies"] = catalog_df["production_companies"].apply(
            lambda x: ", ".join(eval(x))
        )
        catalog_df["production_countries"] = catalog_df["production_countries"].apply(
            lambda x: ", ".join(eval(x))
        )
        catalog_df["belongs_to_collection"] = catalog_df["belongs_to_collection"].apply(
            lambda x: ", ".join(eval(x))
        )
        # add in synthetic tags as all having 10 votes
        catalog_df["synthetic_tags"] = catalog_df["synthetic_tags"].apply(
            lambda x: {k: 10 for k in eval(x)}
        )
        catalog_df["tags"] = catalog_df.apply(
            lambda row: json.dumps({**eval(row["tags"]), **row["synthetic_tags"]}),
            axis=1,
        )
        # Treat budget/revenue 0 as missing (same as NA)
        for col in ("budget", "revenue"):
            if col in catalog_df.columns:
                catalog_df[col] = catalog_df[col].replace(0, pd.NA)
        # Convert vote_avg from stars out of 10 to -> stars out of 5
        catalog_df["vote_average"] = catalog_df["vote_average"] / 2

        # drop synthetic tags column
        catalog_df = catalog_df.drop(columns=["synthetic_tags"])

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
        if "popularity" in catalog_df.columns:
            popularity_df = pd.DataFrame(
                {"popularity": catalog_df["popularity"]}, index=catalog_df.index
            )
            popularity_df.index = popularity_df.index.astype(str)
            popularity_df = popularity_df.loc[self.catalog.index]

        # Load true features
        true_features = json.load(
            open(f"{assets_dir}/{self._config['feature_descriptions']}")
        )

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
            identifying_cols=[
                "id",
                "title",
            ],
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
                name=f"movielens_{ix}",
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
        """Return the TMDB poster for a movie.

        Args:
            id: Catalog item ID string.
            return_image_url: When ``True``, return the full TMDB URL string
                instead of fetching and decoding the image.
            max_px: Maximum width/height for the returned PIL Image thumbnail.

        Returns:
            A TMDB URL string when ``return_image_url=True``; a PIL Image when
            ``False`` and the fetch succeeds; or ``""`` on any error.
        """
        try:
            # Get poster_path from catalog
            if id not in self.catalog.index:
                return ""
            poster_path = self.catalog.loc[id].get("poster_path", "")
            if pd.isna(poster_path) or not poster_path:
                return ""

            # Construct full TMDB URL
            if poster_path.startswith("/"):
                image_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            else:
                image_url = f"{TMDB_IMAGE_BASE_URL}/{poster_path}"

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
            of movie ID strings from ``{idx}_items.txt``.
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
        """Build historical-data structures from a user's movie rating history.

        Args:
            user_info: Parsed user JSON as returned by
                :meth:`load_transaction_data`.
            filter_rating_threshold: When set, only movies with a rating
                ``>= filter_rating_threshold`` are included in
                ``historical_ids``.  ``None`` includes all rated movies.

        Returns:
            A tuple ``(historical_data, historical_ids, historical_ratings)``
            where ``historical_data`` is a ``{movie_id: "I rated it N/5.0
            stars"}`` dict, ``historical_ids`` is a list of movie ID strings
            (filtered by threshold), and ``historical_ratings`` is the
            corresponding list of float ratings.
        """
        historical_data: Dict[str, str] = {}
        historical_ids: List[str] = []
        historical_ratings: List[float] = []
        previous_ratings = user_info.get("previous_ratings", [])
        for rating_entry in previous_ratings:
            if isinstance(rating_entry, dict):
                movie_id = str(rating_entry.get("id", ""))
                rating = rating_entry.get("rating", "")
                if movie_id and rating and movie_id in self.catalog.index:
                    historical_data[movie_id] = f"I rated it {rating}/5.0 stars"
                    if (
                        filter_rating_threshold is None
                        or float(rating) >= filter_rating_threshold
                    ):
                        historical_ids.append(movie_id)
                        historical_ratings.append(float(rating))
        return historical_data, historical_ids, historical_ratings

    def get_simulator_profile(
        self,
        user_info: dict,
        max_historical_ratings: int = 5,
        include_historical_item_list: bool = True,
    ) -> str:
        """Build the simulator persona string for one user.

        Includes total rating count, fraction of high ratings, top genres and
        languages, and optionally an inline list of recently-rated movies.

        Args:
            user_info: Parsed user JSON from :meth:`load_transaction_data`.
            max_historical_ratings: Maximum number of movies listed inline
                when ``include_historical_item_list=True``.  Defaults to ``5``.
            include_historical_item_list: When ``False``, omits the per-movie
                list (used for simulators that retrieve history via the history
                search tool).

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
                f"I have rated {len(previous_ratings)} movies previously, with {len(high_ratings)} movies rated 4.0 or higher."
            )

            try:
                genre_counts = Counter(
                    [g for r in previous_ratings for g in eval(r.get("genres", "[]"))]
                )
                language_counts = Counter(
                    [
                        l
                        for r in previous_ratings
                        for l in eval(r.get("spoken_languages", "[]"))
                    ]
                )
                parts.append(
                    f"My most frequent genres are: {', '.join([g for g, _ in genre_counts.most_common(3)])}"
                )
                parts.append(
                    f"My most frequent languages are: {', '.join([l for l, _ in language_counts.most_common(3)])}"
                )
            except Exception:
                pass

            if include_historical_item_list:
                parts.append("\nMovies I have recently rated:")
            for rating_entry in (
                previous_ratings[:max_historical_ratings]
                if include_historical_item_list
                else []
            ):
                if isinstance(rating_entry, dict):
                    movie_id = str(rating_entry.get("id", ""))
                    rating = rating_entry.get("rating", "")
                    if movie_id in catalog.index:
                        item_row = catalog.loc[movie_id]
                        movie_title = item_row.get("title", f"Movie {movie_id}")
                        year = item_row.get("year", "")
                        genres = item_row.get("genres", "")
                        vote_average = item_row.get("vote_average", "")

                        movie_desc = f"- {movie_title}"
                        if year and pd.notna(year):
                            movie_desc += f" ({int(year)})"
                        if genres:
                            movie_desc += f" [{', '.join(genres[:2])}]"  # Limit to 2 genres
                        if pd.notna(vote_average):
                            movie_desc += f" [Avg rating: {vote_average:.1f}/10]"
                        movie_desc += f" - My rating: {rating}/5.0 stars"
                        parts.append(movie_desc)

                        overview = item_row.get("overview", "")
                        if overview and pd.notna(overview):
                            parts.append(f"\t{overview}")
                    else:
                        parts.append(f"- Movie {movie_id} - My rating: {rating}/5.0 stars")

        if parts:
            text = "\n".join(parts)
        else:
            text = ""

        return text


def _popularity_percentile_label(value: float, popularity_series: pd.Series) -> str:
    """Map a raw popularity value to a percentile-based label for zstar display.
    Uses catalog popularity distribution: top 10% -> 'very popular', bottom 10% -> 'obscure'.
    """
    if popularity_series is None or len(popularity_series) == 0 or pd.isna(value):
        return str(value)
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    # Fraction of catalog with popularity strictly less than this value
    pct = (popularity_series < val).sum() / len(popularity_series) * 100.0
    if pct >= 95:
        return "blockbuster-level popular"
    if pct >= 85:
        return "very popular"
    if pct >= 70:
        return "popular"
    if pct >= 55:
        return "above average"
    if pct >= 45:
        return "average"
    if pct >= 30:
        return "below average"
    if pct >= 15:
        return "less well-known"
    if pct >= 5:
        return "little-known"
    return "rarely known"


def _format_movielens_feature_value(
    feature: str, value: Any, popularity_series: Optional[pd.Series] = None
) -> str:
    """Format a single feature value for movies (used by constraint message builders).
    Numeric fields are vague-ified (rounded) to avoid leaking exact values.
    For popularity, when popularity_series is provided, uses percentile-based labels.
    """
    if feature == "vote_average":
        try:
            rating = float(value)
            rating_rounded = math.floor(rating * 2) / 2
            return f">= {rating_rounded:.1f}"
        except Exception:
            return str(value)
    if feature == "vote_count":
        try:
            num_ratings = float(value)
            num_ratings_rounded = math.floor(num_ratings / 100) * 100
            return f">= {num_ratings_rounded:,}"
        except Exception:
            return str(value)
    if feature == "popularity":
        if popularity_series is not None and len(popularity_series) > 0:
            try:
                return ">= " + _popularity_percentile_label(
                    float(value), popularity_series
                )
            except (TypeError, ValueError):
                pass
        try:
            pop = float(value)
            pop_rounded = math.floor(pop * 10) / 10
            return f">= {pop_rounded:.1f}"
        except Exception:
            return str(value)
    if feature in ["budget", "revenue"]:
        try:
            amt = float(value)
            amt_rounded = math.floor(amt / 100000) * 100000
            return f">= {amt_rounded:,}"
        except Exception:
            return str(value)
    if feature == "runtime":
        try:
            runtime = float(value)
            # Round up to nearest half-hour (30 min)
            runtime_rounded = math.ceil(runtime / 30) * 30
            return f"<= {int(runtime_rounded)}"
        except Exception:
            return str(value)
    if feature == "year":
        try:
            return _format_year_to_20yr_block_label(value)
        except Exception:
            return str(value)
    if feature == "adult":
        return "Yes" if value else "No"
    return str(value)


def _apply_movielens_feature_aggregation(
    feature: str,
    value: Any,
    include_inequalities: bool = False,
) -> Any:
    """
    Apply MovieLens-specific aggregation (rounding) to a feature value.
    This matches the logic in _format_movielens_feature_value.

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
        # Popularity currently does not appear in agg_features, so we can safely
        # omit popularity_series here.
        return _format_movielens_feature_value(feature, value)

    try:
        if feature == "vote_average":
            rating = float(value)
            rating_rounded = math.floor(rating * 2) / 2
            return str(rating_rounded)
        elif feature == "vote_count":
            num_ratings = float(value)
            num_ratings_rounded = math.floor(num_ratings / 100) * 100
            return str(int(num_ratings_rounded))
        elif feature == "popularity":
            pop = float(value)
            pop_rounded = math.floor(pop * 10) / 10
            return str(pop_rounded)
        elif feature in ["budget", "revenue"]:
            amt = float(value)
            amt_rounded = math.floor(amt / 100000) * 100000
            return str(int(amt_rounded))
        elif feature == "runtime":
            runtime = float(value)
            runtime_rounded = math.ceil(runtime / 30) * 30
            return str(int(runtime_rounded))
        elif feature == "year":
            year = float(value)
            year_rounded = math.floor(year / 20) * 20
            return str(int(year_rounded))
        elif feature == "adult":
            return "Yes" if value else "No"
    except (ValueError, TypeError):
        pass

    return value


def _format_decade_label(decade_start: int) -> str:
    """Format a decade start year as a full label like '1980s' or '2000s'."""
    return f"{decade_start}s"


def _format_year_to_20yr_block_label(value: Any) -> str:
    """
    Format a year-like value into a 20-year phrase made of two adjacent decades.
    Example outputs: '1970s or 1980s', '1980s or 1990s'.
    """
    year = int(float(value))
    decade_start = (year // 10) * 10
    if random.choice([True, False]):
        left, right = decade_start - 10, decade_start
    else:
        left, right = decade_start, decade_start + 10
    return f"{_format_decade_label(left)} or {_format_decade_label(right)}"

def make_render_item_fn(
    catalog: pd.DataFrame,
    get_image_fn: Callable[..., Any],
    representation: Representation,
) -> Callable[..., Any]:
    """Create the ``render_item_fn`` closure for MovieLens movies.

    Args:
        catalog: Full catalog DataFrame (index = movie IDs).
        get_image_fn: Callable mapping a movie ID to a TMDB URL or PIL Image.
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
                "title": f"Movie {id}",
                "body_html": "<div style='opacity:0.85;'>Movie not found in catalog.</div>",
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

        title = item.get("title", "Unknown Movie")
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
            else "<div style='font-style:italic;opacity:0.8;margin:6px 0 8px;'>Poster not available</div>"
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