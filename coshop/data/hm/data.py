"""H&M fashion dataset for the coshop benchmark.

Loads the H&M online-shop catalog and customer transaction data.  Each
benchmark task (``Specification``) is built from one customer's purchase
history: the most-recent purchase(s) become the ground-truth target (xstar)
and all earlier purchases form the historical context shown to the simulator.

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
import base64
import io
import math
from ..dataset import (
    Dataset,
    Specification,
    OPEN_TO_ANYTHING,
)
from ..representation import representation_text_to_html, Representation
from PIL import Image
import html as _html

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

class HMDataset(Dataset):
    """H&M fashion dataset.

    Catalog: ~105k clothing items with structured attributes (product type,
    colour, fit, price, fabric, etc.) and free-text product descriptions.
    Tasks: each task is built from one customer's transaction history.  The
    most-recent purchase(s) are held out as xstar; earlier purchases become
    the user's historical context.

    Assets are downloaded automatically from Google Drive on first instantiation
    if not already present in ``data/hm/assets/``.
    """

    @property
    def dataset_name(self) -> str:
        return "hm"
    @property
    def dataset_description(self) -> str:
        return "Work with the assistant to **shop for clothes from H&M Online.**"

    @property
    def item_name(self) -> str:
        return "clothing item"

    @property
    def special_match_functions(self) -> Dict[str, str]:
        return dict(self._config.get("special_match_functions") or {})

    @property
    def assets_file_id(self) -> str:
        return "1Psy7oOCYY9uoZW8vhUSXBGZpRAegvTIy"

    def _vagueify_simulator_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce precision of simulator-visible feature values.

        Applied transforms:

        * **price** — rounded up to the nearest $10 and formatted as ``<= $N``
          (the simulator sees an upper-bound, not the exact price).
        * **NA values** — filled with ``OPEN_TO_ANYTHING`` so the simulator
          treats missing features as unconstrained.
        """
        out = df.copy()
        # Price: round up to nearest $10
        try:
            out["price"] = out["price"].apply(
                lambda v: f"<= ${math.ceil(float(v) / 10) * 10}" if pd.notna(v) else v
            )
        except (TypeError, ValueError):
            pass
        out = out.fillna(OPEN_TO_ANYTHING)
        return out

    def __init__(
        self,
        indexes: Optional[List[int]] = None,
        subset_fraction: float = None,
        version: str = "v2",
        **kwargs,
    ) -> None:
        """Initialise the H&M dataset.

        Args:
            indexes: Optional list of spec indices to pre-load.  When ``None``
                specs are loaded lazily on first access.
            subset_fraction: If set, sub-samples the catalog to this fraction
                of its full size while guaranteeing that all xstar and selected
                historical items are retained.  Useful for fast development
                iteration.  ``None`` uses the full catalog.
            **kwargs: Forwarded to :class:`~coshop.data.dataset.Dataset`.
                Common options: ``max_xstar`` (int),
                ``max_search_features`` (int), ``dropout_extra_search`` (bool).
        """
        from .. import load_dataset_config
        self._config = load_dataset_config(DATASET_ROOT, version)
        self.version = self._config.get("version", version)
        # Download assets (catalog, transactions, images, ...) on first use if needed.
        self._ensure_assets_available()
        assets_dir = f"{DATASET_ROOT}/assets"
        catalog_path = f"{assets_dir}/{self._config['catalog']}"
        catalog_df = pd.read_csv(catalog_path, index_col="id")
        catalog_df["tags"] = catalog_df["tags"].apply(lambda x: ", ".join(eval(x)))
        # Add visual_desc column if missing (catalog.csv without visual descriptions)
        if "visual_desc" not in catalog_df.columns:
            catalog_df["visual_desc"] = pd.NA

        # Concatenate detail_desc and visual_desc into a single detail_desc column
        def _concat_desc(row):
            parts = []
            d = row.get("detail_desc")
            if pd.notna(d) and str(d).strip():
                parts.append(str(d).strip())
            v = row.get("visual_desc")
            if pd.notna(v) and str(v).strip():
                parts.append(str(v).strip())
            return " ".join(parts) if parts else pd.NA

        catalog_df["detail_desc"] = catalog_df.apply(_concat_desc, axis=1)
        catalog_df = catalog_df.drop(columns=["visual_desc"], errors="ignore")
        # Ensure catalog index (IDs) are strings
        catalog_df.index = catalog_df.index.astype(str)
        true_features = json.load(
            open(f"{assets_dir}/{self._config['feature_descriptions']}")
        )
        # Remove visual_desc from metadata (we merged it into detail_desc)
        true_features.pop("visual_desc", None)
        # Store catalog as DataFrame
        if subset_fraction is not None:
            from ..subset_catalog import subset_catalog_from_transactions
            from ..dataset import TEST_SET_SIZE
            catalog_df = subset_catalog_from_transactions(
                catalog_df,
                subset_fraction,
                self.transactions_dir,
                test_set_size=TEST_SET_SIZE,
                user_suffix="_customer.json",
                history_extractor=lambda info: [
                    str(p) for p in info.get("previous_purchases", [])
                ],
            )

        self.catalog = catalog_df

        # Extract popularity_df from kwargs if provided, otherwise will load it later
        popularity_path = f"{assets_dir}/{self._config['popularity']}"
        if os.path.exists(popularity_path):
            popularity_df = pd.read_csv(popularity_path, index_col="id")
            # Ensure popularity_df index (IDs) are strings
            popularity_df.index = popularity_df.index.astype(str)
            popularity_df = popularity_df.rename(columns={"num_users": "popularity"})
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
            identifying_cols=["id", "prod_name"],
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
            customer_info, xstars = self.load_transaction_data(ix)
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

            historical_data, historical_ids = self.build_historical_data(customer_info)
            historical_ratings = [5.0] * len(historical_ids)  # all ratings are 5.0

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
                name=f"hm_{ix}",
                baseline_queries=[z0_baseline.get("query")],
                simulator_persona=self.get_simulator_profile(
                    customer_info, include_historical_item_list=False
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
        """Return the product image for an H&M item.

        Images are stored locally as JPEG files under
        ``data/hm/assets/images/``.

        Args:
            id: Catalog item ID string.
            return_image_url: When ``True``, return a base64-encoded JPEG data
                URI (``data:image/jpeg;base64,...``).  When ``False``, return a
                PIL :class:`~PIL.Image.Image`.
            max_px: Maximum width/height in pixels (thumbnail constraint).
                Defaults to ``256``.

        Returns:
            A PIL Image, a base64 data URI string, or ``""`` if the image file
            does not exist or cannot be loaded.
        """
        try:
            img_path = f"{DATASET_ROOT}/assets/images/0{id[:2]}/0{id}.jpg"
            if not os.path.exists(img_path):
                return ""
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((max_px, max_px))
            if return_image_url:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80, optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:image/jpeg;base64,{b64}"
            else:
                return img
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
            A tuple ``(customer_info, target_ids)`` where ``customer_info`` is
            the parsed JSON from ``{idx}_customer.json`` and ``target_ids`` is
            a list of item ID strings read from ``{idx}_items.txt`` (first
            line, comma-separated).
        """
        customer_file = os.path.join(
            self.transactions_dir, f"{example_idx}_customer.json"
        )
        with open(customer_file, "r") as f:
            customer_info = json.load(f)

        items_file = os.path.join(self.transactions_dir, f"{example_idx}_items.txt")
        with open(items_file, "r") as f:
            lines = f.read().splitlines()
        if len(lines) == 0:
            raise ValueError(f"Empty items file: {items_file}")

        # Line 1 is comma-separated list of target ids; line 2 (if present) is t_dat.
        target_ids = [s.strip() for s in lines[0].split(",") if s.strip()]
        return customer_info, [str(x) for x in target_ids]

    def build_historical_data(
        self, customer_info: dict
    ) -> Tuple[Dict[str, str], List[str]]:
        """Build the historical-data structures from a customer's purchase history.

        The returned ``historical_data`` dict maps each item ID to a brief
        purchase note (``"I purchased this item."``).  Only items present in
        the current catalog (which may be a subset) are included.

        Args:
            customer_info: Parsed customer JSON as returned by
                :meth:`load_transaction_data`.

        Returns:
            A tuple ``(historical_data, historical_ids)`` where
            ``historical_data`` is a ``{item_id: text}`` dict and
            ``historical_ids`` is the raw list of purchase IDs (including any
            not in the catalog).
        """
        historical_data: Dict[str, str] = {}
        previous_purchases = customer_info.get("previous_purchases", [])
        for product_id in previous_purchases:
            product_id_str = str(product_id)
            if product_id_str in self.catalog.index:
                historical_data[product_id_str] = "I purchased this item."
        return historical_data, [str(x) for x in previous_purchases]

    def get_simulator_profile(
        self,
        customer_info: dict,
        max_historical_items: int = 5,
        include_historical_item_list: bool = True,
    ) -> str:
        """Build the simulator persona string for one customer.

        The profile includes age, club membership status, newsletter frequency,
        total number of prior purchases, average/max purchase price, and
        optionally a formatted list of past items.

        Args:
            customer_info: Parsed customer JSON as returned by
                :meth:`load_transaction_data`.
            max_historical_items: Maximum number of past items to describe
                inline when ``include_historical_item_list=True``.  Defaults
                to ``5``.
            include_historical_item_list: When ``True`` (default), appends a
                formatted list of past purchases.  Set to ``False`` for
                multi-turn simulators that retrieve history via the history
                search tool instead.

        Returns:
            A free-text persona string passed to the user simulator as
            ``simulator_persona``.
        """
        catalog = self.catalog
        parts = []
        age = customer_info.get("age")
        if age is not None and not pd.isna(age):
            parts.append(f"a {age} year old")
        club_member_status = customer_info.get("club_member_status")
        if club_member_status == "ACTIVE":
            parts.append("an H&M club member")
        gets_newsletter = customer_info.get("fashion_news_frequency")
        if gets_newsletter != "NONE" and gets_newsletter is not None:
            parts.append(f"read the H&M newsletter {gets_newsletter.lower()}")

        if len(parts):
            text = "I am " + " and ".join(parts) + "."
        else:
            text = ""

        # past purchases info -- some may not be in our filtered catalog, so filter first
        previous_purchases = customer_info.get("previous_purchases")
        previous_purchases = [str(p) for p in previous_purchases if str(p) in catalog.index]
        pdf = catalog.loc[previous_purchases]
        text += f" I have previously purchased {len(previous_purchases)} items from H&M. "
        if len(previous_purchases) > 0:
            if include_historical_item_list:
                text += "My favorite items from before were:"
                # describe each item, grouped by index_group (capped at max_historical_items for simulator persona)
                items_shown = 0
                for index_group, group_df in pdf.groupby("index_group_name"):
                    if items_shown >= max_historical_items:
                        break
                    text += f"\n\n---- {index_group} ----\n\n"
                    for i, (id, row) in enumerate(group_df.iterrows()):
                        if items_shown >= max_historical_items:
                            break
                        text += f"{i + 1}) **{row['prod_name']}** - {row['product_type_name']} (\${row['price']:.2f}) | Color: {row['perceived_colour_master_name']} | Pattern: {row['graphical_appearance_name']} | Section: {row['section_name']}\n"
                        text += f"\t{row['detail_desc']}\n"
                        items_shown += 1

            # add summary statistics
            text += "\n\n"
            text += f"My average purchase price was \${pdf['price'].mean():.2f}. "
            text += f"My most expensive purchase was \${pdf['price'].max():.2f}. "
            text += "I wear a M size and am looking to buy from H&M."
        return text



def make_render_item_fn(
    catalog: pd.DataFrame,
    get_image_fn: Callable[..., Any],
    representation: Representation,
) -> Callable[..., Any]:
    """Create the ``render_item_fn`` closure for H&M items.

    Returns a callable with signature
    ``render_item_fn(id, show_features, width, *, return_html, image_thumb_px)``
    that either renders to Streamlit directly or returns an HTML payload dict.

    Args:
        catalog: Full catalog DataFrame (index = item IDs).
        get_image_fn: Callable that maps an item ID to a base64 image URI or
            PIL Image (as returned by :meth:`HMDataset.get_image_fn`).
        representation: Item representation used to format feature text.

    Returns:
        A render function that:

        * When ``return_html=False`` — renders into the active Streamlit
          context and returns ``None``.
        * When ``return_html=True`` — returns a dict
          ``{"title": str, "body_html": str, "image_url": str}``.
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
                "title": f"Product {id}",
                "body_html": "<div style='opacity:0.85;'>Product not found in catalog.</div>",
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

        title = item.get("prod_name", "Unknown Product")
        image_url = get_image_fn(
            str(item.name), return_image_url=True, max_px=image_thumb_px
        )

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
            else "<div style='font-style:italic;opacity:0.8;margin:6px 0 8px;'>Image not available</div>"
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