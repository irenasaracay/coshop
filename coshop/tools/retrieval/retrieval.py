"""Core retrieval abstractions and the LLM-based eval-expression prefilter.

Main exports:

- :class:`RetrievalFunction` — abstract base; ``__call__`` returns a two-column
  DataFrame (``id``, ``text``).
- :class:`DFRetrievalFunction` — concrete base backed by a pandas DataFrame catalog;
  adds the LLM-generated eval-expression prefilter pipeline.
- :func:`get_retrieval_fn` — factory that instantiates ``BM25`` or ``VectorSearch``
  by name.
"""

import os
import re

import pandas as pd
from typing import Literal, Optional, List, Dict, Tuple
import json

from ...data.representation import Representation

def get_retrieval_fn(retrieval_name: str, **kwargs):
    """Instantiate a retrieval backend by name.

    Args:
        retrieval_name: Backend to use.  One of ``"BM25"`` or ``"VectorSearch"``.
        **kwargs: Keyword arguments forwarded to the selected class constructor.

    Returns:
        An instance of :class:`BM25` or :class:`VectorSearch`.

    Raises:
        AttributeError: If ``retrieval_name`` is not a recognised backend.
    """
    # Allow passing an already-instantiated retrieval function through unchanged
    # (e.g. get_final_predictions reuses the elicitation retrieval_fn directly).
    if isinstance(retrieval_name, RetrievalFunction):
        return retrieval_name

    normalized = (retrieval_name or "").strip().lower()
    if normalized == "bm25":
        from .bm25 import BM25
        return BM25(**kwargs)
    elif normalized == "vectorsearch":
        from .vector_search import VectorSearch
        return VectorSearch(**kwargs)
    raise AttributeError(f"module {__name__!r} has no attribute {retrieval_name!r}")


def _normalize_eval_expression(expr: str) -> str:
    """
    Normalize an eval expression so pandas.eval() does not misinterpret it.
    In pandas eval, backticks denote column names. If the LM wrapped string
    literals in backticks (e.g. `'Comedy'`), or the whole expression in
    backticks, strip those so the expression evaluates correctly.
    Also fix common boolean comparisons like `has_ribbing == 'True'` by
    converting them to `has_ribbing == True` so they work with boolean columns.
    """
    # Remove backticks that wrap a quoted string literal (e.g. `'Comedy'` -> 'Comedy').
    # This avoids "name 'BACKTICK_QUOTED_STRING_...' is not defined".
    expr = re.sub(r"`(\'[^\']*\'|\"[^\"]*\")`", r"\1", expr)
    # If the entire expression is wrapped in a single pair of backticks, remove them.
    if expr.startswith("`") and expr.endswith("`") and expr.count("`") == 2:
        expr = expr[1:-1]

    # Normalize boolean comparisons like `has_ribbing == 'True'` or `flag != "False"`
    # into `has_ribbing == True` / `flag != False` so they work with boolean columns.
    def _replace_bool_match(match: re.Match) -> str:
        col = match.group(1)
        op = match.group(2)
        value = match.group(4)
        # value is "True" or "False"; map to Python boolean literal
        bool_literal = "True" if value == "True" else "False"
        return f"{col} {op} {bool_literal}"

    expr = re.sub(
        r"(\b\w+\b)\s*([=!]=)\s*(['\"])(True|False)\3",
        _replace_bool_match,
        expr,
    )

    return expr

class RetrievalFunction:
    """Abstract base class for catalog retrieval functions.

    Subclasses implement :meth:`_retrieve`; callers use :meth:`__call__`, which
    enforces the output contract (columns ``["id", "text"]``, at most ``m`` rows,
    optional text truncation, and result ordering).

    Attributes:
        OUTPUT_COLUMNS: Always ``["id", "text"]``.
        output_ordering: How results are ordered before returning.  One of
            ``"random"``, ``"id"``, ``"relevance"`` (default — subclass ordering
            is preserved), or ``"popularity"`` (requires a ``popularity_df``;
            only supported in :class:`DFRetrievalFunction`).
        max_text_len: If set, each ``text`` value is truncated to this many
            characters before returning.
    """

    OUTPUT_COLUMNS = ["id", "text"]

    def __init__(
        self,
        output_ordering: Literal[
            "random", "id", "relevance", "popularity"
        ] = "relevance",
        max_text_len: Optional[int] = 2000,
    ):
        """
        Args:
            output_ordering: Result ordering applied after retrieval.  ``"random"``
                shuffles results; ``"id"`` sorts by item ID; ``"relevance"`` preserves
                the order returned by :meth:`_retrieve`; ``"popularity"`` sorts by
                the ``popularity`` column in ``popularity_df`` (only available in
                :class:`DFRetrievalFunction`).
            max_text_len: Truncate each ``text`` value to this many characters.
                ``None`` disables truncation.  Defaults to ``2000``.
        """
        self.output_ordering = output_ordering
        self.max_text_len = max_text_len

    def __call__(
        self, q: Optional[str], m: int, allowed_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Run retrieval and return the top-``m`` results.

        Args:
            q: Natural-language query string.  ``None`` or empty string behavior
                is subclass-specific (typically returns the full catalog).
            m: Maximum number of rows to return.
            allowed_ids: Optional whitelist of item IDs to restrict results to.

        Returns:
            A DataFrame with exactly the columns ``["id", "text"]`` and at most
            ``m`` rows, ordered according to :attr:`output_ordering`.
        """
        results_df, _ = self._retrieve(q, m, allowed_ids=allowed_ids)
        # Drop score column if present, keep only OUTPUT_COLUMNS
        output = (
            results_df[self.OUTPUT_COLUMNS]
            if len(results_df) > 0
            else pd.DataFrame(columns=self.OUTPUT_COLUMNS)
        )

        assert m is None or len(output) <= m, (
            "Number of results must be less than or equal to m"
        )
        assert sorted(output.columns.tolist()) == sorted(self.OUTPUT_COLUMNS), (
            "Output columns must be the same as the output columns"
        )

        if self.output_ordering == "random":
            output = output.sample(frac=1)
        elif self.output_ordering == "id":
            output = output.sort_values("id")
        elif self.output_ordering == "relevance":
            pass  # assume _retrieve returns items sorted by relevance
        elif self.output_ordering == "popularity":
            # Popularity ordering requires popularity_df, which is only available in DFRetrievalFunction
            # Subclasses that support popularity should override __call__ to handle it
            raise ValueError(
                "Popularity ordering is not supported in this RetrievalFunction subclass. "
                "Use DFRetrievalFunction with popularity_df parameter."
            )
        else:
            raise ValueError(f"Invalid output ordering: {self.output_ordering}")

        # Optionally truncate text column
        if self.max_text_len is not None and "text" in output.columns:
            output["text"] = output["text"].astype(str).str.slice(0, self.max_text_len)
        return output


class DFRetrievalFunction(RetrievalFunction):
    """DataFrame-backed retrieval with LLM-generated eval-expression prefiltering.

    This class wraps a pandas DataFrame catalog and adds an optional
    *eval-expression prefilter* pipeline that runs before the semantic search:

    1. An LLM is prompted with the query and a list of filterable column names
       plus example values.  It returns a JSON list of simple ``pandas.DataFrame.eval``
       boolean expressions (one simple clause per item, e.g.
       ``"genres.str.contains('Comedy')"``.
    2. Expressions are applied **greedily** in order: each expression is applied
       only if it would leave at least one row; otherwise it is skipped.
    3. The resulting filtered ``allowed_ids`` are passed to :meth:`_retrieve`.
    4. If the filtered results are fewer than ``m``, they are supplemented with
       unrestricted retrieval results (deduplication preserves filtered results first).

    This two-stage approach can dramatically narrow the search space before BM25
    or vector similarity scoring, improving precision for highly specific queries.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        representation: Representation,
        retrieval_representation: Optional[Representation] = None,
        output_ordering: Literal[
            "random", "id", "relevance", "popularity"
        ] = "relevance",
        max_text_len: Optional[int] = None,
        popularity_df: Optional[pd.DataFrame] = None,
        prefilter: bool = True,
        num_expression_ordering_trials: int = 5,
        eval_expression_model_name: str = "gpt-5-nano",
        temperature: float = 0.0,
        eval_expression_columns: Optional[List[str]] = None,
        verbose: bool = False,
        **eval_expression_model_kwargs,
    ):
        """
        Args:
            catalog: Full item catalog.  The DataFrame index is used as item IDs.
            representation: :class:`~coshop.data.representation.Representation` used
                to render each catalog row into the ``text`` field returned to callers.
            retrieval_representation: Representation used *internally* during retrieval
                (e.g. BM25 index construction).  If ``None``, ``representation`` is
                used for both retrieval and output.
            output_ordering: Result ordering.  See :class:`RetrievalFunction`.
                ``"popularity"`` requires ``popularity_df``.
            max_text_len: Per-item text truncation length.  ``None`` disables
                truncation.
            popularity_df: DataFrame indexed by item ID with a ``"popularity"``
                column.  Required when ``output_ordering="popularity"``.
            prefilter: When ``True`` (default), run the LLM eval-expression pipeline
                before retrieval.  When ``False``, pass the query directly to
                :meth:`_retrieve` with no prefiltering.
            num_expression_ordering_trials: Stored for backwards compatibility but
                not currently used; eval expressions are applied greedily in the
                order the LLM generated them (see
                :meth:`_apply_eval_expressions_greedy`).  Defaults to ``5``.
            eval_expression_model_name: LLM model name used to generate eval
                expressions.  Defaults to ``"gpt-5-nano"``.
            temperature: Sampling temperature for the eval-expression LLM.
                Defaults to ``0.0``.
            eval_expression_columns: Subset of catalog columns the LLM may filter
                on.  If ``None``, no eval-expression columns are available and
                prefiltering degrades to unrestricted retrieval.
            verbose: If ``True``, print eval-expression decisions to stdout.
            **eval_expression_model_kwargs: Extra kwargs forwarded to the OpenAI
                client (e.g. ``base_url``, ``api_key``, ``model_provider``,
                ``vllm_api_url``).  ``model_provider="vllm"`` together with
                ``vllm_api_url`` routes requests to a local vLLM server.
        """
        super().__init__(output_ordering=output_ordering, max_text_len=max_text_len)

        if output_ordering == "popularity" and popularity_df is None:
            raise ValueError(
                "popularity_df is required when output_ordering is 'popularity'"
            )

        self.popularity_df = popularity_df

        # Eval expression logic
        self.prefilter = prefilter
        self.num_expression_ordering_trials = max(
            1, int(num_expression_ordering_trials)
        )
        self.eval_expression_columns = eval_expression_columns or []
        self.eval_expression_model_name = eval_expression_model_name
        self.eval_expression_model_kwargs = dict(eval_expression_model_kwargs)
        provider = self.eval_expression_model_kwargs.pop("model_provider", None)
        vllm_url = (
            self.eval_expression_model_kwargs.pop("vllm_api_url", None)
            or os.environ.get("POLICY_VLLM_API_URL", "")
            or os.environ.get("VLLM_API_URL", "")
        )
        if provider == "vllm" and vllm_url:
            vllm_url = str(vllm_url).rstrip("/")
            self.eval_expression_model_kwargs["base_url"] = (
                vllm_url if vllm_url.endswith("/v1") else vllm_url + "/v1"
            )

        # Build search index
        self.catalog = catalog
        self.output_df = pd.DataFrame(
            {
                "id": catalog.index,
                "text": catalog.apply(representation.row_to_str, axis=1),
            }
        ).reset_index(drop=True)
        self.df = pd.DataFrame(
            {
                "id": catalog.index,
                "text": catalog.apply(
                    retrieval_representation.row_to_str
                    if retrieval_representation
                    else representation.row_to_str,
                    axis=1,
                ),
            }
        ).reset_index(drop=True)
        assert len(self.df) > 0, "Catalog must not be empty"

    def _get_lm_client(self):
        """Create OpenAI client with eval_expression_model_kwargs (e.g. base_url for custom endpoints)."""
        from openai import OpenAI

        kwargs = dict(self.eval_expression_model_kwargs)
        # OpenAI client uses base_url; api_base is used by LiteLLM
        if "api_base" in kwargs and "base_url" not in kwargs:
            kwargs["base_url"] = kwargs.pop("api_base")
        client_params = {
            k: v
            for k, v in kwargs.items()
            if k in ("base_url", "api_key", "timeout", "max_retries", "organization")
        }
        return OpenAI(**client_params)

    def _retrieve(
        self,
        q: Optional[str],
        m: Optional[int],
        allowed_ids: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Retrieval method that subclasses should override. Should return results with scores if available.

        Args:
            q: Query string
            m: Maximum number of results

        Returns:
            Tuple of (DataFrame with results and optional score column, name of score column or None)
            The DataFrame should contain at least OUTPUT_COLUMNS, and optionally a score column.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(
        self,
        q: Optional[str],
        m: Optional[int],
        verbose: bool = False,
        allowed_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Override to handle None or empty queries by returning the whole catalog.

        Args:
            q: query string (if None or empty, returns whole catalog)
            m: number of results to return

        Returns:
            pd.DataFrame containing the results. There will be two columns: "id" and "text".
        """
        # If query is None or empty, return the whole catalog
        if q is None or (isinstance(q, str) and q.strip() == ""):
            output = self.output_df.copy()[self.OUTPUT_COLUMNS]
        elif re.match(r"ID (\w+)$", q):
            # Special query: "ID <id>" returns the rows with the given id
            id = re.match(r"ID (\w+)$", q).group(1)
            output = self.output_df[self.output_df["id"] == id].copy()[
                self.OUTPUT_COLUMNS
            ]
        elif self.prefilter:
            expressions = self._get_eval_expressions(q)
            if expressions:
                filtered_catalog, applied = self._apply_eval_expressions_greedy(
                    self.catalog, expressions, verbose=verbose
                )
                if verbose and applied:
                    print(
                        f"Eval expressions applied ({len(applied)} of {len(expressions)}): {applied}; filtered catalog length: {len(filtered_catalog)}",
                    )

                # Start with ids that survived eval-expression filtering
                filtered_ids = filtered_catalog.index.astype(str).tolist()

                # Respect any externally provided allowed_ids by intersecting
                if allowed_ids is not None:
                    allowed_ids_set = set(str(i) for i in allowed_ids)
                    filtered_ids = [i for i in filtered_ids if i in allowed_ids_set]

                if verbose:
                    print(f"Filtered catalog length: {len(filtered_ids)}")

                # First, query restricted to the filtered ids
                filtered_results, score_col = self._retrieve(
                    q, m, allowed_ids=filtered_ids
                )

                # If we have a target m and filtered results are too few,
                # supplement with additional results from the baseline retrieval
                if m is not None and len(filtered_results) < m:
                    baseline_results, _ = self._retrieve(q, m, allowed_ids=allowed_ids)
                    if len(baseline_results) > 0:
                        # Drop any ids we already have from filtered_results
                        existing_ids = set(filtered_results["id"].astype(str))
                        supplemental = baseline_results[
                            ~baseline_results["id"].astype(str).isin(existing_ids)
                        ]
                        if len(supplemental) > 0:
                            filtered_results = pd.concat(
                                [filtered_results, supplemental], ignore_index=True
                            )
                output = filtered_results
            else:
                # No expressions generated; fall back to unrestricted query.
                output, _ = self._retrieve(q, m, allowed_ids=allowed_ids)
        else:
            # No prefiltering; unrestricted retrieval
            output, _ = self._retrieve(q, m, allowed_ids=allowed_ids)

        # Enforce m just in case
        if m is not None:
            output = output.head(m)

        # Enforce our text representations by re-rendering returned ids via output_df,
        # while preserving the order in which _retrieve returned them (relevance order).
        ordered_ids = output["id"].astype(str).drop_duplicates().tolist()
        rendered = self.output_df.copy()
        rendered = rendered[rendered["id"].astype(str).isin(set(ordered_ids))]
        sort_key = rendered["id"].astype(str).map(
            {id_str: pos for pos, id_str in enumerate(ordered_ids)}
        )
        output = (
            rendered.assign(_order=sort_key.values)
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        assert m is None or len(output) <= m, (
            "Number of results must be less than or equal to m"
        )
        assert sorted(output.columns.tolist()) == sorted(self.OUTPUT_COLUMNS), (
            "Output columns must be the same as the output columns"
        )

        if self.output_ordering == "random":
            output = output.sample(frac=1)
        elif self.output_ordering == "id":
            output = output.sort_values("id")
        elif self.output_ordering == "relevance":
            pass  # assume _retrieve returns items sorted by relevance
        elif self.output_ordering == "popularity":
            if self.popularity_df is None:
                raise ValueError("popularity_df is required for popularity ordering")
            output = output.merge(
                self.popularity_df[["popularity"]],
                left_on="id",
                right_index=True,
                how="left",
            )
            # Sort by popularity (descending), then by id for tie-breaking
            output = output.sort_values(
                ["popularity", "id"], ascending=[False, True], na_position="last"
            )
            # Drop the popularity column to keep only OUTPUT_COLUMNS
            output = output[self.OUTPUT_COLUMNS]
        else:
            raise ValueError(f"Invalid output ordering: {self.output_ordering}")

        # Safety filter: only return ids that exist in catalog (critical for subset_catalog)
        catalog_idx_str = set(self.catalog.index.astype(str))
        output = output[output["id"].astype(str).isin(catalog_idx_str)]

        # Optionally truncate text column
        if self.max_text_len is not None and "text" in output.columns:
            output["text"] = output["text"].astype(str).str.slice(0, self.max_text_len)

        return output.reset_index(drop=True)

    def _get_eval_expressions(
        self, query: str, max_examples_per_column: int = 5
    ) -> List[str]:
        """
        Ask an LM to produce a list of simple pandas DataFrame.eval() boolean expressions.
        Each expression will be applied in order and we take the intersection of matching rows;
        expressions that would reduce results to zero are skipped (greedy inclusion).

        Returns:
            List of expression strings suitable for df.eval(expr). Empty list if generation fails.
        """
        # Restrict to configured eval-expression columns when provided; otherwise
        # fall back to all non-text columns.
        column_names = [
            c for c in self.eval_expression_columns if c in self.catalog.columns
        ]
        # Build up to max_examples_per_column example values per column to ground the model.
        examples_by_column: Dict[str, List[str]] = {}
        column_types: Dict[str, str] = {}
        for col in column_names:
            if col not in self.catalog.columns:
                continue
            series = self.catalog[col].dropna()
            # Note dtype for the LM (numeric vs string affects expression syntax)
            dtype = self.catalog[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                column_types[col] = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                column_types[col] = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                column_types[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                column_types[col] = "datetime"
            else:
                column_types[col] = "string"
            if len(series) == 0:
                continue
            unique_vals = series.astype(str).unique().tolist()
            examples_by_column[col] = unique_vals[:max_examples_per_column]
        client = self._get_lm_client()

        prompt = [
            {
                "role": "system",
                "content": (
                    "You produce a list of SIMPLE pandas DataFrame.eval() expressions. "
                    "Each expression must evaluate to a boolean Series. We will apply them one by one and take the intersection of matching rows; if an expression would leave zero rows, we skip it. "
                    "So each expression should be ONE simple clause (e.g. one column comparison). Do NOT combine multiple conditions with & or | in a single expression—use separate list items instead. "
                    "Use only column names provided. "
                    "For string literals always use single or double quotes (e.g. 'Comedy', \"Drama\"). Be case sensitive. "
                    "Use backticks ONLY for column names that contain spaces or are not valid Python identifiers (e.g. `Area (cm^2)`), never for string values. "
                    "When the user lists multiple values for the same attribute, use one expression per value (e.g. one item \"genres.str.contains('Comedy')\", another \"genres.str.contains('Drama')\"). "
                    "Return a JSON object with a key 'expressions' containing a list of expression strings. No explanation, no code block. "
                    "Try not to read into semantic meaning. Filter based on obvious intent. Avoid filtering on titles or descriptions. "
                    "Fewer, simpler expressions are better than one long combined expression. "
                    "DO NOT add any filters that are not cleary implied by the string, in particular do NOT add an is_ebook filter. For example, 'book popular' does not mean you should threshold on avg_rating or num_ratings because that is reading too much into the string, and 'recent' does not mean you can safely filter on a year because that is reading too much into the string."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Query: {query}\n\n"
                            f"Filterable columns (name -> type): {column_types}\n\n"
                            f"Example values by column: {examples_by_column}\n\n"
                            'Give a list of simple pandas eval expressions (booleans) to filter rows. Return JSON: {"expressions": ["expr1", "expr2", ...]}'
                        ),
                    },
                ],
            },
        ]

        try:
            # eval_expression_model_kwargs is a catch-all for kwargs passed to
            # the retrieval backend constructor, so it can contain OpenAI client
            # params (handled in _get_lm_client) as well as retrieval/tool infra
            # kwargs (e.g. dataset_name, version) that are NOT valid arguments to
            # responses.create(). Drop both groups before issuing the request.
            _non_request_keys = {
                "base_url",
                "api_base",
                "api_key",
                "timeout",
                "max_retries",
                "organization",
                "dataset_name",
                "version",
                "vector_search_api_url",
                "filterable_features",
                "max_items_limit",
                "execution_global_max",
                "execution_max_queries",
            }
            request_kwargs = {
                k: v
                for k, v in self.eval_expression_model_kwargs.items()
                if k not in _non_request_keys
            }
            create_kwargs = {
                "model": self.eval_expression_model_name,
                "input": prompt,
                **request_kwargs,
            }
            response = client.responses.create(**create_kwargs)
            response_text = (response.output_text or "").strip()
            # Remove markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                response_text = "\n".join(lines)
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "expressions" in parsed:
                exprs = parsed["expressions"]
                if isinstance(exprs, list):
                    return [
                        _normalize_eval_expression(str(e).strip())
                        for e in exprs
                        if e and str(e).strip()
                    ]
            return []
        except Exception as e:
            print(f"Warning: Failed to generate eval expressions: {e}")
            return []

    def _apply_eval_expressions_greedy(
        self,
        merged: pd.DataFrame,
        expressions: List[str],
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply a list of eval expressions greedily: for each expression, apply it only if
        the resulting filtered dataframe would have at least one row. Returns the
        intersection of rows matching all applied expressions and the list of applied expressions.
        """
        working = merged
        applied: List[str] = []

        for expr in expressions:
            if verbose:
                print("Trying expression:", expr)
            if not expr:
                continue
            try:
                mask = working.eval(expr)
                if not hasattr(mask, "dtype") or mask.dtype != bool:
                    continue
                filtered = working.loc[mask]
                if len(filtered) == 0:
                    if verbose:
                        print("> Skipping expression:", expr)
                    continue
                if verbose:
                    print("> Keeping expression:", expr, "new length:", len(filtered))
                working = filtered
                applied.append(expr)
            except Exception:
                continue
        if applied and verbose:
            print(f"Eval expressions applied ({len(applied)}): {applied}")
        return working, applied
