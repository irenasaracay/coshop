"""BM25 retrieval backend for catalog search.

Uses the Okapi BM25 algorithm (via the ``rank_bm25`` library) to rank items in
a pandas DataFrame catalog against a natural-language query.  Text is normalized
(unicode NFKD, lowercase, whitespace collapse, optional ``ftfy`` encoding repair)
before indexing and at query time.
"""

import re
import pandas as pd
import unicodedata
from typing import List

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is required. Install it with: pip install rank-bm25")

try:
    import ftfy

    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

from .retrieval import DFRetrievalFunction
from ...data.representation import Representation

class BM25(DFRetrievalFunction):
    """Catalog retrieval using the Okapi BM25 ranking algorithm.

    An inverted-token index is built from the catalog at construction time.
    All query and document text is normalized and tokenized with
    :meth:`_normalize_string` and :meth:`_tokenize` before indexing or scoring.

    Inherits the eval-expression prefilter pipeline from
    :class:`~coshop.tools.retrieval.retrieval.DFRetrievalFunction`.
    """

    def __init__(self, catalog: pd.DataFrame, representation: Representation, threshold: float = None, **kwargs):
        """
        Args:
            catalog: Full item catalog DataFrame.  The index is used as item IDs.
            representation: Converts each catalog row to the text string that is
                indexed and returned to callers.
            threshold: Minimum BM25 score required for a result to be returned.
                Items scoring below this value are dropped.  ``None`` (default)
                disables score filtering.
            **kwargs: Forwarded to :class:`~coshop.tools.retrieval.retrieval.DFRetrievalFunction`.
        """
        super().__init__(catalog, representation, **kwargs)

        self.threshold = threshold

        # Normalize all text in the catalog
        self.df = self.df.copy()
        self.df["normalized_text"] = self.df["text"].apply(self._normalize_string)

        # Tokenize documents for BM25
        self.tokenized_docs = [
            self._tokenize(text) for text in self.df["normalized_text"]
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _normalize_string(self, s: str) -> str:
        """
        Standard normalization function for queries and catalog content.

        This function:
        - Fixes encoding issues (mojibake)
        - Converts to lowercase
        - Normalizes unicode characters (NFKD form)
        - Strips leading/trailing whitespace
        - Collapses multiple whitespace characters into single spaces

        Args:
            s: Input string to normalize

        Returns:
            Normalized string
        """
        if not isinstance(s, str):
            s = str(s)

        # Fix encoding issues (e.g., mojibake)
        if HAS_FTFY:
            s = ftfy.fix_text(s)

        # Normalize unicode characters (decompose and recompose)
        s = unicodedata.normalize("NFKD", s)

        # Convert to lowercase
        s = s.lower()

        # Strip leading/trailing whitespace
        s = s.strip()

        # Collapse multiple whitespace characters into single spaces
        s = re.sub(r"\s+", " ", s)

        return s

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search.

        Args:
            text: Normalized text string

        Returns:
            List of tokens
        """
        # Simple tokenization: split on word boundaries
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _retrieve(self, q: str, m: int, allowed_ids=None):
        """
        Query the catalog using BM25 ranking with scores.

        Args:
            q: Query string
            m: Number of results to return
            allowed_ids: If set, only rows whose id is in this list (compared as str) are ranked.

        Returns:
            Tuple of (DataFrame with results and bm25_score column, "bm25_score")
        """
        # Normalize query
        normalized_query = self._normalize_string(q)

        # Tokenize query
        tokenized_query = self._tokenize(normalized_query)

        # If query is empty after tokenization, return empty results
        if not tokenized_query:
            empty_df = pd.DataFrame(columns=self.OUTPUT_COLUMNS + ["bm25_score"])
            return empty_df, "bm25_score"

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Create results dataframe with scores
        results_df = self.df.copy()
        results_df["bm25_score"] = scores

        if allowed_ids is not None:
            allowed_set = {str(i) for i in allowed_ids}
            results_df = results_df[
                results_df["id"].astype(str).isin(allowed_set)
            ]

        # Sort by score (descending) and take top m
        results_df = results_df.sort_values("bm25_score", ascending=False)
        if self.threshold is not None:
            results_df = results_df[results_df["bm25_score"] >= self.threshold]

        if m is not None:
            results_df = results_df.head(m)
        else:
            results_df = results_df.copy()

        return results_df[self.OUTPUT_COLUMNS + ["bm25_score"]], "bm25_score"
