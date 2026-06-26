"""Vector-search retrieval backend backed by a remote API server.

Delegates embedding computation and nearest-neighbour lookup to a running
:mod:`coshop.tools.vector_search_server` instance via
:class:`~coshop.tools.vector_search_api_client.VectorSearchAPIClient`.
Inherits the eval-expression prefilter pipeline from
:class:`~coshop.tools.retrieval.retrieval.DFRetrievalFunction`.
"""

from typing import Optional

import pandas as pd

from .retrieval import DFRetrievalFunction
from ...data.representation import Representation


class VectorSearch(DFRetrievalFunction):
    """Catalog retrieval via a remote vector-search API server.

    At query time, the query string is sent to the server along with an
    ``allowed_ids`` whitelist derived from the local catalog (and any
    eval-expression prefilter).  The server returns ranked item IDs and
    similarity scores; the local catalog text is used if the server does not
    return item text.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        representation: Representation,
        vector_search_api_url: str,
        dataset_name: Optional[str] = None,
        version: Optional[str] = None,
        threshold: Optional[float] = None,
        output_ordering: str = "relevance",
        **kwargs,
    ):
        """
        Args:
            catalog: Catalog DataFrame.
            representation: Representation used to convert rows to text.
            vector_search_api_url: Base URL of the vector search API server.
            dataset_name: Dataset name sent to the server so it loads the correct catalog.
                Required; a ``ValueError`` is raised if not provided.
            version: Optional dataset version forwarded to the server.
            threshold: Optional similarity threshold; results below it are dropped.
            output_ordering: "relevance" (default), "random", or "id".
        """
        super().__init__(catalog, representation, output_ordering=output_ordering, **kwargs)

        if not vector_search_api_url:
            raise ValueError(
                "VectorSearch requires vector_search_api_url. "
                "Pass vector_search_api_url=... or set VECTOR_SEARCH_API_URL."
            )

        self._dataset_name = dataset_name
        self._version = version
        if not self._dataset_name:
            raise ValueError(
                "VectorSearch requires dataset_name. "
            )

        self.threshold = threshold

        from ..vector_search_api_client import VectorSearchAPIClient

        self._remote_client = VectorSearchAPIClient(api_url=vector_search_api_url)
        self._df_by_id = self.df.set_index("id", drop=False)

    def _retrieve(
        self,
        q: str,
        m: Optional[int],
        allowed_ids: Optional[list] = None,
    ):
        server_allowed_ids = self._df_by_id.index.astype(str).tolist()
        if allowed_ids is not None:
            allowed_set = {str(i) for i in allowed_ids}
            server_allowed_ids = [i for i in server_allowed_ids if i in allowed_set]

        ids, sims, texts = self._remote_client.vector_search(
            dataset=self._dataset_name,
            version=self._version,
            q=q,
            m=m,
            threshold=self.threshold,
            corrupt_representations=False,
            allowed_ids=server_allowed_ids,
        )

        if not ids:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS + ["similarity"]), "similarity"

        if texts is not None and len(texts) == len(ids):
            ranked = pd.DataFrame(
                {"id": ids, "text": texts, "similarity": sims or [None] * len(ids)}
            )
        else:
            try:
                ranked = self._df_by_id.loc[ids].reset_index(drop=True)
            except Exception:
                df_str = self.df.copy()
                df_str["_id_str"] = df_str["id"].astype(str)
                df_str = df_str.set_index("_id_str", drop=False)
                ranked = df_str.loc[[str(i) for i in ids]].reset_index(drop=True)
                ranked = ranked[self.OUTPUT_COLUMNS]
            ranked = ranked[self.OUTPUT_COLUMNS].copy()
            ranked["similarity"] = sims if sims is not None else [None] * len(ranked)

        return ranked[self.OUTPUT_COLUMNS + ["similarity"]], "similarity"
