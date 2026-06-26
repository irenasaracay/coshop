"""
History DataFrame tool for policies that can manipulate the user's history as a pandas DataFrame.

Unlike history_search (semantic search over history via the vector_search API), this tool exposes
the user's history (spec.historical_df) as a pandas DataFrame named `df` and lets the model run a
single pandas/Python expression against it — filtering, sorting, grouping, aggregation, value
counts, correlations, etc. Each call is stateless: it operates on a fresh copy of the original
history df.

Safety: the expression is compiled in eval mode (single expression only — no imports, assignments,
or multi-statement code), evaluated with a small whitelist of builtins, and expressions containing
dunder attribute access ("__") are rejected. Errors are returned as readable strings so the model
can self-correct rather than crashing the loop.
"""

import ast
import json
from typing import Optional

import pandas as pd
from langchain_core.tools import tool, Tool


# Whitelisted builtins available inside an expression. Anything not listed here
# (e.g. __import__, open, eval, exec) is unavailable.
_SAFE_BUILTIN_NAMES = [
    "len", "min", "max", "sum", "sorted", "abs", "round",
    "list", "dict", "str", "int", "float", "bool", "set",
    "range", "enumerate", "zip",
]


def _build_safe_builtins() -> dict:
    # __builtins__ may be a module or a dict depending on the import context.
    src = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    return {name: src[name] for name in _SAFE_BUILTIN_NAMES if name in src}


_SAFE_BUILTINS = _build_safe_builtins()


def _serialize(result, max_rows: int) -> str:
    """Serialize an expression result (DataFrame / Series / scalar) to a JSON string."""
    if isinstance(result, pd.DataFrame):
        total = len(result)
        out = result.head(max_rows).reset_index()
        # Name the (reset) index column "id" so the model can reference items by id.
        out = out.rename(columns={out.columns[0]: "id"})
        payload = {
            "rows": out.to_dict(orient="records"),
            "n_total": total,
            "truncated": total > max_rows,
        }
        return json.dumps(payload, default=str)

    if isinstance(result, pd.Series):
        s = result.head(max_rows)
        payload = {
            "series": [{"index": str(i), "value": v} for i, v in s.items()],
            "n_total": len(result),
            "truncated": len(result) > max_rows,
        }
        return json.dumps(payload, default=str)

    return json.dumps({"result": result}, default=str)


def get_history_df_tool(
    historical_df: pd.DataFrame,
    max_rows_limit: int = 10,
    max_text_len: Optional[int] = 2000,
) -> Tool:
    """
    Get a tool that lets the policy manipulate the user's history as a pandas DataFrame.

    Args:
        historical_df: The user's history as a DataFrame, indexed by item id. Typically
            spec.historical_df (catalog columns for the user's history plus user_rating_of_5).
        max_rows_limit: Maximum number of rows (for DataFrame/Series results) returned per call.
        max_text_len: If set, the serialized output string is truncated to this many characters
            (mirrors query.QueryFunction.max_text_len). None disables truncation.

    Returns:
        A LangChain Tool for manipulate_history(expr, max_rows).
    """
    description = (
        "Run a single pandas expression over the user's history, exposed as a pandas DataFrame "
        "named `df` and indexed by item id. Only `df` and `pd` are available; the expression must "
        "be a single pandas/Python expression with no imports, assignments, or statements. "
        "Each call is independent: `df` is reset to the user's full, original history every time, "
        "so results do not carry over between calls — express each query against the complete history. "
        "The available columns are not listed here; inspect them yourself (e.g. \"df.columns.tolist()\" "
        "or \"df.head()\") before relying on specific column names. "
        "Examples: \"df[df.user_rating_of_5 >= 4].sort_values('year').head(10)\"; "
        "\"df.groupby('genre')['user_rating_of_5'].mean()\"; \"len(df)\". "
        f"At most {max_rows_limit} rows are returned for table/series results"
        + (
            f", and the returned text is truncated to {max_text_len} characters — "
            "prefer expressions that select only the columns/rows you need."
            if max_text_len is not None
            else "."
        )
    )

    @tool(description=description)
    def manipulate_history(expr: str, max_rows: Optional[int] = None) -> str:
        """Evaluate a single pandas expression against the user's history DataFrame `df`."""
        max_rows = min(max_rows or max_rows_limit, max_rows_limit)

        if "__" in expr:
            return "Error: '__' (dunder access) is not allowed."

        try:
            compiled = compile(ast.parse(expr, mode="eval"), "<history_df>", "eval")
        except SyntaxError as e:
            return f"Error: expression must be a single pandas expression. {e}"

        try:
            result = eval(
                compiled,
                {"__builtins__": _SAFE_BUILTINS, "pd": pd},
                {"df": historical_df.copy()},
            )
        except Exception as e:
            return f"Error during evaluation: {type(e).__name__}: {e}"

        out = _serialize(result, max_rows)
        if max_text_len is not None and len(out) > max_text_len:
            out = out[:max_text_len]
        return out

    return manipulate_history
