"""Representation classes for converting catalog rows to text formats.

A :class:`Representation` is a stateless formatter that turns a pandas
:class:`~pandas.Series` catalog row into a plain-text or markdown string
suitable for passing to an LLM.  Representations are pure formatters: they
accept an optional ``columns`` argument to restrict which columns are shown,
but do not apply any corruption or noise (that logic lives in the retrieval
layer).

Main exports:
    Representation: Abstract base class.
    ParagraphRepresentation: Markdown-style ``**col (desc)**: value`` format.
    representation_text_to_html: Convert paragraph representation text to HTML.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from ..utils.misc import check_na
import re


class Representation:
    """Abstract base class for catalog item representations.

    Subclasses must implement :meth:`row_to_str` and :meth:`str_to_id`.

    Attributes:
        restricted_columns: When set, only these column names are ever included
            in the output of :meth:`row_to_str`.  Callers may further restrict
            via the ``columns`` argument, which is intersected with this list.
        show_na: Whether to include columns whose value is ``NaN`` / empty in
            the output.
    """

    def __init__(
        self,
        restricted_columns: Optional[List[str]] = None,
        show_na: bool = False,
        **kwargs,
    ):
        """Initialise a Representation.

        Args:
            restricted_columns: Optional allowlist of column names.  When
                provided, columns not in this list are always omitted from
                :meth:`row_to_str` output regardless of the ``columns``
                argument.  ``None`` means no restriction.
            show_na: When ``True``, columns with ``NaN`` or empty values are
                included in the output (with their missing value displayed).
                Defaults to ``True``.
            **kwargs: Ignored; accepted for subclass compatibility.
        """
        self.restricted_columns = restricted_columns
        self.show_na = show_na

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.__repr__()

    def row_to_str(
        self,
        row: pd.Series,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Convert a catalog row to a text string.

        The output includes only columns in the intersection of ``columns``
        and :attr:`restricted_columns` (when both are set).

        Args:
            row: A :class:`~pandas.Series` representing one catalog item.
                The series index should be column names; ``row.name`` should
                be the item ID.
            columns: Optional secondary allowlist of column names.  Only
                columns present in both this list and
                :attr:`restricted_columns` are included.  ``None`` means
                use :attr:`restricted_columns` only.

        Returns:
            A formatted string representation of the row.
        """
        raise NotImplementedError

    def str_to_id(self, text: str) -> str:
        """Extract the item ID from a string produced by :meth:`row_to_str`.

        Args:
            text: A string as returned by :meth:`row_to_str`.

        Returns:
            The item ID string, or ``None`` if the ID cannot be parsed.
        """
        raise NotImplementedError

    def __eq__(self, row1: pd.Series, text2: str) -> bool:
        """Check whether a catalog row matches a text representation.

        Compares by item ID: parses ``text2`` with :meth:`str_to_id` and
        checks equality with ``str(row1.name)``.

        Args:
            row1: A :class:`~pandas.Series` representing one catalog item.
            text2: A text string as produced by :meth:`row_to_str`.

        Returns:
            ``True`` if the item IDs match.
        """
        id2 = self.str_to_id(text2)
        return str(row1.name) == id2


def _filter_row_to_columns(
    row: pd.Series,
    restricted_cols_1: Optional[List[str]] = None,
    restricted_cols_2: Optional[List[str]] = None,
) -> pd.Series:
    """Filter row to the intersection of restricted_cols_1 and restricted_cols_2 columns."""
    if restricted_cols_1 is not None:
        restricted_cols_1 = set(restricted_cols_1)
    else:
        restricted_cols_1 = set(row.index).union({"id"})
    if restricted_cols_2 is not None:
        restricted_cols_2 = set(restricted_cols_2)
    else:
        restricted_cols_2 = set(row.index).union({"id"})
    available_cols = restricted_cols_1.intersection(restricted_cols_2)
    available_cols = list(sorted(available_cols))

    if "id" in available_cols:
        row = row.copy()
        row["id"] = str(row.name)
        # move to front
        available_cols.remove("id")
        available_cols.insert(0, "id")

    return row[available_cols]


class ParagraphRepresentation(Representation):
    """Markdown-style item representation.

    Formats each column as one line::

        **col (description)**: value

    where *description* is taken from ``feature_descriptions`` (with any
    parenthetical suffix stripped).  The ``id`` column is always prepended.

    Example output::

        **id**: 123
        **color (Colour of the garment)**: red
        **size (Clothing size)**: large
    """

    def __init__(
        self, feature_descriptions: dict = None, feature_order: list = None, **kwargs
    ):
        """Initialise a ParagraphRepresentation.

        Args:
            feature_descriptions: Mapping from column name to human-readable
                feature description (e.g. ``{"color": "Colour of the garment
                (e.g. red, blue)"}``).  Parenthetical suffixes are stripped
                automatically.  Corresponds to the ``true_features`` /
                ``feature_descriptions`` argument on :class:`~coshop.data.dataset.Dataset`.
            feature_order: Preferred column ordering in the output.  Columns
                not listed here appear after those that are.  Defaults to the
                key order of ``feature_descriptions``.
            **kwargs: Forwarded to :class:`Representation`.
        """
        super().__init__(**kwargs)
        self.feature_descriptions = {
            k: v.split("(")[0]
            for k, v in (
                feature_descriptions if feature_descriptions is not None else {}
            ).items()
        }
        self.feature_order = (
            feature_order
            if feature_order is not None
            else list(self.feature_descriptions.keys())
        )
        self.feature_descriptions["id"] = "id"
        self.inverse_feature_descriptions = {
            v: k for k, v in self.feature_descriptions.items()
        }
        self.inverse_feature_descriptions["id"] = "id"

    def row_to_str(
        self,
        row: pd.Series,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Format a catalog row as a markdown paragraph.

        Args:
            row: A :class:`~pandas.Series` for one catalog item.
            columns: Optional secondary column allowlist intersected with
                :attr:`~Representation.restricted_columns`.

        Returns:
            A newline-joined string with one ``**col (desc)**: value`` line
            per included column.
        """
        new_row = _filter_row_to_columns(row, columns, self.restricted_columns)

        lines = []
        for col in new_row.index:
            value = new_row.get(col, None)
            if (not check_na(value) and value != "") or (
                self.show_na and check_na(value)
            ):
                lines.append(
                    f"**{col} ({self.feature_descriptions.get(col, col).strip()})**: {value}"
                )
        return "\n".join(lines)

    def str_to_id(self, text: str) -> str:
        """Extract the item ID from a paragraph string.

        Args:
            text: A string as produced by :meth:`row_to_str`.

        Returns:
            The numeric ID string (as matched by ``**id**: <digits>``), or
            ``None`` if the pattern is not found.
        """
        match = re.search(r"\*\*id\*\*: (\d+)", text)
        if match:
            return match.group(1)
        return None

    def dump_state(self) -> Dict[str, Any]:
        """Serialize the representation state."""
        return {
            "type": "paragraph",
            "kwargs": {
                "feature_descriptions": self.feature_descriptions,
            },
        }


def representation_text_to_html(text: str) -> str:
    """Convert a paragraph representation string to HTML.

    Handles the following patterns produced by :class:`ParagraphRepresentation`
    and related formatting:

    * ``# Heading`` → ``<div>`` with large bold text
    * ``**Label:** value`` → ``<div><b>Label:</b> value</div>``
    * ``-------- Section --------`` → section header ``<div>``
    * Blank lines → ``<br/>``
    * Plain text → plain ``<div>``

    All content is HTML-escaped to prevent injection.

    Args:
        text: A string as produced by :meth:`ParagraphRepresentation.row_to_str`
            or similar formatting helpers.

    Returns:
        An HTML string suitable for embedding in a webpage or Streamlit
        ``components.html`` call.
    """
    import html as _html

    def esc(x: str) -> str:
        return _html.escape(x or "")

    lines = text.strip().split("\n")
    html_parts: List[str] = []
    for line in lines:
        if not line.strip():
            html_parts.append("<br/>")
            continue
        if line.strip().startswith("--------"):
            html_parts.append(
                f"<div style='margin-top:12px;font-weight:600;opacity:0.9;'>{esc(line.strip().replace('-', ' ').strip())}</div>"
            )
        elif line.startswith("# "):
            html_parts.append(
                f"<div style='font-weight:600;font-size:1.1em;margin-bottom:4px;'>{esc(line[2:].strip())}</div>"
            )
        elif "**" in line:
            # **Label:** value
            match = re.match(r"\*\*(.+?)\*\*:\s*(.*)", line.strip())
            if match:
                label, value = match.groups()
                html_parts.append(f"<div><b>{esc(label)}:</b> {esc(value)}</div>")
            else:
                html_parts.append(f"<div>{esc(line)}</div>")
        else:
            html_parts.append(f"<div style='margin-bottom:4px;'>{esc(line)}</div>")
    return "".join(html_parts)
