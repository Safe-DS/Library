from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import LazyComputationError

if TYPE_CHECKING:
    import polars as pl


def _safe_collect_lazy_frame(frame: pl.LazyFrame) -> pl.DataFrame:
    """
    Collect a LazyFrame into a DataFrame and raise a custom error if an error occurs.

    Parameters
    ----------
    frame:
        The LazyFrame to collect.

    Returns
    -------
    frame:
        The collected DataFrame.

    Raises
    ------
    LazyComputationError
        If an error occurs during the computation.
    """
    from polars.exceptions import PolarsError

    try:
        return frame.collect()
    except PolarsError as e:
        raise LazyComputationError(str(e)) from None


def _safe_collect_lazy_frame_schema(frame: pl.LazyFrame) -> pl.Schema:
    """
    Collect the schema of a LazyFrame.

    Parameters
    ----------
    frame:
        The LazyFrame to collect the schema of.

    Returns
    -------
    schema:
        The collected schema.

    Raises
    ------
    LazyComputationError
        If an error occurs during the computation.
    """
    from polars.exceptions import PolarsError

    try:
        return frame.collect_schema()
    except PolarsError as e:
        raise LazyComputationError(str(e)) from None
