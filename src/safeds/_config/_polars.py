from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def _get_polars_config() -> pl.Config:
    import polars as pl

    return pl.Config(
        float_precision=5,
        tbl_cell_numeric_alignment="RIGHT",
        tbl_formatting="ASCII_FULL_CONDENSED",
        tbl_hide_dataframe_shape=True,
    )
