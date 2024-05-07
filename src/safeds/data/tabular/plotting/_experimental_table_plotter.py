from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import ExperimentalTable


class ExperimentalTablePlotter:
    def __init__(self, table: ExperimentalTable):
        self._table: ExperimentalTable = table

    def box_plots(self) -> Image:
        raise NotImplementedError

    def correlation_heatmap(self) -> Image:
        raise NotImplementedError

    def histograms(self, *, number_of_bins: int = 10) -> Image:
        raise NotImplementedError

    def line_plot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    def scatter_plot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    # TODO: equivalent to Column.plot_compare_columns that takes a list of column names (index_plot)?
