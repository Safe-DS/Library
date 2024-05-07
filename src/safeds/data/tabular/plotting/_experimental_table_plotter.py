from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import ExperimentalTable


class ExperimentalTablePlotter:
    def __init__(self, table: ExperimentalTable):
        self.table: ExperimentalTable = table

    def boxplots(self) -> Image:
        raise NotImplementedError

    def correlation_heatmap(self) -> Image:
        raise NotImplementedError

    def histograms(self, *, number_of_bins: int = 10) -> Image:
        raise NotImplementedError

    def lineplot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    def scatterplot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError
