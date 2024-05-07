from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.image.containers import Image
    from safeds.data.tabular.containers import ExperimentalColumn


class ExperimentalColumnPlotter:
    def __init__(self, column: ExperimentalColumn):
        self._column: ExperimentalColumn = column

    def box_plot(self) -> Image:
        raise NotImplementedError

    def histogram(self) -> Image:
        raise NotImplementedError

    def lag_plot(self) -> Image:
        raise NotImplementedError
