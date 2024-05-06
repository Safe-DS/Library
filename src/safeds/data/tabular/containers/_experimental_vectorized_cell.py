from ._experimental_polars_cell import ExperimentalPolarsCell
from ._experimental_polars_column import ExperimentalPolarsColumn


class _VectorizedCell(ExperimentalPolarsCell):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, column: ExperimentalPolarsColumn):
        self._column: ExperimentalPolarsColumn = column
