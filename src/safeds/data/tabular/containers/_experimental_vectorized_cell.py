from safeds.data.tabular.containers import ExperimentalPolarsCell, ExperimentalPolarsColumn


class _VectorizedCell(ExperimentalPolarsCell):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, column: ExperimentalPolarsColumn):
        self._column: ExperimentalPolarsColumn = column
