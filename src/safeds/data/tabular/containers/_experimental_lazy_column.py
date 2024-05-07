# TODO: polars expressions get optimized first, before being applied. For further performance improvements (if needed),
#  we should mirror this when transitioning from a vectorized row to a cell.

from abc import ABC

from safeds.data.tabular.containers import ExperimentalPolarsCell


class _LazyColumn(ExperimentalPolarsCell, ABC):
    pass
