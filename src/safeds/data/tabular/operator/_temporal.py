from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table


class Temporal:
    """
    A class that contains temporal methods for a column.

    Parameters
    ----------
    table:
        The table to plot.

    Examples
    --------

    """
    def __init__(self, table: Table):
