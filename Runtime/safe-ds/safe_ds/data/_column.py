import pandas as pd
from safe_ds.exceptions import IndexOutOfBoundsError


class Column:
    def __init__(self, data: pd.Series):
        self._data: pd.Series = data

    def get_value_by_position(self, index: int):
        """Returns column value at specified index, starting at 0.

        Parameters
        ----------
        index : int
            Index of requested element as integer.

        Returns
        -------
        value
            Value at index in column.

        Raises
        ------
        IndexOutOfBoundsError
            If the given index does not exist in the column.
        """
        if index < 0 or index >= self._data.size:
            raise IndexOutOfBoundsError(index)

        return self._data[index]
