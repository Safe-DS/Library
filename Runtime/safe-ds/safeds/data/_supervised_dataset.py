from IPython.core.display_functions import DisplayHandle

from ._column import Column
from ._table import Table


class SupervisedDataset:
    """
    A supervised dataset is split in feature and target vectors from a table for a specific column name.
    It can be used for training models.

    Parameters
    ----------
    table : Table
        The table used to derive the feature and target vectors.
    target_column : str
        Name of the target feature column.
    """

    def __init__(self, table: Table, target_column: str):
        self._y: Column = table.get_column(target_column)
        self._X: Table = table.drop_columns([target_column])

    @property
    def feature_vectors(self) -> Table:
        return self._X

    @property
    def target_values(self) -> Column:
        return self._y

    def __repr__(self) -> str:
        tmp = self._X.add_column(self._y)
        header_info = "Target Column is '" + self._y.name + "'\n"
        return header_info + tmp.__repr__()

    def __str__(self) -> str:
        tmp = self._X.add_column(self._y)
        header_info = "Target Column is '" + self._y.name + "'\n"
        return header_info + tmp.__str__()

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._X.add_column(self._y)
        header_info = "Target Column is '" + self._y.name + "'\n"
        print(header_info)
        return tmp._ipython_display_()
