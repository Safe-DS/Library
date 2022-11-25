from ._column import Column
from ._table import Table


class SupervisedDataset:
    """
    A supervised dataset is split in feature and target vectors from a table for a specific column name.
    It can be used for training models.

    Parameters
    ----------
    table: Table
        The table used to derive the feature and target vectors
    target_column: str
        Name of the target feature column
    """

    def __init__(self, table: Table, target_column: str):
        self._y: Column = table.get_column_by_name(target_column)
        self._X: Table = table.drop_columns([target_column])

    @property
    def feature_vectors(self) -> Table:
        return self._X

    @property
    def target_values(self) -> Column:
        return self._y
