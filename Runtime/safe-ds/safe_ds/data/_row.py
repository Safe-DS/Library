import pandas as pd
from safe_ds.exceptions import ColumnNameError

from ._table_schema import TableSchema


class Row:
    def __init__(self, data: pd.Series, schema: TableSchema):
        self._data: pd.Series = data
        self.schema: TableSchema = schema

    def get_value_by_column_name(self, column_name: str):
        """
        Returns the value of the column of the row.

        Parameters
        ----------
        column_name: str
            The column name

        Returns
        -------
        The value of the column
        """
        if not self.schema.has_column(column_name):
            raise ColumnNameError([column_name])
        return self._data[self.schema._get_column_index_by_name(column_name)]
