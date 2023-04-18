from collections.abc import Iterable, Iterator
from hashlib import md5
from typing import Any

import pandas as pd
from IPython.core.display_functions import DisplayHandle, display
from pandas.core.util.hashing import hash_pandas_object

from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing import ColumnType, Schema


class Row:
    """
    A row is a collection of values, where each value is associated with a column name.

    Parameters
    ----------
    data : Iterable
        The data.
    schema : Schema
        The schema of the row.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: Iterable, schema: Schema | None = None):
        self._data: pd.Series = data if isinstance(data, pd.Series) else pd.Series(data)
        self._data = self._data.reset_index(drop=True)

        self._schema: Schema
        if schema is not None:
            self._schema = schema
        else:
            column_names = [f"column_{i}" for i in range(len(self._data))]
            dataframe = self._data.to_frame().T
            dataframe.columns = column_names
            # noinspection PyProtectedMember
            self._schema = Schema._from_dataframe(dataframe)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Row):
            return NotImplemented
        if self is other:
            return True
        return self._schema == other._schema and self._data.equals(other._data)

    def __getitem__(self, column_name: str) -> Any:
        return self.get_value(column_name)

    def __hash__(self) -> int:
        data_hash_string = md5(hash_pandas_object(self._data, index=True).values).hexdigest()
        column_names_frozenset = frozenset(self.get_column_names())

        return hash((data_hash_string, column_names_frozenset))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.get_column_names())

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        tmp = self._data.to_frame().T
        tmp.columns = self.get_column_names()
        return tmp.__repr__()

    def __str__(self) -> str:
        tmp = self._data.to_frame().T
        tmp.columns = self.get_column_names()
        return tmp.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def schema(self) -> Schema:
        """
        Return the schema of the row.

        Returns
        -------
        schema : Schema
            The schema.
        """
        return self._schema

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    def get_value(self, column_name: str) -> Any:
        """
        Return the value of a specified column.

        Parameters
        ----------
        column_name : str
            The column name.

        Returns
        -------
        value :
            The value of the column.
        """
        if not self._schema.has_column(column_name):
            raise UnknownColumnNameError([column_name])
        # noinspection PyProtectedMember
        return self._data[self._schema._get_column_index_by_name(column_name)]

    def has_column(self, column_name: str) -> bool:
        """
        Return whether the row contains a given column.

        Alias for self.schema.hasColumn(column_name: str) -> bool.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        contains : bool
            True, if row contains the column.
        """
        return self._schema.has_column(column_name)

    def get_column_names(self) -> list[str]:
        """
        Return a list of all column names saved in this schema.

        Alias for self.schema.get_column_names() -> list[str].

        Returns
        -------
        column_names : list[str]
            The column names.
        """
        return self._schema.get_column_names()

    def get_type_of_column(self, column_name: str) -> ColumnType:
        """
        Return the type of a specified column.

        Alias for self.schema.get_type_of_column(column_name: str) -> ColumnType.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        type : ColumnType
            The type of the column.

        Raises
        ------
        ColumnNameError
            If the specified target column name does not exist.
        """
        return self._schema.get_type_of_column(column_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of columns in this row.

        Returns
        -------
        count : int
            The number of columns.
        """
        return len(self._data)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._data.to_frame().T
        tmp.columns = self.get_column_names()

        with pd.option_context("display.max_rows", tmp.shape[0], "display.max_columns", tmp.shape[1]):
            return display(tmp)
