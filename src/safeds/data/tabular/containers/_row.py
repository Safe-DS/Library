from __future__ import annotations

import copy
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from safeds.data.tabular.typing import ColumnType, Schema
from safeds.exceptions import UnknownColumnNameError

if TYPE_CHECKING:
    from collections.abc import Iterator


class Row(Mapping[str, Any]):
    """
    A row is a collection of named values.

    Parameters
    ----------
    data : Mapping[str, Any] | None
        The data. If None, an empty row is created.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Row
    >>> row = Row({"a": 1, "b": 2})
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Row:
        """
        Create a row from a dictionary that maps column names to column values.

        Parameters
        ----------
        data : dict[str, Any]
            The data.

        Returns
        -------
        row : Row
            The created row.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        """
        return Row(data)

    @staticmethod
    def _from_pandas_dataframe(data: pd.DataFrame, schema: Schema | None = None) -> Row:
        """
        Create a row from a `pandas.DataFrame`.

        Parameters
        ----------
        data : pd.DataFrame
            The data.
        schema : Schema | None
            The schema. If None, the schema is inferred from the data.

        Returns
        -------
        row : Row
            The created row.

        Raises
        ------
        ValueError
            If the dataframe does not contain exactly one row.

        Examples
        --------
        >>> import pandas as pd
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row._from_pandas_dataframe(pd.DataFrame({"a": [1], "b": [2]}))
        """
        if data.shape[0] != 1:
            raise ValueError("The dataframe has to contain exactly one row.")

        data = data.reset_index(drop=True)

        result = object.__new__(Row)
        result._data = data

        if schema is None:
            # noinspection PyProtectedMember
            result._schema = Schema._from_pandas_dataframe(data)
        else:
            result._schema = schema

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        """
        Create a row from a mapping of column names to column values.

        Parameters
        ----------
        data : Mapping[str, Any] | None
            The data. If None, an empty row is created.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        """
        if data is None:
            data = {}

        data = {key: [value] for key, value in data.items()}

        self._data: pd.DataFrame = pd.DataFrame(data)
        # noinspection PyProtectedMember
        self._schema: Schema = Schema._from_pandas_dataframe(self._data)

    def __contains__(self, obj: Any) -> bool:
        """
        Check whether the row contains an object as key.

        Parameters
        ----------
        obj : Any
            The object.

        Returns
        -------
        has_column : bool
            True, if the row contains the object as key, False otherwise.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> "a" in row
        True

        >>> "c" in row
        False
        """
        return isinstance(obj, str) and self.has_column(obj)

    def __eq__(self, other: Any) -> bool:
        """
        Check whether this row is equal to another object.

        Parameters
        ----------
        other : Any
            The other object.

        Returns
        -------
        equal : bool
            True if the other object is an identical row. False if the other object is a different row. NotImplemented
            if the other object is not a row.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row1 = Row({"a": 1, "b": 2})
        >>> row2 = Row({"a": 1, "b": 2})
        >>> row1 == row2
        True

        >>> row3 = Row({"a": 1, "b": 3})
        >>> row1 == row3
        False
        """
        if not isinstance(other, Row):
            return NotImplemented
        if self is other:
            return True
        return self._schema == other._schema and self._data.equals(other._data)

    def __getitem__(self, column_name: str) -> Any:
        """
        Return the value of a specified column.

        Parameters
        ----------
        column_name : str
            The column name.

        Returns
        -------
        value : Any
            The column value.

        Raises
        ------
        UnknownColumnNameError
            If the row does not contain the specified column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row["a"]
        1
        """
        return self.get_value(column_name)

    def __iter__(self) -> Iterator[Any]:
        """
        Create an iterator for the column names of this row.

        Returns
        -------
        iterator : Iterator[Any]
            The iterator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> list(row)
        ['a', 'b']
        """
        return iter(self.column_names)

    def __len__(self) -> int:
        """
        Return the number of columns in this row.

        Returns
        -------
        number_of_columns : int
            The number of columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> len(row)
        2
        """
        return self._data.shape[1]

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of this row.

        Returns
        -------
        representation : str
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1})
        >>> repr(row)
        "Row({'a': 1})"
        """
        return f"Row({self!s})"

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of this row.

        Returns
        -------
        representation : str
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1})
        >>> str(row)
        "{'a': 1}"
        """
        match len(self):
            case 0:
                return "{}"
            case 1:
                return str(self.to_dict())
            case _:
                lines = (f"    {name!r}: {value!r}" for name, value in self.to_dict().items())
                joined = ",\n".join(lines)
                return f"{{\n{joined}\n}}"

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        """
        Return a list of all column names in the row.

        Returns
        -------
        column_names : list[str]
            The column names.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row.column_names
        ['a', 'b']
        """
        return self._schema.column_names

    @property
    def number_of_column(self) -> int:
        """
        Return the number of columns in this row.

        Returns
        -------
        number_of_column : int
            The number of columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row.number_of_column
        2
        """
        return self._data.shape[1]

    @property
    def schema(self) -> Schema:
        """
        Return the schema of the row.

        Returns
        -------
        schema : Schema
            The schema.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> schema = row.schema
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
        value : Any
            The column value.

        Raises
        ------
        UnknownColumnNameError
            If the row does not contain the specified column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row.get_value("a")
        1
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])

        return self._data.loc[0, column_name]

    def has_column(self, column_name: str) -> bool:
        """
        Check whether the row contains a given column.

        Parameters
        ----------
        column_name : str
            The column name.

        Returns
        -------
        has_column : bool
            True, if the row contains the column, False otherwise.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row.has_column("a")
        True

        >>> row.has_column("c")
        False
        """
        return self._schema.has_column(column_name)

    def get_column_type(self, column_name: str) -> ColumnType:
        """
        Return the type of the specified column.

        Parameters
        ----------
        column_name : str
            The column name.

        Returns
        -------
        type : ColumnType
            The type of the column.

        Raises
        ------
        UnknownColumnNameError
            If the row does not contain the specified column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row.get_column_type("a")
        Integer
        """
        return self._schema.get_column_type(column_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def sort_columns(
        self,
        comparator: Callable[[tuple, tuple], int] = lambda col1, col2: (col1[0] > col2[0]) - (col1[0] < col2[0]),
    ) -> Row:
        """
        Sort the columns of a `Row` with the given comparator and return a new `Row`.

        The original row is not modified. The comparator is a function that takes two tuples of (ColumnName,
        Value) `col1` and `col2` and returns an integer:

        * If `col1` should be ordered before `col2`, the function should return a negative number.
        * If `col1` should be ordered after `col2`, the function should return a positive number.
        * If the original order of `col1` and `col2` should be kept, the function should return 0.

        If no comparator is given, the columns will be sorted alphabetically by their name.

        Parameters
        ----------
        comparator : Callable[[tuple, tuple], int]
            The function used to compare two tuples of (ColumnName, Value).

        Returns
        -------
        new_row : Row
            A new row with sorted columns.
        """
        sorted_row_dict = dict(sorted(self.to_dict().items(), key=functools.cmp_to_key(comparator)))
        return Row.from_dict(sorted_row_dict)

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary that maps column names to column values.

        Returns
        -------
        data : dict[str, Any]
            Dictionary representation of the row.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> row.to_dict()
        {'a': 1, 'b': 2}
        """
        return {column_name: self.get_value(column_name) for column_name in self.column_names}

    def to_html(self) -> str:
        """
        Return an HTML representation of the row.

        Returns
        -------
        output : str
            The generated HTML.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row({"a": 1, "b": 2})
        >>> html = row.to_html()
        """
        return self._data.to_html(max_rows=1, max_cols=self._data.shape[1])

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the row.

        Returns
        -------
        output : str
            The generated HTML.
        """
        return self._data.to_html(max_rows=1, max_cols=self._data.shape[1], notebook=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _copy(self) -> Row:
        """
        Return a copy of this row.

        Returns
        -------
        copy : Row
            The copy of this row.
        """
        return copy.deepcopy(self)
