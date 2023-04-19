from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing import ColumnType, Schema

if TYPE_CHECKING:
    from collections.abc import Iterator


class Row:
    """
    A row is a collection of values, where each value is associated with a column name.

    To create a row manually, use the static method [from_dict][safeds.data.tabular.containers._row.Row.from_dict].

    Examples
    --------
    >>> from safeds.data.tabular.containers import Row
    >>> row = Row.from_dict({"a": 1, "b": 2})
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
            The generated row.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        """
        return Row(pl.DataFrame(data))

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: pl.DataFrame, schema: Schema | None = None):
        """
        Initialize a row from a `polars.DataFrame`.

        **Do not use this method directly.** It is not part of the public interface and may change in the future
        without a major version bump. Use the static method
        [from_dict][safeds.data.tabular.containers._row.Row.from_dict] instead.

        Parameters
        ----------
        data : polars.DataFrame
            The data.
        schema : Schema | None
            The schema. If None, the schema is inferred from the data.
        """
        self._data: pl.DataFrame = data

        self._schema: Schema
        if schema is not None:
            self._schema = schema
        else:
            # noinspection PyProtectedMember
            self._schema = Schema._from_polars_dataframe(self._data)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Row):
            return NotImplemented
        if self is other:
            return True
        return self._schema == other._schema and self._data.frame_equal(other._data)

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
            The value of the column.

        Raises
        ------
        UnknownColumnNameError
            If the row does not contain the specified column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row["a"]
        1
        """

        return self.get_value(column_name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.get_column_names())

    def __len__(self) -> int:
        """
        Return the number of columns in this row.

        Returns
        -------
        count : int
            The number of columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> len(row)
        2
        """
        return self._data.shape[1]

    def __repr__(self) -> str:
        return f"Row({str(self)})"

    def __str__(self) -> str:
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
        >>> row = Row.from_dict({"a": 1, "b": 2})
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
            The value of the column.

        Raises
        ------
        UnknownColumnNameError
            If the row does not contain the specified column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row.get_value("a")
        1
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])

        return self._data[0, column_name]

    def has_column(self, column_name: str) -> bool:
        """
        Return whether the row contains a given column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        has_column : bool
            True, if row contains the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row.has_column("a")
        True

        >>> row.has_column("c")
        False
        """
        return self._schema.has_column(column_name)

    def get_column_names(self) -> list[str]:
        """
        Return a list of all column names in the row.

        Returns
        -------
        column_names : list[str]
            The column names.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row.get_column_names()
        ['a', 'b']
        """
        return self._schema.get_column_names()

    def get_type_of_column(self, column_name: str) -> ColumnType:
        """
        Return the type of the specified column.

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
        UnknownColumnNameError
            If the row does not contain the specified column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row.get_type_of_column("a")
        Integer
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row.count()
        2
        """
        return self._data.shape[1]

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
        >>> row = Row.from_dict({"a": 1, "b": 2})
        >>> row.to_dict()
        {'a': 1, 'b': 2}
        """
        return {column_name: self.get_value(column_name) for column_name in self.get_column_names()}

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
        # noinspection PyProtectedMember
        return self._data._repr_html_()
