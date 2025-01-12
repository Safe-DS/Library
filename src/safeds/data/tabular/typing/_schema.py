from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_columns_exist

from ._polars_column_type import _PolarsColumnType

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from ._column_type import ColumnType


class Schema:
    """The schema of a row or table."""

    # ------------------------------------------------------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_polars_schema(schema: pl.Schema) -> Schema:
        result = object.__new__(Schema)
        result._schema = schema
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, schema: Mapping[str, ColumnType]) -> None:
        import polars as pl

        self._schema: pl.Schema = pl.Schema(
            [(name, type_._polars_data_type) for name, type_ in schema.items()],
            check_dtypes=False,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        if self is other:
            return True
        return self._schema == other._schema

    def __hash__(self) -> int:
        return _structural_hash(tuple(self._schema.keys()), [str(type_) for type_ in self._schema.values()])

    def __repr__(self) -> str:
        return f"Schema({self!s})"

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._schema)

    def __str__(self) -> str:
        match self._schema.len():
            case 0:
                return "{}"
            case 1:
                name, type_ = next(iter(self._schema.items()))
                return f"{{{name!r}: {_PolarsColumnType(type_)}}}"
            case _:
                lines = (f"    {name!r}: {_PolarsColumnType(type_)}" for name, type_ in self._schema.items())
                joined = ",\n".join(lines)
                return f"{{\n{joined}\n}}"

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_count(self) -> int:
        """
        The number of columns.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType, Schema
        >>> schema = Schema({"a": ColumnType.int64(), "b": ColumnType.float32()})
        >>> schema.column_count
        2
        """
        return self._schema.len()

    @property
    def column_names(self) -> list[str]:
        """
        The names of the columns.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType, Schema
        >>> schema = Schema({"a": ColumnType.int64(), "b": ColumnType.float32()})
        >>> schema.column_names
        ['a', 'b']
        """
        # polars already creates a defensive copy
        return self._schema.names()

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    def get_column_type(self, name: str) -> ColumnType:
        """
        Get the type of a column.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        type:
            The type of the column.

        Raises
        ------
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType, Schema
        >>> schema = Schema({"a": ColumnType.int64(), "b": ColumnType.float32()})
        >>> schema.get_column_type("a")
        int64
        """
        _check_columns_exist(self, name)

        return _PolarsColumnType(self._schema[name])

    def has_column(self, name: str) -> bool:
        """
        Check if the table has a column with a specific name.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        has_column:
            Whether the table has a column with the specified name.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType, Schema
        >>> schema = Schema({"a": ColumnType.int64(), "b": ColumnType.float32()})
        >>> schema.has_column("a")
        True

        >>> schema.has_column("c")
        False
        """
        return name in self._schema

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, ColumnType]:
        """
        Return a dictionary that maps column names to column types.

        Returns
        -------
        data:
            The dictionary representation of the schema.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        >>> table.schema.to_dict()
        {'A': int64, 'B': string}
        """
        return {name: _PolarsColumnType(type_) for name, type_ in self._schema.items()}

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_markdown_(self) -> str:
        """
        Return a Markdown representation of the schema for IPython.

        Returns
        -------
        markdown:
            The generated Markdown.
        """
        if self._schema.len() == 0:
            return "Empty schema"

        lines = (f"| {name} | {type_} |" for name, type_ in self._schema.items())
        joined = "\n".join(lines)
        return f"| Column Name | Column Type |\n| --- | --- |\n{joined}"
