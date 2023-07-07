from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from safeds.data.tabular.typing import Anything, Integer, Nothing, RealNumber
from safeds.data.tabular.typing._column_type import ColumnType
from safeds.exceptions import UnknownColumnNameError

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Schema:
    """
    Store column names and corresponding data types for a `Table` or `Row`.

    Parameters
    ----------
    schema : dict[str, ColumnType]
        Map from column names to data types.

    Examples
    --------
    >>> from safeds.data.tabular.typing import Integer, Schema, String
    >>> schema = Schema({"A": Integer(), "B": String()})
    """

    _schema: dict[str, ColumnType]

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_pandas_dataframe(dataframe: pd.DataFrame) -> Schema:
        """
        Create a schema from a `pandas.DataFrame`.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe.

        Returns
        -------
        schema : Schema
            The schema.
        """
        names = dataframe.columns
        # noinspection PyProtectedMember
        types = []
        for col in dataframe:
            types.append(ColumnType._data_type(dataframe[col]))

        return Schema(dict(zip(names, types, strict=True)))

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, schema: dict[str, ColumnType]):
        self._schema = dict(schema)  # Defensive copy

    def __hash__(self) -> int:
        """
        Return a hash value for the schema.

        Returns
        -------
        hash : int
            The hash value.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer(), "B": String()})
        >>> hash_value = hash(schema)
        """
        column_names = self._schema.keys()
        column_types = map(repr, self._schema.values())
        return hash(tuple(zip(column_names, column_types, strict=True)))

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of this row.

        Returns
        -------
        representation : str
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer()})
        >>> repr(schema)
        "Schema({'A': Integer})"
        """
        return f"Schema({self!s})"

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the schema.

        Returns
        -------
        string : str
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer()})
        >>> str(schema)
        "{'A': Integer}"
        """
        match len(self._schema):
            case 0:
                return "{}"
            case 1:
                return str(self._schema)
            case _:
                lines = (f"    {name!r}: {type_}" for name, type_ in self._schema.items())
                joined = ",\n".join(lines)
                return f"{{\n{joined}\n}}"

    @property
    def column_names(self) -> list[str]:
        """
        Return a list of all column names saved in this schema.

        Returns
        -------
        column_names : list[str]
            The column names.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer(), "B": String()})
        >>> schema.column_names
        ['A', 'B']
        """
        return list(self._schema.keys())

    def has_column(self, column_name: str) -> bool:
        """
        Return whether the schema contains a given column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        contains : bool
            True if the schema contains the column.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer(), "B": String()})
        >>> schema.has_column("A")
        True

        >>> schema.has_column("C")
        False
        """
        return column_name in self._schema

    def get_column_type(self, column_name: str) -> ColumnType:
        """
        Return the type of the given column.

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
            If the specified column name does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer(), "B": String()})
        >>> schema.get_column_type("A")
        Integer
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])
        return self._schema[column_name]

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, ColumnType]:
        """
        Return a dictionary that maps column names to column types.

        Returns
        -------
        data : dict[str, ColumnType]
            Dictionary representation of the schema.

        Examples
        --------
        >>> from safeds.data.tabular.typing import Integer, Schema, String
        >>> schema = Schema({"A": Integer(), "B": String()})
        >>> schema.to_dict()
        {'A': Integer, 'B': String}
        """
        return dict(self._schema)  # defensive copy

    @staticmethod
    def merge_multiple_schemas(schemas: list[Schema]) -> Schema:
        """
        Merge multiple schemas into one.

        For each type missmatch the new schema will have the least common supertype.

        The type hierarchy is as follows:
        * Anything
            * RealNumber
                * Integer
            * Boolean
            * String

        Parameters
        ----------
        schemas : list[Schema]
            the list of schemas you want to merge

        Returns
        -------
        schema : Schema
            the new merged schema

        Raises
        ------
        UnknownColumnNameError
            if not all schemas have the same column names
        """
        schema_dict = schemas[0]._schema
        missing_col_names = set()
        for schema in schemas:
            missing_col_names.update(set(schema.column_names) - set(schema_dict.keys()))
        if len(missing_col_names) > 0:
            raise UnknownColumnNameError(list(missing_col_names))
        for schema in schemas:
            if schema_dict != schema._schema:
                for col_name in schema_dict:
                    nullable = False
                    if schema_dict[col_name].is_nullable() or schema.get_column_type(col_name).is_nullable():
                        nullable = True
                    if isinstance(schema_dict[col_name], type(schema.get_column_type(col_name))):
                        if schema.get_column_type(col_name).is_nullable() and not schema_dict[col_name].is_nullable():
                            schema_dict[col_name] = type(schema.get_column_type(col_name))(nullable)
                        continue
                    if (
                        isinstance(schema_dict[col_name], RealNumber)
                        and isinstance(schema.get_column_type(col_name), Integer)
                    ) or (
                        isinstance(schema_dict[col_name], Integer)
                        and isinstance(schema.get_column_type(col_name), RealNumber)
                    ):
                        schema_dict[col_name] = RealNumber(nullable)
                        continue
                    if isinstance(schema_dict[col_name], Nothing):
                        schema_dict[col_name] = type(schema.get_column_type(col_name))(nullable)
                        continue
                    if isinstance(schema.get_column_type(col_name), Nothing):
                        schema_dict[col_name] = type(schema_dict[col_name])(nullable)
                        continue
                    schema_dict[col_name] = Anything(nullable)
        return Schema(schema_dict)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython Integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_markdown_(self) -> str:
        """
        Return a Markdown representation of the schema.

        Returns
        -------
        markdown : str
            The Markdown representation.
        """
        if len(self._schema) == 0:
            return "Empty Schema"

        lines = (f"| {name} | {type_} |" for name, type_ in self._schema.items())
        joined = "\n".join(lines)
        return f"| Column Name | Column Type |\n| --- | --- |\n{joined}"
