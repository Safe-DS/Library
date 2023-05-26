from __future__ import annotations

from collections import Counter
from typing import Any

from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.transformation._table_transformer import (
    InvertibleTableTransformer,
)
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError, ValueNotPresentWhenFittedError


class OneHotEncoder(InvertibleTableTransformer):
    """
    A way to deal with categorical features that is particularly useful for unordered (i.e. nominal) data.

    It replaces a column with a set of columns, each representing a unique value in the original column. The value of
    each new column is 1 if the original column had that value, and 0 otherwise. Take the following table as an
    example:

    | col1 |
    |------|
    | "a"  |
    | "b"  |
    | "c"  |
    | "a"  |

    The one-hot encoding of this table is:

    | col1__a | col1__b | col1__c |
    |---------|---------|---------|
    | 1       | 0       | 0       |
    | 0       | 1       | 0       |
    | 0       | 0       | 1       |
    | 1       | 0       | 0       |

    The name "one-hot" comes from the fact that each row has exactly one 1 in it, and the rest of the values are 0s.
    One-hot encoding is closely related to dummy variable / indicator variables, which are used in statistics.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table
    >>> from safeds.data.tabular.transformation import OneHotEncoder
    >>> table = Table({"col1": ["a", "b", "c", "a"]})
    >>> transformer = OneHotEncoder()
    >>> transformer.fit_and_transform(table, ["col1"])
       col1__a  col1__b  col1__c
    0      1.0      0.0      0.0
    1      0.0      1.0      0.0
    2      0.0      0.0      1.0
    3      1.0      0.0      0.0
    """

    def __init__(self) -> None:
        # Maps each old column to (list of) new columns created from it:
        self._column_names: dict[str, list[str]] | None = None
        # Maps concrete values (tuples of old column and value) to corresponding new column names:
        self._value_to_column: dict[tuple[str, Any], str] | None = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: list[str] | None) -> OneHotEncoder:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table : Table
            The table used to fit the transformer.
        column_names : Optional[list[str]]
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer : TableTransformer
            The fitted transformer.
        """
        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = set(column_names) - set(table.column_names)
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(list(missing_columns))

        data = table._data.copy()
        data.columns = table.column_names

        result = OneHotEncoder()

        result._column_names = {}
        result._value_to_column = {}

        # Keep track of number of occurrences of column names;
        # initially all old column names appear exactly once:
        name_counter = Counter(data.columns)

        # Iterate through all columns to-be-changed:
        for column in column_names:
            result._column_names[column] = []
            for element in table.get_column(column).get_unique_values():
                base_name = f"{column}__{element}"
                name_counter[base_name] += 1
                new_column_name = base_name
                # Check if newly created name matches some other existing column name:
                if name_counter[base_name] > 1:
                    new_column_name += f"#{name_counter[base_name]}"
                # Update dictionary entries:
                result._column_names[column] += [new_column_name]
                result._value_to_column[(column, element)] = new_column_name

        return result

    # noinspection PyProtectedMember
    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        The table is not modified.

        Parameters
        ----------
        table : Table
            The table to which the learned transformation is applied.

        Returns
        -------
        transformed_table : Table
            The transformed table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        # Transformer has not been fitted yet
        if self._column_names is None or self._value_to_column is None:
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names.keys()) - set(table.column_names)
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        encoded_values = {}
        for new_column_name in self._value_to_column.values():
            encoded_values[new_column_name] = [0.0 for _ in range(table.number_of_rows)]

        for old_column_name in self._column_names:
            for i in range(table.number_of_rows):
                value = table.get_column(old_column_name).get_value(i)
                try:
                    new_column_name = self._value_to_column[(old_column_name, value)]
                except KeyError:
                    # This happens when a column in the to-be-transformed table contains a new value that was not
                    # already present in the table the OneHotEncoder was fitted on.
                    raise ValueNotPresentWhenFittedError(value, old_column_name) from None
                encoded_values[new_column_name][i] = 1.0

            for new_column in self._column_names[old_column_name]:
                table = table.add_column(Column(new_column, encoded_values[new_column]))

        # New columns may not be sorted:
        column_names = []
        for name in table.column_names:
            if name not in self._column_names.keys():
                column_names.append(name)
            else:
                column_names.extend(
                    [f_name for f_name in self._value_to_column.values() if f_name.startswith(name)],
                )

        # Drop old, non-encoded columns:
        # (Don't do this earlier - we need the old column nams for sorting,
        # plus we need to prevent the table from possibly having 0 columns temporarily.)
        table = table.remove_columns(list(self._column_names.keys()))

        # Apply sorting and return:
        return table.sort_columns(lambda col1, col2: column_names.index(col1.name) - column_names.index(col2.name))

    # noinspection PyProtectedMember
    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation.

        The table is not modified.

        Parameters
        ----------
        transformed_table : Table
            The table to be transformed back to the original version.

        Returns
        -------
        table : Table
            The original table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        # Transformer has not been fitted yet
        if self._column_names is None or self._value_to_column is None:
            raise TransformerNotFittedError

        original_columns = {}
        for original_column_name in self._column_names:
            original_columns[original_column_name] = [None for _ in range(transformed_table.number_of_rows)]

        for original_column_name, value in self._value_to_column:
            constructed_column = self._value_to_column[(original_column_name, value)]
            for i in range(transformed_table.number_of_rows):
                if transformed_table.get_column(constructed_column)[i] == 1.0:
                    original_columns[original_column_name][i] = value

        table = transformed_table

        for column_name, encoded_column in original_columns.items():
            table = table.add_column(Column(column_name, encoded_column))

        column_names = [
            (
                name
                if name not in [value for value_list in list(self._column_names.values()) for value in value_list]
                else list(self._column_names.keys())[
                    [
                        list(self._column_names.values()).index(value)
                        for value in list(self._column_names.values())
                        if name in value
                    ][0]
                ]
            )
            for name in table.column_names
        ]

        # Drop old column names:
        table = table.remove_columns(list(self._value_to_column.values()))

        return table.sort_columns(lambda col1, col2: column_names.index(col1.name) - column_names.index(col2.name))

    def is_fitted(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the transformer is fitted.
        """
        return self._column_names is not None and self._value_to_column is not None

    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the OneHotEncoder.

        Returns
        -------
        added_columns : list[str]
            A list of names of the added columns, ordered as they will appear in the table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if self._column_names is None:
            raise TransformerNotFittedError
        return [name for column_names in self._column_names.values() for name in column_names]

    # (Must implement abstract method, cannot instantiate class otherwise.)
    def get_names_of_changed_columns(self) -> list[str]:
        """
         Get the names of all columns that have been changed by the OneHotEncoder (none).

        Returns
        -------
        changed_columns : list[str]
             The empty list.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted():
            raise TransformerNotFittedError
        return []

    def get_names_of_removed_columns(self) -> list[str]:
        """
        Get the names of all columns that have been removed by the OneHotEncoder.

        Returns
        -------
        removed_columns : list[str]
            A list of names of the removed columns, ordered as they appear in the table the OneHotEncoder was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if self._column_names is None:
            raise TransformerNotFittedError
        return list(self._column_names.keys())
