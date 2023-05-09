from __future__ import annotations

from collections import Counter

from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import TransformerNotFittedError, UnknownColumnNameError
from safeds.data.tabular.transformation._table_transformer import (
    InvertibleTableTransformer,
)


class OneHotEncoder(InvertibleTableTransformer):
    """Encodes categorical columns to numerical features [0,1] that represent the existence for each value."""

    def __init__(self) -> None:
        # Maps each old column to (list of) new columns created from it:
        self._column_names: dict[str, list[str]] | None = None
        # Maps concrete values (tuples of old column and value) to corresponding new column names:
        self._value_to_column: dict[tuple[str, any], str] | None = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: list[str] | None) -> OneHotEncoder:
        """
        Learn a transformation for a set of columns in a table.

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
                # TODO: Change to double underscore and change tests accordingly.
                base_name = f"{column}_{element}"
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
        if not self.is_fitted():
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names.keys()) - set(table.column_names)
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        # Make a copy of the table:
        # TODO: change to copy method once implemented
        new_table = table.remove_columns([])

        encoded_values = {}
        for new_column_name in self._value_to_column.values():
            encoded_values[new_column_name] = [0.0 for _ in range(table.number_of_rows)]

        for old_column_name in self._column_names:
            for i in range(table.number_of_rows):
                value = table.get_column(old_column_name).get_value(i)
                new_column_name = self._value_to_column[(old_column_name, value)]
                # TODO: Catch KeyError (this concrete value did not exist in the table the transformer was fitted on)
                #       and raise user-understandable Error/Exception.
                encoded_values[new_column_name][i] = 1.0

            for new_column in self._column_names[old_column_name]:
                new_table = new_table.add_column(Column(new_column, encoded_values[new_column]))

        # Drop corresponding old columns:
        new_table = new_table.remove_columns(list(self._column_names.keys()))

        column_names = []

        for name in table.column_names:
            if name not in self._column_names.keys():
                column_names.append(name)
            else:
                column_names.extend(
                    [f_name for f_name in self._value_to_column.values() if f_name.startswith(name)],
                )
        return new_table.sort_columns(lambda col1, col2: column_names.index(col1.name) - column_names.index(col2.name))

    # noinspection PyProtectedMember
    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation.

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
        if not self.is_fitted():
            raise TransformerNotFittedError

        # Make a copy of the table:
        # TODO: change to copy method once implemented
        new_table = transformed_table.remove_columns([])

        original_columns = {}
        for original_column_name in self._column_names:
            original_columns[original_column_name] = [None for _ in range(transformed_table.number_of_rows)]

        for (original_column_name, value) in self._value_to_column:
            constructed_column = self._value_to_column[(original_column_name, value)]
            for i in range(transformed_table.number_of_rows):
                if transformed_table.get_column(constructed_column)[i] == 1.0:
                    original_columns[original_column_name][i] = value

        for column_name, encoded_column in original_columns.items():
            new_table = new_table.add_column(Column(column_name, encoded_column))

        # Drop old column names:
        new_table = new_table.remove_columns(list(self._value_to_column.values()))

        column_names = [
            name
            if name not in [value for value_list in list(self._column_names.values()) for value in value_list]
            else list(self._column_names.keys())[
                [
                    list(self._column_names.values()).index(value)
                    for value in list(self._column_names.values())
                    if name in value
                ][0]
            ]
            for name in transformed_table.column_names
        ]
        return new_table.sort_columns(lambda col1, col2: column_names.index(col1.name) - column_names.index(col2.name))

    def is_fitted(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the transformer is fitted.
        """
        return self._column_names is not None
