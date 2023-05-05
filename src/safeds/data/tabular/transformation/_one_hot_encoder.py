from __future__ import annotations

from collections import Counter
from typing import Tuple

import pandas as pd

from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import TransformerNotFittedError, UnknownColumnNameError
from safeds.data.tabular.transformation._table_transformer import (
    InvertibleTableTransformer,
)


class OneHotEncoder(InvertibleTableTransformer):
    """Encodes categorical columns to numerical features [0,1] that represent the existence for each value."""

    def __init__(self) -> None:
        # Maps old to new column names (useful for reversing the transformation):
        self._column_names: dict[str, list[str]] | None = None
        # Maps concrete values to new column names:
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
        # initially all old column names appear exactly ones:
        name_counter = Counter(data.columns)

        # Iterate through all columns to-be-changed:
        for column in column_names:
            result._column_names[column] = []
            for element in table.get_column(column).get_unique_values():
                base_name = f"{column}_{element}"
                name_counter[base_name] += 1
                new_column_name = base_name
                # Check if newly created name matches some other existing column name:
                if name_counter[base_name] > 1:
                    new_column_name += f"#{name_counter[base_name]}"
                result._column_names[column].append(new_column_name)
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
        # ~zzril: may change this to a call to is_fitted() ?
        if self._column_names is None:
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names.keys()) - set(table.column_names)
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        original = table._data.copy()
        original.columns = table.schema.column_names

        #one_hot_encoded = pd.DataFrame(
        #    self._wrapped_transformer.transform(original[self._column_names.keys()]).toarray(),
        #)
        #one_hot_encoded.columns = self._wrapped_transformer.get_feature_names_out()

        unchanged = original.drop(self._column_names.keys(), axis=1)

        #res = Table(pd.concat([unchanged, one_hot_encoded], axis=1))



        column_names = []

        for name in table.column_names:
            if name not in self._column_names.keys():
                column_names.append(name)
            else:
                column_names.extend(
                    [f_name for f_name in self._wrapped_transformer.get_feature_names_out() if f_name.startswith(name)],
                )
        res = res.sort_columns(lambda col1, col2: column_names.index(col1.name) - column_names.index(col2.name))

        return res

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
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        data = transformed_table._data.copy()
        data.columns = transformed_table.column_names

        decoded = pd.DataFrame(
            self._wrapped_transformer.inverse_transform(
                transformed_table.keep_only_columns(self._wrapped_transformer.get_feature_names_out())._data,
            ),
            columns=list(self._column_names.keys()),
        )
        unchanged = data.drop(self._wrapped_transformer.get_feature_names_out(), axis=1)

        res = Table(pd.concat([unchanged, decoded], axis=1))
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
        res = res.sort_columns(lambda col1, col2: column_names.index(col1.name) - column_names.index(col2.name))

        return res

    def is_fitted(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the transformer is fitted.
        """
        return self._column_names is not None
