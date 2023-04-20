from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import TransformerNotFittedError, UnknownColumnNameError
from safeds.data.tabular.transformation._table_transformer import (
    InvertibleTableTransformer,
)


class OneHotEncoder(InvertibleTableTransformer):
    """Encodes categorical columns to numerical features [0,1] that represent the existence for each value."""

    def __init__(self) -> None:
        self._wrapped_transformer: sk_OneHotEncoder | None = None
        self._column_names: dict[str, list[str]] | None = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: list[str] | None = None) -> OneHotEncoder:
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

        wrapped_transformer = sk_OneHotEncoder()
        wrapped_transformer.fit(data[column_names])

        result = OneHotEncoder()
        result._wrapped_transformer = wrapped_transformer
        result._column_names = {
            column: [f"{column}_{element}" for element in table.get_column(column).get_unique_values()]
            for column in column_names
        }

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
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names.keys()) - set(table.column_names)
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        original = table._data.copy()
        original.columns = table.schema.column_names

        one_hot_encoded = pd.DataFrame(
            self._wrapped_transformer.transform(original[self._column_names.keys()]).toarray(),
        )
        one_hot_encoded.columns = self._wrapped_transformer.get_feature_names_out()

        unchanged = original.drop(self._column_names.keys(), axis=1)

        res = Table(pd.concat([unchanged, one_hot_encoded], axis=1))
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
        return self._wrapped_transformer is not None
