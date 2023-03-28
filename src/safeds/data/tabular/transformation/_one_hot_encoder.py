from __future__ import annotations

from typing import Optional

import pandas as pd
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import (
    InvertibleTableTransformer,
)
from safeds.exceptions import NotFittedError, UnknownColumnNameError
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder


class OneHotEncoder(InvertibleTableTransformer):
    """
    The OneHotEncoder encodes categorical columns to numerical features [0,1] that represent the existence for each value.
    """

    def __init__(self) -> None:
        self._wrapped_transformer: Optional[sk_OneHotEncoder] = None
        self._column_names: Optional[list[str]] = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: Optional[list[str]] = None) -> OneHotEncoder:
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
            column_names = table.get_column_names()
        else:
            missing_columns = set(column_names) - set(table.get_column_names())
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(list(missing_columns))

        data = table._data.copy()
        data.columns = table.get_column_names()

        wrapped_transformer = sk_OneHotEncoder()
        wrapped_transformer.fit(data[column_names])

        result = OneHotEncoder()
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

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
        ----------
        NotFittedError
            If the transformer has not been fitted yet.
        """

        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise NotFittedError()

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names) - set(table.get_column_names())
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        original = table._data.copy()
        original.columns = table.schema.get_column_names()

        one_hot_encoded = pd.DataFrame(self._wrapped_transformer.transform(original[self._column_names]).toarray())
        one_hot_encoded.columns = self._wrapped_transformer.get_feature_names_out()

        unchanged = original.drop(self._column_names, axis=1)

        return Table(pd.concat([unchanged, one_hot_encoded], axis=1))

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
        ----------
        NotFittedError
            If the transformer has not been fitted yet.
        """
        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise NotFittedError()

        data = transformed_table._data.copy()
        data.columns = transformed_table.get_column_names()

        decoded = pd.DataFrame(
            self._wrapped_transformer.inverse_transform(transformed_table._data), columns=self._column_names
        )
        unchanged = data.drop(self._wrapped_transformer.get_feature_names_out(), axis=1)

        return Table(pd.concat([unchanged, decoded], axis=1))
