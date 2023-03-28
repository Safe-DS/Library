from __future__ import annotations

from typing import Optional

import pandas as pd
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import InvertibleTableTransformer
from safeds.exceptions import LearningError, NotFittedError, UnknownColumnNameError
from sklearn import exceptions
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

        indices = [
            table.schema._get_column_index_by_name(name) for name in column_names
        ]

        wrapped_transformer = sk_OneHotEncoder()
        wrapped_transformer.fit(table._data[indices])

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

        data = table._data.copy()
        indices = [
            table.schema._get_column_index_by_name(name) for name in self._column_names
        ]
        data[indices] = pd.DataFrame(
            self._wrapped_transformer.transform(data[indices]), columns=indices
        )
        return Table(data, table.schema)

        try:
            table_k_columns = table.keep_only_columns(self._encoder.feature_names_in_)
            df_k_columns = table_k_columns._data
            df_k_columns.columns = table_k_columns.schema.get_column_names()
            df_new = pd.DataFrame(self._encoder.transform(df_k_columns).toarray())
            df_new.columns = self._encoder.get_feature_names_out()
            df_concat = table._data.copy()
            df_concat.columns = table.schema.get_column_names()
            data_new = pd.concat([df_concat, df_new], axis=1).drop(
                self._encoder.feature_names_in_, axis=1
            )
            return Table(data_new)
        except Exception as exc:
            raise NotFittedError from exc

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

        try:
            data = self._wrapped_transformer.inverse_transform(
                transformed_table.keep_only_columns(self._wrapped_transformer.get_feature_names_out())._data
            )
            df = pd.DataFrame(data)
            df.columns = self._wrapped_transformer.feature_names_in_
            new_table = Table(df)
            for col in transformed_table.drop_columns(
                self._wrapped_transformer.get_feature_names_out()
            ).to_columns():
                new_table = new_table.add_column(col)
            return new_table
        except exceptions.NotFittedError as exc:
            raise NotFittedError from exc
