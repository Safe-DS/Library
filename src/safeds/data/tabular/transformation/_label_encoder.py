from __future__ import annotations

import warnings
from typing import Any, Optional

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import (
    InvertibleTableTransformer,
)
from safeds.exceptions import NotFittedError, UnknownColumnNameError
from sklearn.preprocessing import OrdinalEncoder as sk_OrdinalEncoder


def warn(*_: Any, **__: Any) -> None:
    pass


warnings.warn = warn


# noinspection PyProtectedMember
class LabelEncoder(InvertibleTableTransformer):
    """
    The LabelEncoder encodes one or more given columns into labels.
    """

    def __init__(self) -> None:
        self._wrapped_transformer: Optional[sk_OrdinalEncoder] = None
        self._column_names: Optional[list[str]] = None

    def fit(self, table: Table, column_names: Optional[list[str]] = None) -> LabelEncoder:
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

        indices = [table.schema._get_column_index_by_name(name) for name in column_names]

        wrapped_transformer = sk_OrdinalEncoder()
        wrapped_transformer.fit(table._data[indices])

        result = LabelEncoder()
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

        return result

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
        data.columns = table.get_column_names()
        data[self._column_names] = self._wrapped_transformer.transform(data[self._column_names])
        return Table(data)

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
        data[self._column_names] = self._wrapped_transformer.inverse_transform(data[self._column_names])
        return Table(data)
