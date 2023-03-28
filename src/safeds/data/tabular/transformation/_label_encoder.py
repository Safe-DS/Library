from __future__ import annotations

import warnings
from typing import Any, Optional

from sklearn.exceptions import NotFittedError as sk_NotFittedError
from sklearn.preprocessing import LabelEncoder as sk_LabelEncoder

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import InvertibleTableTransformer
from safeds.exceptions import LearningError, NotFittedError


def warn(*_: Any, **__: Any) -> None:
    pass


warnings.warn = warn


# noinspection PyProtectedMember


class LabelEncoder(InvertibleTableTransformer):
    """
    The LabelEncoder encodes one or more given columns into labels.
    """

    def __init__(self) -> None:
        self._wrapped_transformer: Optional[sk_LabelEncoder] = None
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
        try:
            self._wrapped_transformer.fit(table.keep_only_columns(column_names)._data)
        except sk_NotFittedError as exc:
            raise LearningError("") from exc

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
        p_df = table._data
        p_df.columns = table.schema.get_column_names()
        try:
            p_df[self._column_names] = self._wrapped_transformer.transform(p_df[self._column_names])
            return Table(p_df)
        except Exception as exc:
            raise NotFittedError from exc

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
        try:
            p_df = transformed_table._data
            p_df.columns = transformed_table.schema.get_column_names()
            p_df[self._column_names] = self._wrapped_transformer.inverse_transform(p_df[self._column_names])
            return Table(p_df)
        except sk_NotFittedError as exc:
            raise NotFittedError from exc
