from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Optional

from safeds.data.tabular.containers import Table


class TableTransformer(ABC):
    """
    A `TableTransformer` learns a transformation for a set of columns in a `Table` and can then apply the learned
    transformation to another `Table` with the same columns.
    """

    @abstractmethod
    def fit(self, table: Table, column_names: Optional[list[str]] = None) -> TableTransformer:
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

    @abstractmethod
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

    def fit_transform(self, table: Table, column_names: Optional[list[str]] = None) -> tuple[Table, TableTransformer]:
        """
        Learn a transformation for a set of columns in a table and apply the learned transformation to the same table.

        Parameters
        ----------
        table : Table
            The table used to fit the transformer. The transformer is then applied to this table.
        column_names : Optional[list[str]]
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        transformed_table : Table
            The transformed table.
        fitted_transformer : TableTransformer
            The fitted transformer.
        """
        fitted_transformer = self.fit(table, column_names)
        transformed_table = fitted_transformer.transform(table)
        return transformed_table, fitted_transformer


class InvertibleTableTransformer(ABC, TableTransformer):
    """
    An `InvertibleTableTransformer` is a `TableTransformer` that can also undo the learned transformation after it has
    been applied.
    """

    @abstractmethod
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
