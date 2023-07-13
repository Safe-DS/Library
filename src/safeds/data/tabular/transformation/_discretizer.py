from __future__ import annotations

from sklearn.preprocessing import KBinsDiscretizer as sk_KBinsDiscretizer

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import TableTransformer
from safeds.exceptions import (
    ClosedBound,
    NonNumericColumnError,
    OutOfBoundsError,
    TransformerNotFittedError,
    UnknownColumnNameError,
)


class Discretizer(TableTransformer):
    """
    The Discretizer bins continuous data into intervals.

    Parameters
    ----------
    number_of_bins: float
        The number of bins to be created.

    Raises
    ------
    OutOfBoundsError
        If the given number_of_bins is less than 2.
    """

    def __init__(self, number_of_bins: float = 5):
        self._column_names: list[str] | None = None
        self._wrapped_transformer: sk_KBinsDiscretizer | None = None

        if number_of_bins < 2:
            raise OutOfBoundsError(number_of_bins, name="number_of_bins", lower_bound=ClosedBound(2))
        self._number_of_bins = number_of_bins

    def fit(self, table: Table, column_names: list[str] | None) -> Discretizer:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table : Table
            The table used to fit the transformer.
        column_names : list[str] | None
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer : TableTransformer
            The fitted transformer.

        Raises
        ------
        ValueError
            If the table is empty.
        NonNumericColumnError
            If one of the columns, that should be fitted is non-numeric.
        UnknownColumnNameError
            If one of the columns, that should be fitted is not in the table.
        """
        if table.number_of_rows == 0:
            raise ValueError("The Discretizer cannot be fitted because the table contains 0 rows")

        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = set(column_names) - set(table.column_names)
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(
                    sorted(
                        missing_columns,
                        key={val: ix for ix, val in enumerate(column_names)}.__getitem__,
                    ),
                )

            for column in column_names:
                if not table.get_column(column).type.is_numeric():
                    raise NonNumericColumnError(f"{column} is of type {table.get_column(column).type}.")

        wrapped_transformer = sk_KBinsDiscretizer(n_bins=self._number_of_bins, encode="ordinal")
        wrapped_transformer.fit(table._data[column_names])

        result = Discretizer(self._number_of_bins)
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

        return result

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
        ValueError
            If the table is empty.
        UnknownColumnNameError
            If one of the columns, that should be transformed is not in the table.
        NonNumericColumnError
            If one of the columns, that should be fitted is non-numeric.
        """
        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        if table.number_of_rows == 0:
            raise ValueError("The table cannot be transformed because it contains 0 rows")

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names) - set(table.column_names)
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(
                sorted(
                    missing_columns,
                    key={val: ix for ix, val in enumerate(self._column_names)}.__getitem__,
                ),
            )

        for column in self._column_names:
            if not table.get_column(column).type.is_numeric():
                raise NonNumericColumnError(f"{column} is of type {table.get_column(column).type}.")

        data = table._data.copy()
        data.columns = table.column_names
        data[self._column_names] = self._wrapped_transformer.transform(data[self._column_names])
        return Table._from_pandas_dataframe(data)

    def is_fitted(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the transformer is fitted.
        """
        return self._wrapped_transformer is not None

    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the Discretizer.

        Returns
        -------
        added_columns : list[str]
            A list of names of the added columns, ordered as they will appear in the table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted():
            raise TransformerNotFittedError
        return []

    # (Must implement abstract method, cannot instantiate class otherwise.)
    def get_names_of_changed_columns(self) -> list[str]:
        """
         Get the names of all columns that may have been changed by the Discretizer.

        Returns
        -------
        changed_columns : list[str]
             The list of (potentially) changed column names, as passed to fit.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if self._column_names is None:
            raise TransformerNotFittedError
        return self._column_names

    def get_names_of_removed_columns(self) -> list[str]:
        """
        Get the names of all columns that have been removed by the Discretizer.

        Returns
        -------
        removed_columns : list[str]
            A list of names of the removed columns, ordered as they appear in the table the Discretizer was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted():
            raise TransformerNotFittedError
        return []
