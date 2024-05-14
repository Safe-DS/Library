from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _check_columns_exist, _ClosedBound
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    NonNumericColumnError,
    TransformerNotFittedError,
)

from ._table_transformer import TableTransformer

if TYPE_CHECKING:
    from sklearn.preprocessing import KBinsDiscretizer as sk_KBinsDiscretizer


class Discretizer(TableTransformer):
    """
    The Discretizer bins continuous data into intervals.

    Parameters
    ----------
    number_of_bins:
        The number of bins to be created.

    Raises
    ------
    OutOfBoundsError
        If the given number_of_bins is less than 2.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, number_of_bins: int = 5) -> None:
        TableTransformer.__init__(self)

        _check_bounds("number_of_bins", number_of_bins, lower_bound=_ClosedBound(2))

        self._wrapped_transformer: sk_KBinsDiscretizer | None = None
        self._number_of_bins = number_of_bins

    def __hash__(self) -> int:
        return _structural_hash(
            TableTransformer.__hash__(self),
            self._number_of_bins,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_bins(self) -> int:
        return self._number_of_bins

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table, column_names: list[str] | None) -> Discretizer:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer.
        column_names:
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.

        Raises
        ------
        ValueError
            If the table is empty.
        NonNumericColumnError
            If one of the columns, that should be fitted is non-numeric.
        ColumnNotFoundError
            If one of the columns, that should be fitted is not in the table.
        """
        from sklearn.preprocessing import KBinsDiscretizer as sk_KBinsDiscretizer

        if table.number_of_rows == 0:
            raise ValueError("The Discretizer cannot be fitted because the table contains 0 rows")

        if column_names is None:
            column_names = table.column_names
        else:
            _check_columns_exist(table, column_names)

            for column in column_names:
                if not table.get_column(column).type.is_numeric:
                    raise NonNumericColumnError(f"{column} is of type {table.get_column(column).type}.")

        wrapped_transformer = sk_KBinsDiscretizer(n_bins=self._number_of_bins, encode="ordinal")
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

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
        table:
            The table to which the learned transformation is applied.

        Returns
        -------
        transformed_table:
            The transformed table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        ValueError
            If the table is empty.
        ColumnNotFoundError
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
        _check_columns_exist(table, self._column_names)

        for column in self._column_names:
            if not table.get_column(column).type.is_numeric:
                raise NonNumericColumnError(f"{column} is of type {table.get_column(column).type}.")

        new_data = self._wrapped_transformer.transform(
            table.remove_columns_except(self._column_names)._data_frame,
        )
        return Table._from_polars_lazy_frame(
            table._lazy_frame.update(new_data.lazy()),
        )
