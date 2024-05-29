from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _check_columns_exist, _ClosedBound
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
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
    bin_count:
        The number of bins to be created.
    column_names:
        The list of columns used to fit the transformer. If `None`, all numeric columns are used.

    Raises
    ------
    OutOfBoundsError
        If the given `bin_count` is less than 2.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        bin_count: int = 5,
        *,
        column_names: str | list[str] | None = None,
    ) -> None:
        TableTransformer.__init__(self, column_names)

        _check_bounds("bin_count", bin_count, lower_bound=_ClosedBound(2))

        self._wrapped_transformer: sk_KBinsDiscretizer | None = None
        self._bin_count = bin_count

    def __hash__(self) -> int:
        return _structural_hash(
            TableTransformer.__hash__(self),
            self._bin_count,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._wrapped_transformer is not None

    @property
    def bin_count(self) -> int:
        """The number of bins to be created."""
        return self._bin_count

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> Discretizer:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer.

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

        if table.row_count == 0:
            raise ValueError("The Discretizer cannot be fitted because the table contains 0 rows")

        if self._column_names is None:
            column_names = [name for name in table.column_names if table.get_column_type(name).is_numeric]
        else:
            column_names = self._column_names
            _check_columns_exist(table, column_names)
            _check_columns_are_numeric(table, column_names, operation="fit a Discretizer")

        wrapped_transformer = sk_KBinsDiscretizer(n_bins=self._bin_count, encode="ordinal")
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

        result = Discretizer(self._bin_count, column_names=column_names)
        result._wrapped_transformer = wrapped_transformer

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

        if table.row_count == 0:
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
