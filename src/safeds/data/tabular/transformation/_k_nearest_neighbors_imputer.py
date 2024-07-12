from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _check_columns_exist, _ClosedBound
from safeds.data.tabular.containers import Table
from safeds.exceptions import TransformerNotFittedError

from ._table_transformer import TableTransformer

if TYPE_CHECKING:
    from sklearn.impute import KNNImputer as sk_KNNImputer


class KNearestNeighborsImputer(TableTransformer):
    """
    The KNearestNeighborsImputer replaces missing values in given Columns with the mean value of the K-nearest neighbors.

    Parameters
    ----------
    neighbor_count:
        The number of neighbors to consider when imputing missing values.
    column_names:
        The list of columns used to impute missing values. If 'None', all columns are used.
    value_to_replace:
        The placeholder for the missing values. All occurrences of`missing_values` will be imputed.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        neighbor_count: int,
        *,
        column_names: str | list[str] | None = None,
        value_to_replace: float | str | None = None,
    ) -> None:
        super().__init__(column_names)

        _check_bounds(name="neighbor_count", actual=neighbor_count, lower_bound=_ClosedBound(1))

        # parameter
        self._neighbor_count: int = neighbor_count
        self._value_to_replace: float | str | None = value_to_replace

        # attributes
        self._wrapped_transformer: sk_KNNImputer | None = None

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._neighbor_count,
            self._value_to_replace,
            # Leave out the internal state for faster hashing
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._wrapped_transformer is not None

    @property
    def neighbor_count(self) -> int:
        """The number of neighbors to consider when imputing missing values."""
        return self._neighbor_count

    @property
    def value_to_replace(self) -> float | str | None:
        """The value to replace."""
        return self._value_to_replace

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> KNearestNeighborsImputer:
        """
        Learn a transformation for a set of columns in a table.

        **Note:** This transformer is not modified.

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
        ColumnNotFoundError
            If one of the columns, that should be fitted is not in the table.
        """
        from sklearn.impute import KNNImputer as sk_KNNImputer

        if table.row_count == 0:
            raise ValueError("The KNearestNeighborsImputer cannot be fitted because the table contains 0 rows.")

        if self._column_names is None:
            column_names = table.column_names
        else:
            column_names = self._column_names
            _check_columns_exist(table, column_names)

        value_to_replace = self._value_to_replace

        if self._value_to_replace is None:
            from numpy import nan

            value_to_replace = nan

        wrapped_transformer = sk_KNNImputer(n_neighbors=self._neighbor_count, missing_values=value_to_replace)
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

        result = KNearestNeighborsImputer(self._neighbor_count, column_names=column_names)
        result._wrapped_transformer = wrapped_transformer

        return result

    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        **Note:** The given table is not modified.

        Parameters
        ----------
        table:
            The table to wich the learned transformation is applied.

        Returns
        -------
        transformed_table:
            The transformed table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer is not fitted.
        ColumnNotFoundError
            If one of the columns, that should be transformed is not in the table.
        """
        if self._column_names is None or self._wrapped_transformer is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)

        new_data = self._wrapped_transformer.transform(
            table.remove_columns_except(self._column_names)._data_frame,
        )

        return Table._from_polars_lazy_frame(
            table._lazy_frame.with_columns(new_data),
        )
