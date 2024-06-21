from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._validation import _check_columns_exist
from safeds.data.tabular.containers import Table
from safeds.exceptions import TransformerNotFittedError

from ._table_transformer import TableTransformer

if TYPE_CHECKING:
    from sklearn.impute import KNNImputer as sk_KNNImputer

class KNearestNeighborsImputer(TableTransformer):
    """
    The KNearestNeighborsImputer replaces missing values in a table with the mean value of the K-nearest neighbors.

    Parameters
    ----------
    neighbor_count:
        The number of neighbors to consider when imputing missing values.
    column_names:
        The list of columns used to impute missing values. If 'None', all columns are used. 
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

        # parameter
        self._neighbor_count: int = neighbor_count
        self._value_to_replace: float | str | None = value_to_replace

        # attributes
        self._wrapped_transformer: sk_KNNImputer | None = None

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

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> KNearestNeighborsImputer:
        """
        Learn a trandformation for a set of columns in a table.

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
        ColumnNotFoundError
            If one of the columns, that should be fitted is not in the table.
        """
        from sklearn.impute import KNNImputer as sk_KNNImputer

        if table.row_count == 0:
            raise ValueError("The KNearestNeighborsImputer cannot be fitted because the table contains 0 rows.")
        
        if self._column_names is None:
            self._column_names = table.column_names
        else:
            column_names = self._column_names
            _check_columns_exist(Table, column_names)
        
        wrapped_transformer = sk_KNNImputer(missing_values=self._value_to_replace, n_neighbors=self._neighbor_count)
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

        result = KNearestNeighborsImputer(self._neighbor_count, column_names=self._column_names, value_to_replace=self._value_to_replace)
        result._wrapped_transformer = wrapped_transformer

        return result
    
    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        The Table is not modified.

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
        if self._column_names is None or self._neighbor_count is None or self._wrapped_transformer is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)

        new_data = self._wrapped_transformer.transform(
            table.remove_columns_except(self._column_names)._data_frame,
        )

        return Table._from_polars_lazy_frame(
            table._lazy_frame.update(new_data.lazify()),
        )