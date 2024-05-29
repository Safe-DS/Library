from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._validation import _check_columns_exist
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.tabular.containers import Table
from safeds.exceptions import TransformerNotFittedError

from ._invertible_table_transformer import InvertibleTableTransformer

if TYPE_CHECKING:
    import polars as pl


class StandardScaler(InvertibleTableTransformer):
    """
    The StandardScaler transforms column values to a range by removing the mean and scaling to unit variance.

    Parameters
    ----------
    column_names:
        The list of columns used to fit the transformer. If `None`, all numeric columns are used.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, *, column_names: str | list[str] | None = None) -> None:
        super().__init__(column_names)

        # Internal state
        self._data_mean: pl.DataFrame | None = None
        self._data_standard_deviation: pl.DataFrame | None = None

    def __hash__(self) -> int:
        # Leave out the internal state for faster hashing
        return super().__hash__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._data_mean is not None and self._data_standard_deviation is not None

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> StandardScaler:
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
        ColumnNotFoundError
            If column_names contain a column name that is missing in the table.
        ColumnTypeError
            If at least one of the specified columns in the table contains non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        if self._column_names is None:
            column_names = [name for name in table.column_names if table.get_column_type(name).is_numeric]
        else:
            column_names = self._column_names
            _check_columns_exist(table, column_names)
            _check_columns_are_numeric(table, column_names, operation="fit a StandardScaler")

        if table.row_count == 0:
            raise ValueError("The StandardScaler cannot be fitted because the table contains 0 rows")

        # Learn the transformation (ddof=0 is used to match the behavior of scikit-learn)
        _data_mean = table._lazy_frame.select(column_names).mean().collect()
        _data_standard_deviation = table._lazy_frame.select(column_names).std(ddof=0).collect()

        # Create a copy with the learned transformation
        result = StandardScaler(column_names=column_names)
        result._data_mean = _data_mean
        result._data_standard_deviation = _data_standard_deviation

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
        ColumnNotFoundError
            If the input table does not contain all columns used to fit the transformer.
        ColumnTypeError
            If at least one of the columns in the input table that is used to fit contains non-numerical data.
        """
        import polars as pl

        # Used in favor of is_fitted, so the type checker is happy
        if self._column_names is None or self._data_mean is None or self._data_standard_deviation is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)
        _check_columns_are_numeric(table, self._column_names, operation="transform with a StandardScaler")

        columns = [
            (pl.col(name) - self._data_mean.get_column(name)) / self._data_standard_deviation.get_column(name)
            for name in self._column_names
        ]

        return Table._from_polars_lazy_frame(
            table._lazy_frame.with_columns(columns),
        )

    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation.

        The table is not modified.

        Parameters
        ----------
        transformed_table:
            The table to be transformed back to the original version.

        Returns
        -------
        original_table:
            The original table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        ColumnNotFoundError
            If the input table does not contain all columns used to fit the transformer.
        ColumnTypeError
            If the transformed columns of the input table contain non-numerical data.
        """
        import polars as pl

        # Used in favor of is_fitted, so the type checker is happy
        if self._column_names is None or self._data_mean is None or self._data_standard_deviation is None:
            raise TransformerNotFittedError

        _check_columns_exist(transformed_table, self._column_names)
        _check_columns_are_numeric(
            transformed_table,
            self._column_names,
            operation="inverse-transform with a StandardScaler",
        )

        columns = [
            pl.col(name) * self._data_standard_deviation.get_column(name) + self._data_mean.get_column(name)
            for name in self._column_names
        ]

        return Table._from_polars_lazy_frame(
            transformed_table._lazy_frame.with_columns(columns),
        )
