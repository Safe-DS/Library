from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._validation import _check_columns_exist
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.tabular.containers import Table
from safeds.exceptions import TransformerNotFittedError

from ._invertible_table_transformer import InvertibleTableTransformer

if TYPE_CHECKING:
    import polars as pl


class RobustScaler(InvertibleTableTransformer):
    """
    The RobustScaler transforms column values to a range by removing the median and scaling to the interquartile range.

    Currently, for columns with high stability (IQR == 0), it will only substract the median and not scale to avoid dividing by zero.

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
        self._data_median: pl.DataFrame | None = None
        self._data_scale: pl.DataFrame | None = None

    def __hash__(self) -> int:
        # Leave out the internal state for faster hashing
        return super().__hash__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._data_median is not None and self._data_scale is not None

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> RobustScaler:
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
            If column_names contain a column name that is missing in the table.
        ColumnTypeError
            If at least one of the specified columns in the table contains non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        import polars as pl

        if self._column_names is None:
            column_names = [name for name in table.column_names if table.get_column_type(name).is_numeric]
        else:
            column_names = self._column_names
            _check_columns_exist(table, column_names)
            _check_columns_are_numeric(table, column_names, operation="fit a RobustScaler")

        if table.row_count == 0:
            raise ValueError("The RobustScaler cannot be fitted because the table contains 0 rows")

        _data_median = table._lazy_frame.select(column_names).median().collect()
        q1 = table._lazy_frame.select(column_names).quantile(0.25).collect()
        q3 = table._lazy_frame.select(column_names).quantile(0.75).collect()
        _data_scale = q3 - q1

        # To make sure there is no division by zero
        for col_e in column_names:
            _data_scale = _data_scale.with_columns(
                pl.when(pl.col(col_e) == 0).then(1).otherwise(pl.col(col_e)).alias(col_e),
            )

        # Create a copy with the learned transformation
        result = RobustScaler(column_names=column_names)
        result._data_median = _data_median
        result._data_scale = _data_scale

        return result

    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        **Note:** The given table is not modified.

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
        if self._column_names is None or self._data_median is None or self._data_scale is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)
        _check_columns_are_numeric(table, self._column_names, operation="transform with a RobustScaler")

        columns = [
            (pl.col(name) - self._data_median.get_column(name)) / self._data_scale.get_column(name)
            for name in self._column_names
        ]

        return Table._from_polars_lazy_frame(
            table._lazy_frame.with_columns(columns),
        )

    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation.

        **Note:** The given table is not modified.

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
        if self._column_names is None or self._data_median is None or self._data_scale is None:
            raise TransformerNotFittedError

        _check_columns_exist(transformed_table, self._column_names)
        _check_columns_are_numeric(
            transformed_table,
            self._column_names,
            operation="inverse-transform with a RobustScaler",
        )

        columns = [
            pl.col(name) * self._data_scale.get_column(name) + self._data_median.get_column(name)
            for name in self._column_names
        ]

        return Table._from_polars_lazy_frame(
            transformed_table._lazy_frame.with_columns(columns),
        )
