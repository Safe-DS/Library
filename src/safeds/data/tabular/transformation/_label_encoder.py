from __future__ import annotations

import warnings
from typing import Any

from safeds._utils import _structural_hash
from safeds._validation import _check_columns_exist
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.tabular.containers import Table
from safeds.exceptions import TransformerNotFittedError

from ._invertible_table_transformer import InvertibleTableTransformer


class LabelEncoder(InvertibleTableTransformer):
    """
    The LabelEncoder encodes one or more given columns into labels.

    Parameters
    ----------
    column_names:
        The list of columns used to fit the transformer. If `None`, all non-numeric columns are used.
    partial_order:
        The partial order of the labels. The labels are encoded in the order of the given list. Additional values are
        assigned labels in the order they are encountered during fitting.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        column_names: str | list[str] | None = None,
        partial_order: list[Any] | None = None,
    ) -> None:
        super().__init__(column_names)

        if partial_order is None:
            partial_order = []

        # Parameters
        self._partial_order = partial_order

        # Internal state
        self._mapping: dict[str, dict[Any, int]] | None = None  # Column name -> value -> label
        self._inverse_mapping: dict[str, dict[int, Any]] | None = None  # Column name -> label -> value

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._partial_order,
            # Leave out the internal state for faster hashing
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._mapping is not None and self._inverse_mapping is not None

    @property
    def partial_order(self) -> list[Any]:
        """The partial order of the labels."""
        return list(self._partial_order)  # defensive copy

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> LabelEncoder:
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
        ValueError
            If the table contains 0 rows.
        """
        if self._column_names is None:
            column_names = [name for name in table.column_names if not table.get_column_type(name).is_numeric]
        else:
            column_names = self._column_names
            _check_columns_exist(table, column_names)
            _warn_if_columns_are_numeric(table, column_names)

        if table.row_count == 0:
            raise ValueError("The LabelEncoder cannot be fitted because the table contains 0 rows")

        # Learn the transformation
        mapping = {}
        reverse_mapping = {}

        for name in column_names:
            # Remember partial order
            mapping[name] = {value: index for index, value in enumerate(self._partial_order)}
            reverse_mapping[name] = {index: value for value, index in mapping[name].items()}

            unique_values = table.get_column(name).get_distinct_values()
            for value in unique_values:
                if value not in mapping[name]:
                    label = len(mapping[name])
                    mapping[name][value] = label
                    reverse_mapping[name][label] = value

        # Create a copy with the learned transformation
        result = LabelEncoder(column_names=column_names, partial_order=self._partial_order)
        result._mapping = mapping
        result._inverse_mapping = reverse_mapping

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
        ValueError
            If the table contains 0 rows.
        """
        import polars as pl

        # Used in favor of is_fitted, so the type checker is happy
        if self._column_names is None or self._mapping is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)

        columns = [
            pl.col(name).replace_strict(self._mapping[name], default=None, return_dtype=pl.UInt32)
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
            If the specified columns of the input table contain non-numerical data.
        """
        import polars as pl

        # Used in favor of is_fitted, so the type checker is happy
        if self._column_names is None or self._inverse_mapping is None:
            raise TransformerNotFittedError

        _check_columns_exist(transformed_table, self._column_names)
        _check_columns_are_numeric(
            transformed_table,
            self._column_names,
            operation="inverse-transform with a LabelEncoder",
        )

        columns = [
            pl.col(name).replace_strict(self._inverse_mapping[name], default=None) for name in self._column_names
        ]

        return Table._from_polars_lazy_frame(
            transformed_table._lazy_frame.with_columns(columns),
        )


def _warn_if_columns_are_numeric(table: Table, column_names: list[str]) -> None:
    numeric_columns = table.remove_columns_except(column_names).remove_non_numeric_columns().column_names
    if numeric_columns:
        warnings.warn(
            f"The columns {numeric_columns} contain numerical data. "
            "The LabelEncoder is designed to encode non-numerical values into numerical values",
            UserWarning,
            stacklevel=2,
        )
