from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_columns_exist
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError

from ._invertible_table_transformer import InvertibleTableTransformer

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler as sk_StandardScaler


class StandardScaler(InvertibleTableTransformer):
    """The StandardScaler transforms column values to a range by removing the mean and scaling to unit variance."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self) -> None:
        InvertibleTableTransformer.__init__(self)

        self._wrapped_transformer: sk_StandardScaler | None = None

    def __hash__(self) -> int:
        return _structural_hash(
            InvertibleTableTransformer.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table, column_names: list[str] | None) -> StandardScaler:
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
        ColumnNotFoundError
            If column_names contain a column name that is missing in the table.
        NonNumericColumnError
            If at least one of the specified columns in the table contains non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        from sklearn.preprocessing import StandardScaler as sk_StandardScaler

        if column_names is None:
            column_names = table.column_names
        else:
            _check_columns_exist(table, column_names)

        if table.number_of_rows == 0:
            raise ValueError("The StandardScaler cannot be fitted because the table contains 0 rows")

        if (
            table.remove_columns_except(column_names).remove_non_numeric_columns().number_of_columns
            < table.remove_columns_except(column_names).number_of_columns
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(table.remove_columns_except(column_names).column_names)
                        - set(
                            table.remove_columns_except(column_names).remove_non_numeric_columns().column_names,
                        ),
                    ),
                ),
            )

        wrapped_transformer = sk_StandardScaler()
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

        result = StandardScaler()
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
        ColumnNotFoundError
            If the input table does not contain all columns used to fit the transformer.
        NonNumericColumnError
            If at least one of the columns in the input table that is used to fit contains non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        _check_columns_exist(table, self._column_names)

        if table.number_of_rows == 0:
            raise ValueError("The StandardScaler cannot transform the table because it contains 0 rows")

        if (
            table.remove_columns_except(self._column_names).remove_non_numeric_columns().number_of_columns
            < table.remove_columns_except(self._column_names).number_of_columns
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(table.remove_columns_except(self._column_names).column_names)
                        - set(
                            table.remove_columns_except(self._column_names).remove_non_numeric_columns().column_names,
                        ),
                    ),
                ),
            )

        new_data = self._wrapped_transformer.transform(
            table.remove_columns_except(self._column_names)._data_frame,
        )
        return Table._from_polars_lazy_frame(
            table._lazy_frame.update(new_data.lazy()),
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
        NonNumericColumnError
            If the transformed columns of the input table contain non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        _check_columns_exist(transformed_table, self._column_names)

        if transformed_table.number_of_rows == 0:
            raise ValueError("The StandardScaler cannot transform the table because it contains 0 rows")

        if (
            transformed_table.remove_columns_except(self._column_names).remove_non_numeric_columns().number_of_columns
            < transformed_table.remove_columns_except(self._column_names).number_of_columns
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(transformed_table.remove_columns_except(self._column_names).column_names)
                        - set(
                            transformed_table.remove_columns_except(self._column_names)
                            .remove_non_numeric_columns()
                            .column_names,
                        ),
                    ),
                ),
            )

        new_data = self._wrapped_transformer.inverse_transform(
            transformed_table.remove_columns_except(self._column_names)._data_frame,
        )
        return Table._from_polars_data_frame(
            transformed_table._data_frame.update(new_data),
        )

    @property
    def is_fitted(self) -> bool:
        return self._wrapped_transformer is not None
