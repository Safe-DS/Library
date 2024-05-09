from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.tabular.containers import ExperimentalTable
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError

from ._experimental_invertible_table_transformer import ExperimentalInvertibleTableTransformer

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler as sk_MinMaxScaler


class ExperimentalRangeScaler(ExperimentalInvertibleTableTransformer):
    """
    The RangeScaler transforms column values by scaling each value to a given range.

    Parameters
    ----------
    min_:
        The minimum of the new range after the transformation
    max_:
        The maximum of the new range after the transformation

    Raises
    ------
    ValueError
        If the given minimum is greater or equal to the given maximum
    """

    def __init__(self, min_: float = 0.0, max_: float = 1.0):
        self._column_names: list[str] | None = None
        self._wrapped_transformer: sk_MinMaxScaler | None = None
        if min_ >= max_:
            raise ValueError('Parameter "maximum" must be higher than parameter "minimum".')
        self._minimum = min_
        self._maximum = max_

    def fit(self, table: ExperimentalTable, column_names: list[str] | None) -> ExperimentalRangeScaler:
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
        UnknownColumnNameError
            If column_names contain a column name that is missing in the table.
        NonNumericColumnError
            If at least one of the specified columns in the table contains non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        from sklearn.preprocessing import MinMaxScaler as sk_MinMaxScaler

        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = sorted(set(column_names) - set(table.column_names))
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(missing_columns)

        if table.number_of_rows == 0:
            raise ValueError("The RangeScaler cannot be fitted because the table contains 0 rows")

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

        wrapped_transformer = sk_MinMaxScaler((self._minimum, self._maximum))
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

        result = ExperimentalRangeScaler()
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

        return result

    def transform(self, table: ExperimentalTable) -> ExperimentalTable:
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
        UnknownColumnNameError
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
        missing_columns = sorted(set(self._column_names) - set(table.column_names))
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(missing_columns)

        if table.number_of_rows == 0:
            raise ValueError("The RangeScaler cannot transform the table because it contains 0 rows")

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
        return ExperimentalTable._from_polars_lazy_frame(
            table._lazy_frame.update(new_data.lazy()),
        )

    def inverse_transform(self, transformed_table: ExperimentalTable) -> ExperimentalTable:
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
        UnknownColumnNameError
            If the input table does not contain all columns used to fit the transformer.
        NonNumericColumnError
            If the transformed columns of the input table contain non-numerical data.
        ValueError
            If the table contains 0 rows.
        """
        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        missing_columns = sorted(set(self._column_names) - set(transformed_table.column_names))
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(missing_columns)

        if transformed_table.number_of_rows == 0:
            raise ValueError("The RangeScaler cannot transform the table because it contains 0 rows")

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

        import polars as pl

        new_data = pl.DataFrame(
            self._wrapped_transformer.inverse_transform(
                transformed_table.remove_columns_except(self._column_names)._data_frame,
            )
        )

        name_mapping = dict(zip(new_data.columns, self._column_names, strict=True))

        new_data = new_data.rename(name_mapping)

        return ExperimentalTable._from_polars_data_frame(
            transformed_table._data_frame.update(new_data),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._wrapped_transformer is not None

    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the RangeScaler.

        Returns
        -------
        added_columns:
            A list of names of the added columns, ordered as they will appear in the table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError
        return []

    # (Must implement abstract method, cannot instantiate class otherwise.)
    def get_names_of_changed_columns(self) -> list[str]:
        """
         Get the names of all columns that may have been changed by the RangeScaler.

        Returns
        -------
        changed_columns:
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
        Get the names of all columns that have been removed by the RangeScaler.

        Returns
        -------
        removed_columns:
            A list of names of the removed columns, ordered as they appear in the table the RangeScaler was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError
        return []
