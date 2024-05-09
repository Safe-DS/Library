from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from safeds.data.tabular.containers import ExperimentalTable
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError

from ._experimental_invertible_table_transformer import ExperimentalInvertibleTableTransformer

if TYPE_CHECKING:
    from sklearn.preprocessing import OrdinalEncoder as sk_OrdinalEncoder


class ExperimentalLabelEncoder(ExperimentalInvertibleTableTransformer):
    """The LabelEncoder encodes one or more given columns into labels."""

    def __init__(self) -> None:
        self._wrapped_transformer: sk_OrdinalEncoder | None = None
        self._column_names: list[str] | None = None

    def fit(self, table: ExperimentalTable, column_names: list[str] | None) -> ExperimentalLabelEncoder:
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
        ValueError
            If the table contains 0 rows.
        """
        from sklearn.preprocessing import OrdinalEncoder as sk_OrdinalEncoder

        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = sorted(set(column_names) - set(table.column_names))
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(missing_columns)

        if table.number_of_rows == 0:
            raise ValueError("The LabelEncoder cannot transform the table because it contains 0 rows")

        if table.remove_columns_except(column_names).remove_non_numeric_columns().number_of_columns > 0:
            warnings.warn(
                "The columns"
                f" {table.remove_columns_except(column_names).remove_non_numeric_columns().column_names} contain"
                " numerical data. The LabelEncoder is designed to encode non-numerical values into numerical values",
                UserWarning,
                stacklevel=2,
            )

        # TODO: use polars Enum type instead:
        # my_enum = pl.Enum(['A', 'B', 'C']) <-- create this from the given order
        # my_data = pl.Series(['A', 'A', 'B'], dtype=my_enum)
        wrapped_transformer = sk_OrdinalEncoder()
        wrapped_transformer.set_output(transform="polars")
        wrapped_transformer.fit(
            table.remove_columns_except(column_names)._data_frame,
        )

        result = ExperimentalLabelEncoder()
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
            raise ValueError("The LabelEncoder cannot transform the table because it contains 0 rows")

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
            If the specified columns of the input table contain non-numerical data.
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
            raise ValueError("The LabelEncoder cannot inverse transform the table because it contains 0 rows")

        if transformed_table.remove_columns_except(
            self._column_names,
        ).remove_non_numeric_columns().number_of_columns < len(self._column_names):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(self._column_names)
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
        return ExperimentalTable._from_polars_lazy_frame(
            transformed_table._lazy_frame.update(new_data.lazy()),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._wrapped_transformer is not None

    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the LabelEncoder.

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
         Get the names of all columns that may have been changed by the LabelEncoder.

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
        Get the names of all columns that have been removed by the LabelEncoder.

        Returns
        -------
        removed_columns:
            A list of names of the removed columns, ordered as they appear in the table the LabelEncoder was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError
        return []
