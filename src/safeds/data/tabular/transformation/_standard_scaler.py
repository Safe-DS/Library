from __future__ import annotations

from sklearn.preprocessing import StandardScaler as sk_StandardScaler

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import InvertibleTableTransformer
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError


class StandardScaler(InvertibleTableTransformer):
    """The StandardScaler transforms column values to a range by removing the mean and scaling to unit variance."""

    def __init__(self) -> None:
        self._column_names: list[str] | None = None
        self._wrapped_transformer: sk_StandardScaler | None = None

    def fit(self, table: Table, column_names: list[str] | None) -> StandardScaler:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table : Table
            The table used to fit the transformer.
        column_names : list[str] | None
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer : TableTransformer
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
        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = sorted(set(column_names) - set(table.column_names))
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(missing_columns)

        if table.number_of_rows == 0:
            raise ValueError("The StandardScaler cannot be fitted because the table contains 0 rows")

        if (
            table.keep_only_columns(column_names).remove_columns_with_non_numerical_values().number_of_columns
            < table.keep_only_columns(column_names).number_of_columns
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(table.keep_only_columns(column_names).column_names)
                        - set(
                            table.keep_only_columns(column_names)
                            .remove_columns_with_non_numerical_values()
                            .column_names,
                        ),
                    ),
                ),
            )

        wrapped_transformer = sk_StandardScaler()
        wrapped_transformer.fit(table._data[column_names])

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
        table : Table
            The table to which the learned transformation is applied.

        Returns
        -------
        transformed_table : Table
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
            raise ValueError("The StandardScaler cannot transform the table because it contains 0 rows")

        if (
            table.keep_only_columns(self._column_names).remove_columns_with_non_numerical_values().number_of_columns
            < table.keep_only_columns(self._column_names).number_of_columns
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(table.keep_only_columns(self._column_names).column_names)
                        - set(
                            table.keep_only_columns(self._column_names)
                            .remove_columns_with_non_numerical_values()
                            .column_names,
                        ),
                    ),
                ),
            )

        data = table._data.copy()
        data.columns = table.column_names
        data[self._column_names] = self._wrapped_transformer.transform(data[self._column_names])
        return Table._from_pandas_dataframe(data)

    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation.

        The table is not modified.

        Parameters
        ----------
        transformed_table : Table
            The table to be transformed back to the original version.

        Returns
        -------
        table : Table
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
            raise ValueError("The StandardScaler cannot transform the table because it contains 0 rows")

        if (
            transformed_table.keep_only_columns(self._column_names)
            .remove_columns_with_non_numerical_values()
            .number_of_columns
            < transformed_table.keep_only_columns(self._column_names).number_of_columns
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(transformed_table.keep_only_columns(self._column_names).column_names)
                        - set(
                            transformed_table.keep_only_columns(self._column_names)
                            .remove_columns_with_non_numerical_values()
                            .column_names,
                        ),
                    ),
                ),
            )

        data = transformed_table._data.copy()
        data.columns = transformed_table.column_names
        data[self._column_names] = self._wrapped_transformer.inverse_transform(data[self._column_names])
        return Table._from_pandas_dataframe(data)

    def is_fitted(self) -> bool:
        """
        Check if the transformer is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the transformer is fitted.
        """
        return self._wrapped_transformer is not None

    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the StandardScaler.

        Returns
        -------
        added_columns : list[str]
            A list of names of the added columns, ordered as they will appear in the table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted():
            raise TransformerNotFittedError
        return []

    # (Must implement abstract method, cannot instantiate class otherwise.)
    def get_names_of_changed_columns(self) -> list[str]:
        """
         Get the names of all columns that may have been changed by the StandardScaler.

        Returns
        -------
        changed_columns : list[str]
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
        Get the names of all columns that have been removed by the StandardScaler.

        Returns
        -------
        removed_columns : list[str]
            A list of names of the removed columns, ordered as they appear in the table the StandardScaler was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted():
            raise TransformerNotFittedError
        return []
