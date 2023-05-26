from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer as sk_SimpleImputer

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import TableTransformer
from safeds.data.tabular.typing import ImputerStrategy
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError


class Imputer(TableTransformer):
    """
    Replace missing values with the given strategy.

    Parameters
    ----------
    strategy : ImputerStrategy
        The strategy used to impute missing values. Use the classes nested inside `Imputer.Strategy` to specify it.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column, Table
    >>> from safeds.data.tabular.transformation import Imputer
    >>>
    >>> table = Table.from_columns(
    ...     [
    ...         Column("a", [1, 3, None]),
    ...         Column("b", [None, 2, 3]),
    ...     ],
    ... )
    >>> transformer = Imputer(Imputer.Strategy.Constant(0))
    >>> transformed_table = transformer.fit_and_transform(table)
    """

    class Strategy:
        class Constant(ImputerStrategy):
            """
            An imputation strategy for imputing missing data with given constant values.

            Parameters
            ----------
            value :
                The given value to impute missing values.
            """

            def __init__(self, value: Any):
                self._value = value

            def __str__(self) -> str:
                return f"Constant({self._value})"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "constant"
                imputer.fill_value = self._value

        class Mean(ImputerStrategy):
            """An imputation strategy for imputing missing data with mean values."""

            def __str__(self) -> str:
                return "Mean"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "mean"

        class Median(ImputerStrategy):
            """An imputation strategy for imputing missing data with median values."""

            def __str__(self) -> str:
                return "Median"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "median"

        class Mode(ImputerStrategy):
            """An imputation strategy for imputing missing data with mode values."""

            def __str__(self) -> str:
                return "Mode"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "most_frequent"

    def __init__(self, strategy: ImputerStrategy):
        self._strategy = strategy

        self._wrapped_transformer: sk_SimpleImputer | None = None
        self._column_names: list[str] | None = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: list[str] | None) -> Imputer:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table : Table
            The table used to fit the transformer.
        column_names : Optional[list[str]]
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer : TableTransformer
            The fitted transformer.
        """
        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = set(column_names) - set(table.column_names)
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(list(missing_columns))

        if isinstance(self._strategy, Imputer.Strategy.Mode):
            for name in column_names:
                if len(table.get_column(name).mode()) > 1:
                    raise IndexError("There are multiple most frequent values in a column given for the Imputer")

        wrapped_transformer = sk_SimpleImputer()
        self._strategy._augment_imputer(wrapped_transformer)
        wrapped_transformer.fit(table._data[column_names])

        result = Imputer(self._strategy)
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

        return result

    # noinspection PyProtectedMember
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
        """
        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names) - set(table.column_names)
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        data = table._data.copy()
        data[self._column_names] = pd.DataFrame(
            self._wrapped_transformer.transform(data[self._column_names]),
            columns=self._column_names,
        )
        return Table._from_pandas_dataframe(data, table.schema)

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
        Get the names of all new columns that have been added by the Imputer.

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
         Get the names of all columns that may have been changed by the Imputer.

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
        Get the names of all columns that have been removed by the Imputer.

        Returns
        -------
        removed_columns : list[str]
            A list of names of the removed columns, ordered as they appear in the table the Imputer was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted():
            raise TransformerNotFittedError
        return []
