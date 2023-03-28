from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import TableTransformer
from safeds.exceptions import NotFittedError, UnknownColumnNameError
from sklearn.impute import SimpleImputer as sk_SimpleImputer


class ImputerStrategy(ABC):
    @abstractmethod
    def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
        pass


class Imputer(TableTransformer):
    """
    Impute the data for a given Table.

    Parameters
    ----------
    strategy : ImputerStrategy
        The strategy used to impute missing values.
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
            """
            An imputation strategy for imputing missing data with mean values.
            """

            def __str__(self) -> str:
                return "Mean"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "mean"

        class Median(ImputerStrategy):
            """
            An imputation strategy for imputing missing data with median values.
            """

            def __str__(self) -> str:
                return "Median"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "median"

        class Mode(ImputerStrategy):
            """
            An imputation strategy for imputing missing data with mode values.
            """

            def __str__(self) -> str:
                return "Mode"

            def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
                imputer.strategy = "most_frequent"

    def __init__(self, strategy: ImputerStrategy):
        self._strategy = strategy

        self._wrapped_transformer: Optional[sk_SimpleImputer] = None
        self._column_names: Optional[list[str]] = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: Optional[list[str]] = None) -> Imputer:
        """
        Learn a transformation for a set of columns in a table.

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
            column_names = table.get_column_names()
        else:
            missing_columns = set(column_names) - set(table.get_column_names())
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(list(missing_columns))

        if isinstance(self._strategy, Imputer.Strategy.Mode):
            for name in column_names:
                if len(table.get_column(name).mode()) > 1:
                    raise IndexError("There are multiple most frequent values in a column given for the Imputer")

        indices = [table.schema._get_column_index_by_name(name) for name in column_names]

        wrapped_transformer = sk_SimpleImputer()
        self._strategy._augment_imputer(wrapped_transformer)
        wrapped_transformer.fit(table._data[indices])

        result = Imputer(self._strategy)
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

        return result

    # noinspection PyProtectedMember
    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        Parameters
        ----------
        table : Table
            The table to which the learned transformation is applied.

        Returns
        -------
        transformed_table : Table
            The transformed table.

        Raises
        ----------
        NotFittedError
            If the transformer has not been fitted yet.
        """

        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise NotFittedError()

        # Input table does not contain all columns used to fit the transformer
        missing_columns = set(self._column_names) - set(table.get_column_names())
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(list(missing_columns))

        data = table._data.copy()
        indices = [table.schema._get_column_index_by_name(name) for name in self._column_names]
        data[indices] = pd.DataFrame(self._wrapped_transformer.transform(data[indices]), columns=indices)
        return Table(data, table.schema)
