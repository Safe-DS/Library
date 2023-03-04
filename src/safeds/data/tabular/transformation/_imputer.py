from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from safeds.data.tabular import ColumnStatistics, Table
from sklearn.impute import SimpleImputer


class ImputerStrategy(ABC):
    @abstractmethod
    def _augment_imputer(self, imputer: SimpleImputer) -> None:
        pass


# noinspection PyProtectedMember
class Imputer:
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

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "constant"
                imputer.fill_value = self._value

        class Mean(ImputerStrategy):
            """
            An imputation strategy for imputing missing data with mean values.
            """

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "mean"

        class Median(ImputerStrategy):
            """
            An imputation strategy for imputing missing data with median values.
            """

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "median"

        class Mode(ImputerStrategy):
            """
            An imputation strategy for imputing missing data with mode values.
            """

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "most_frequent"

    def __init__(self, strategy: ImputerStrategy):
        self._imp = SimpleImputer()
        strategy._augment_imputer(self._imp)
        self._column_names: list[str] = []

    def fit(self, table: Table, column_names: Optional[list[str]] = None) -> None:
        """
        Fit the imputer on the dataset.

        Parameters
        ----------
        table : Table
            The table used to learn the imputation values.
        column_names : Optional[list[str]]
            An optional list of column names, if the imputer is only supposed to run on specific columns.
        """
        if column_names is None:
            column_names = table.schema.get_column_names()

        if self._imp.strategy == "most_frequent":
            for name in column_names:
                if 1 < len(ColumnStatistics(table.get_column(name)).mode()):
                    raise IndexError(
                        "There are multiple frequent values in a column given for the Imputer"
                    )

        self._column_names = column_names
        indices = [
            table.schema._get_column_index_by_name(name) for name in self._column_names
        ]
        self._imp.fit(table._data[indices])

    def transform(self, table: Table) -> Table:
        """
        Impute the missing values on the dataset.

        Parameters
        ----------
        table : Table
            The dataset to be imputed.

        Returns
        -------
        table : Table
            The dataset with missing values imputed by the given strategy.
        """
        data = table._data.copy()
        indices = [
            table.schema._get_column_index_by_name(name) for name in self._column_names
        ]
        data[indices] = pd.DataFrame(
            self._imp.transform(data[indices]), columns=indices
        )
        table_imputed = Table(data)
        table_imputed.schema = table.schema
        return table_imputed

    def fit_transform(
        self, table: Table, column_names: Optional[list[str]] = None
    ) -> Table:
        """
        Fit the imputer on the dataset and impute the missing values.

        Parameters
        ----------
        table : Table
            The table used to learn the imputation values.
        column_names : Optional[list[str]]
            An optional list of column names, if the imputer is only supposed to run on specific columns.

        Returns
        -------
        table : Table
            The dataset with missing values imputed by the given strategy.
        """
        self.fit(table, column_names)
        return self.transform(table)
