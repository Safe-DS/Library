from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from sklearn.impute import SimpleImputer

from ._table import Table


class ImputerStrategy(ABC):
    @abstractmethod
    def _augment_imputer(self, imputer: SimpleImputer) -> None:
        pass


# noinspection PyProtectedMember
class Imputer:
    """
    Imputes the data for a given Table.

    Parameters
    ----------
    strategy: ImputerStrategy
        The strategy to impute missing values
    """

    class Strategy:
        class Constant(ImputerStrategy):
            """
            An Imputer-Strategy for imputing the missing data with given constant values

            Parameters
            ----------
            value
                The given values to impute missing values
            """

            def __init__(self, value: Any):
                self.value = value

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "constant"
                imputer.fill_value = self.value

        class Mean(ImputerStrategy):
            """
            An Imputer-Strategy for imputing the missing data with mean values
            """

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "mean"

        class Median(ImputerStrategy):
            """
            An Imputer-Strategy for imputing the missing data with median values
            """

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "median"

        class Mode(ImputerStrategy):
            """
            An Imputer-Strategy for imputing the missing data with mode values
            """

            def _augment_imputer(self, imputer: SimpleImputer) -> None:
                imputer.strategy = "most_frequent"

    def __init__(self, strategy: ImputerStrategy):
        self._imp = SimpleImputer()
        strategy._augment_imputer(self._imp)
        self.column_names: list[str] = []

    def fit(self, table: Table, column_names: Optional[list[str]] = None) -> None:
        """
        Fit the imputer on the given dataset.

        Parameters
        ----------
        table: Table
            the table to learn the new value to impute
        column_names: Optional[list[str]]
            if the imputer should only run on specific columns, these columns can be specified here
        """
        if column_names is None:
            column_names = table.schema.get_column_names()

        self.column_names = column_names
        indices = [
            table.schema._get_column_index_by_name(name) for name in self.column_names
        ]
        self._imp.fit(table._data[indices])

    def transform(self, table: Table) -> Table:
        """
        Impute the missing values on the given dataset

        Parameters
        ----------
        table: Table
            the dataset to be imputed

        Returns
        -------
        table : Table
            a dataset that is equal to the given dataset, with missing values imputed to the given strategy
        """
        data = table._data.copy()
        indices = [
            table.schema._get_column_index_by_name(name) for name in self.column_names
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
        Fit the imputer on the given dataset and impute the missing values

        Parameters
        ----------
        table: Table
            the dataset to learn the new value to impute and to actually impute
        column_names: Optional[list[str]]
            if the imputer should only run on specific columns, these columns can be specified here

        Returns
        -------
        table : Table
            a dataset that is equal to the given dataset, with missing values imputed to the given strategy
        """
        self.fit(table, column_names)
        return self.transform(table)
