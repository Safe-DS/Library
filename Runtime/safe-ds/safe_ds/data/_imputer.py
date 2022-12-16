from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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

    def fit(self, table: Table) -> None:
        """
        Fit the imputer on the given dataset.

        Parameters
        ----------
        table: Table
            the table to learn the new value to impute
        """
        self._imp.fit(table._data)

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
        names = table._data.columns
        return Table(pd.DataFrame(self._imp.transform(table._data), columns=names))

    def fit_transform(self, table: Table) -> Table:
        """
        Fit the imputer on the given dataset and impute the missing values

        Parameters
        ----------
        table: Table
            the dataset to learn the new value to impute and to actually impute

        Returns
        -------
        table : Table
            a dataset that is equal to the given dataset, with missing values imputed to the given strategy
        """
        self._imp.fit(table._data)
        names = table._data.columns
        return Table(pd.DataFrame(self._imp.transform(table._data), columns=names))
