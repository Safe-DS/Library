import numpy as np
import pandas as pd
from safeds.data import Imputer
from safeds.data.tabular import Table


def test_imputer_mean() -> None:
    table = Table(pd.DataFrame(data={"col1": [np.nan, 2, 3, 4, 5]}))
    column = table.get_column("col1")
    imp = Imputer(Imputer.Strategy.Mean())
    new_table = imp.fit_transform(table)

    assert new_table.get_column("col1")._data[0] == column.statistics.mean()


def test_imputer_median() -> None:
    table = Table(pd.DataFrame(data={"col1": [np.nan, 2, 3, 4, 5]}))
    column = table.get_column("col1")
    imp = Imputer(Imputer.Strategy.Median())
    new_table = imp.fit_transform(table)

    assert new_table.get_column("col1")._data[0] == column.statistics.median()


def test_imputer_mode() -> None:
    table = Table(pd.DataFrame(data={"col1": [np.nan, 2, 3, 4, 5]}))
    column = table.get_column("col1")
    imp = Imputer(Imputer.Strategy.Mode())
    new_table = imp.fit_transform(table)

    assert new_table.get_column("col1")._data[0] == column.statistics.mode()


def test_imputer_constant() -> None:
    table = Table(pd.DataFrame(data={"col1": [np.nan, 2, 3, 4, 5]}))
    imp = Imputer(Imputer.Strategy.Constant(0))
    new_table = imp.fit_transform(table)

    assert new_table.get_column("col1")._data[0] == 0


def test_imputer_specific_column() -> None:
    table = Table(
        pd.DataFrame(data={"col1": [np.nan, 2, 3, 4, 5], "col2": [np.nan, 2, 3, 4, 5]})
    )
    imp = Imputer(Imputer.Strategy.Constant(0))
    new_table = imp.fit_transform(table, ["col1"])

    assert new_table.get_column("col1")._data[0] == 0
    assert np.isnan(new_table.get_column("col2")._data[0])


def test_imputer_all_columns() -> None:
    table = Table(
        pd.DataFrame(data={"col1": [np.nan, 2, 3, 4, 5], "col2": [np.nan, 2, 3, 4, 5]})
    )
    imp = Imputer(Imputer.Strategy.Constant(0))
    new_table = imp.fit_transform(table)

    assert new_table.get_column("col1")._data[0] == 0
    assert new_table.get_column("col2")._data[0] == 0
