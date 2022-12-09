import pandas as pd
import pytest
from safe_ds.data import Table


def test_median_invalid() -> None:
    with pytest.raises(TypeError):
        table = Table(pd.DataFrame(data={"col1": ["col1_1", 2]}))
        column = table.get_column("col1")
        column.statistics.median()


def test_median_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4, 5]}))
    column = table.get_column("col1")
    assert column.statistics.median() == 3


def test_median_valid_between_two_values() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6]}))
    column = table.get_column("col1")
    assert column.statistics.median() == 3.5
