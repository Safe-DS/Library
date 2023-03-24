import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError


def test_median_invalid() -> None:
    with pytest.raises(NonNumericColumnError):
        table = Table(pd.DataFrame(data={"col1": ["col1_1", 2]}))
        column = table.get_column("col1")
        column.median()


def test_median_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4, 5]}))
    column = table.get_column("col1")
    assert column.median() == 3


def test_median_valid_between_two_values() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4, 5, 6]}))
    column = table.get_column("col1")
    assert column.median() == 3.5
