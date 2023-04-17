import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import NonNumericColumnError


def test_median_invalid() -> None:
    table = Table.from_dict({"col1": ["col1_1", 2]})
    column = table.get_column("col1")
    with pytest.raises(NonNumericColumnError):
        column.median()


def test_median_valid() -> None:
    table = Table.from_dict({"col1": [1, 2, 3, 4, 5]})
    column = table.get_column("col1")
    assert column.median() == 3


def test_median_valid_between_two_values() -> None:
    table = Table.from_dict({"col1": [1, 2, 3, 4, 5, 6]})
    column = table.get_column("col1")
    assert column.median() == 3.5
