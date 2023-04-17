import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import NonNumericColumnError


def test_maximum_invalid() -> None:
    table = Table.from_dict({"col1": ["col1_1", 2]})
    column = table.get_column("col1")
    with pytest.raises(NonNumericColumnError):
        column.maximum()


def test_maximum_valid() -> None:
    table = Table.from_dict({"col1": [1, 2, 3, 4]})
    column = table.get_column("col1")
    assert column.maximum() == 4
