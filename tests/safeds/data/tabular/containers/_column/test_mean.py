import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError


def test_mean_invalid() -> None:
    table = Table(pd.DataFrame(data={"col1": ["col1_1", 2]}))
    column = table.get_column("col1")
    with pytest.raises(NonNumericColumnError):
        column.mean()


def test_mean_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4]}))
    column = table.get_column("col1")
    assert column.mean() == 2.5
