import pandas as pd
import pytest
from safeds.data.tabular import Table
from safeds.exceptions import NonNumericColumnError


def test_max_invalid() -> None:
    with pytest.raises(NonNumericColumnError):
        table = Table(pd.DataFrame(data={"col1": ["col1_1", 2]}))
        column = table.get_column("col1")
        column.statistics.max()


def test_max_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 3, 4]}))
    column = table.get_column("col1")
    assert column.statistics.max() == 4
