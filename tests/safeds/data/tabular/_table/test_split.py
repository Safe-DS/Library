import pandas as pd
import pytest
from safeds.data.tabular import Table


def test_split_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    result_train_table = Table(pd.DataFrame(data={"col1": [1, 2], "col2": [1, 2]}))
    result_test_table = Table(pd.DataFrame(data={"col1": [1], "col2": [4]}))
    train_table, test_table = table.split(2 / 3)
    assert result_test_table == test_table
    assert result_train_table == train_table


def test_split_invalid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))

    with pytest.raises(ValueError):
        table.split(0.0)

    with pytest.raises(ValueError):
        table.split(1.0)

    with pytest.raises(ValueError):
        table.split(-1.0)

    with pytest.raises(ValueError):
        table.split(2.0)
