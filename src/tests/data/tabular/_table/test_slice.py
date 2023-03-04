import pandas as pd
import pytest
from safeds.data.tabular import Table


def test_slice_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    test_table = Table(pd.DataFrame(data={"col1": [1, 2], "col2": [1, 2]}))
    second_test_table = Table(pd.DataFrame(data={"col1": [1, 1], "col2": [1, 4]}))
    new_table = table.slice(0, 2, 1)
    second_new_table = table.slice(0, 3, 2)
    third_new_table = table.slice()
    assert new_table == test_table
    assert second_new_table == second_test_table
    assert third_new_table == table


def test_slice_invalid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))

    with pytest.raises(ValueError):
        table.slice(3, 2, 1)
    with pytest.raises(ValueError):
        table.slice(4, 0, 1)

    with pytest.raises(ValueError):
        table.slice(0, 4, 1)

    with pytest.raises(ValueError):
        table.slice(-4, 0, 1)

    with pytest.raises(ValueError):
        table.slice(0, -4, 1)
