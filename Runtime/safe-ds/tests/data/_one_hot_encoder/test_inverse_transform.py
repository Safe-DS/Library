import pandas as pd
import pytest
from safeds.data import OneHotEncoder, Table
from safeds.exceptions import NotFittedError


def test_fit_transform() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C", "A"],
                "col2": ["Test1", "Test1", "Test3", "Test1"],
                "col3": [1, 2, 3, 4],
            }
        )
    )
    ohe = OneHotEncoder()
    table_ohe = ohe.fit_transform(table, ["col1", "col2"])
    table_old = ohe.inverse_transform(table_ohe)
    assert table_old == table


def test_fit_transform_invalid() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C", "A"],
                "col2": ["Test1", "Test1", "Test3", "Test1"],
                "col3": [1, 2, 3, 4],
            }
        )
    )
    ohe = OneHotEncoder()
    with pytest.raises(NotFittedError):
        ohe.inverse_transform(table)
