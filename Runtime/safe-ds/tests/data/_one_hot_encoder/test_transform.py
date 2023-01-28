import pandas as pd
import pytest
from safeds.data import OneHotEncoder
from safeds.data.tabular import Table
from safeds.exceptions import NotFittedError


def test_transform_invalid() -> None:
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
        ohe.transform(table)
