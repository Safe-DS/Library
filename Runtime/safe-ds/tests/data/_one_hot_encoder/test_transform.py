import pandas as pd
import pytest
from safe_ds.data import Table
from safe_ds.data._one_hot_encoder import OneHotEncoder
from safe_ds.exceptions import NotFittedError


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
