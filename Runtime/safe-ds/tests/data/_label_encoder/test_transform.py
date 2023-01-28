import pandas as pd
import pytest
from safeds.data import LabelEncoder
from safeds.data.tabular import Table
from safeds.exceptions import NotFittedError


def test_transform_valid() -> None:
    test_table = Table(
        pd.DataFrame({"cities": ["paris", "paris", "tokyo", "amsterdam"]})
    )
    le = LabelEncoder()
    le.fit(test_table, "cities")
    test_table = le.transform(test_table, "cities")
    assert test_table.schema.has_column("cities")
    assert test_table.to_columns()[0].get_value(0) == 1
    assert test_table.to_columns()[0].get_value(2) == 2
    assert test_table.to_columns()[0].get_value(3) == 0


def test_transform_invalid() -> None:
    test_table = Table(
        pd.DataFrame({"cities": ["paris", "paris", "tokyo", "amsterdam"]})
    )
    le = LabelEncoder()
    # le.fit(test_table) removed to force NotFittedError
    with pytest.raises(NotFittedError):
        le.transform(test_table, "cities")
