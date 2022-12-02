import pandas as pd
import pytest
from safe_ds.data import Column, ColumnType
from safe_ds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    "values, result",
    [(["A", "B"], 1), (["A", "A", "A", "B"], 0.5)],
)
def test_idness_valid(values: list[str], result: float):
    column: Column = Column(
        pd.Series(values),
        "test_idness_valid",
        ColumnType.from_numpy_dtype(pd.Series(values).dtype),
    )
    idness = column.idness()
    assert idness == result


def test_idness_invalid():
    column = Column(
        pd.Series([], dtype=int),
        "test_idness_invalid",
        ColumnType.from_numpy_dtype(pd.Series([], dtype=int).dtype),
    )
    with pytest.raises(ColumnSizeError):
        column.idness()
