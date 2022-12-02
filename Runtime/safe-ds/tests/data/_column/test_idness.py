import pandas as pd
import pytest
from safe_ds.data import Column
from safe_ds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    "values, result",
    [(["A", "B"], 1), (["A", "A", "A", "B"], 0.5)],
)
def test_idness_valid(values: list[str], result: float):
    column: Column = Column(pd.Series(values))
    idness = column.idness()
    assert idness == result


def test_idness_invalid():
    column = Column(pd.Series([], dtype=int))
    with pytest.raises(ColumnSizeError):
        column.idness()
