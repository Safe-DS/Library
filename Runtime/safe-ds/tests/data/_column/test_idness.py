import pandas as pd
import pytest
from safeds.data import Column
from safeds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    "values, result",
    [(["A", "B"], 1), (["A", "A", "A", "B"], 0.5)],
)
def test_idness_valid(values: list[str], result: float) -> None:
    column: Column = Column(pd.Series(values), "test_idness_valid")
    idness = column.statistics.idness()
    assert idness == result


def test_idness_invalid() -> None:
    column = Column(pd.Series([], dtype=int), "test_idness_invalid")
    with pytest.raises(ColumnSizeError):
        column.statistics.idness()
