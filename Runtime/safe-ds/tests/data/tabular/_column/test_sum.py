import pandas as pd
import pytest
from safeds.data.tabular import Column
from safeds.exceptions import NonNumericColumnError


def test_sum_valid() -> None:
    c1 = Column(pd.Series([1, 2]), "test")
    assert c1.statistics.sum() == 3


def test_sum_invalid() -> None:
    c1 = Column(pd.Series([1, "a"]), "test")
    with pytest.raises(NonNumericColumnError):
        c1.statistics.sum()
