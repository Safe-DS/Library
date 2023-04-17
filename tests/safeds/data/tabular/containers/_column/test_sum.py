import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import NonNumericColumnError


def test_sum_valid() -> None:
    c1 = Column("test", [1, 2])
    assert c1.sum() == 3


def test_sum_invalid() -> None:
    c1 = Column("test", [1, "a"])
    with pytest.raises(NonNumericColumnError):
        c1.sum()
