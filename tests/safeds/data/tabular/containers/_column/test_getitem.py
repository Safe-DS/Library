import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import IndexOutOfBoundsError


def test_getitem_valid() -> None:
    column = Column("testColumn", [0, "1"])
    assert column[0] == 0
    assert column[1] == "1"


# noinspection PyStatementEffect
def test_getitem_invalid() -> None:
    column = Column("testColumn", [0, "1"])
    with pytest.raises(IndexOutOfBoundsError):
        column[-1]

    with pytest.raises(IndexOutOfBoundsError):
        column[2]
