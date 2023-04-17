import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "result"),
    [(["A", "B"], 1), (["A", "A", "A", "B"], 0.5)],
)
def test_idness_valid(values: list[str], result: float) -> None:
    column = Column("test_idness_valid", values)
    idness = column.idness()
    assert idness == result


def test_idness_invalid() -> None:
    column = Column("test_idness_invalid", [])
    with pytest.raises(ColumnSizeError):
        column.idness()
