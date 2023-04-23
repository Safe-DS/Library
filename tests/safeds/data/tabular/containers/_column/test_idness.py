import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "result"),
    [
        (["A", "B"], 1),
        (["A", "A", "A", "B"], 0.5),
        (["A", "A", "A", "A"], 0.25),
    ],
    ids=[
        "all unique values",
        "some unique values",
        "all same values",
    ],
)
def test_should_return_idness_of_column(values: list[str], result: float) -> None:
    column = Column("A", values)
    assert column.idness() == result


def test_should_raise_if_column_is_empty() -> None:
    column = Column("A", [])
    with pytest.raises(ColumnSizeError):
        column.idness()
