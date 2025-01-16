import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType


def test_should_store_the_name() -> None:
    column = Column("col1", [1])
    assert column.name == "col1"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("col1", []), []),
        (Column("col1", [1]), [1]),
        (Column("col1", [1], type=ColumnType.string()), ["1"]),
    ],
    ids=[
        "empty",
        "non-empty (inferred type)",
        "non-empty (manifest type)",
    ],
)
def test_should_store_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("col1", []), ColumnType.null()),
        (Column("col1", [1]), ColumnType.int64()),
        (Column("col1", [1], type=ColumnType.string()), ColumnType.string()),
    ],
    ids=[
        "empty",
        "non-empty (inferred type)",
        "non-empty (manifest type)",
    ],
)
def test_should_have_correct_type(column: Column, expected: ColumnType) -> None:
    assert column.type == expected
