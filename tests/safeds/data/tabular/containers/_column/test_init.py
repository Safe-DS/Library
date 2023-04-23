import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import String, Boolean, Integer, RealNumber, ColumnType


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("A", []), "A"),
    ],
    ids=[
        "name A"
    ],
)
def test_should_store_the_name(column: Column, expected: str) -> None:
    assert column.name == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("A"), []),
        (Column("A", []), []),
        (Column("A", [1, 2, 3]), [1, 2, 3]),
    ],
    ids=[
        "empty",
        "empty (explicit)",
        "non-empty",
    ],
)
def test_should_store_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("A", []), String()),
        (Column("A", [True, False, True]), Boolean()),
        (Column("A", [1, 2, 3]), Integer()),
        (Column("A", [1.0, 2.0, 3.0]), RealNumber()),
        (Column("A", ["a", "b", "c"]), String()),
        (Column("A", [1, 2.0, "a", True]), String()),
    ],
    ids=["empty", "boolean", "integer", "real number", "string", "mixed"],
)
def test_should_infer_type(column: Column, expected: ColumnType) -> None:
    assert column.type == expected
