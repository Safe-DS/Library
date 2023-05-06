import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("test", []), Column("test", [])),
        (Column("test", [1, 2]), Column("test", [2, 3])),
        (Column("test", [-0.5, 0, 4]), Column("test", [0.5, 1, 5])),
    ],
    ids=["empty", "integers", "floats"],
)
def test_should_transform_column(column: Column, expected: Column) -> None:
    assert column.transform(lambda it: it + 1) == expected


@pytest.mark.parametrize(
    ("column", "original"),
    [
        (Column("test", []), Column("test", [])),
        (Column("test", [1, 2]), Column("test", [1, 2])),
        (Column("test", [-0.5, 0, 4]), Column("test", [-0.5, 0, 4])),
    ],
    ids=["empty", "integers", "floats"],
)
def test_should_not_change_original_column(column: Column, original: Column) -> None:
    column.transform(lambda it: it + 1)
    assert column == original
