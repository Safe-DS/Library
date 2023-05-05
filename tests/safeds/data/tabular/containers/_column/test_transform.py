import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("test", [1, 2]), Column("test", [2, 3])),
        (Column("test", [-0.5, 0, 4]), Column("test", [0.5, 1, 5])),
        (Column("test", []), Column("test", []))
    ],
    ids=[
        "series1",
        "series2",
        "empty series"
    ]
)
def test_should_transform_column(column: Column, expected: Column) -> None:
    assert column.transform(lambda it: it + 1) == expected
