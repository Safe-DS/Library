import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a", []), []),
        (Column("a", [0]), [0]),
        (Column("a", [0, 1]), [0, 1]),
    ],
    ids=[
        "empty",
        "one row",
        "multiple rows",
    ],
)
def test_should_iterate_over_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected
