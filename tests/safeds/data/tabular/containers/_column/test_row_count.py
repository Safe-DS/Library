import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a", []), 0),
        (Column("a", [0]), 1),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_the_number_of_rows(column: Column, expected: int) -> None:
    assert column.row_count == expected
