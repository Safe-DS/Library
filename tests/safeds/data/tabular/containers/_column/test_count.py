import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("col1", [1, 2, 3, 4, 5]), 5),
    ],
)
def test_count_valid(column: Column, expected: int) -> None:
    assert column.n_rows == expected
