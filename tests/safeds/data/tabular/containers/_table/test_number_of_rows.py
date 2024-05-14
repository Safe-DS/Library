import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), 0),
        (Table({"col1": [1]}), 1),
        (Table({"col1": [1, 2]}), 2),
    ],
    ids=["empty", "a row", "2 rows"],
)
def test_should_return_number_of_rows(table: Table, expected: int) -> None:
    assert table.number_of_rows == expected
