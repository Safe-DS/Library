import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table.from_dict({}), 0),
        (Table.from_dict({"col1": [1]}), 1),
        (Table.from_dict({"col1": [1, 2]}), 2),
    ],
    ids=["empty", "a row", "2 rows"],
)
def test_should_compare_number_of_rows(table: Table, expected: int) -> None:
    assert table.number_of_rows == expected
