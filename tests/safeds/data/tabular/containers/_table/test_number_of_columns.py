import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), 0),
        (Table({"col1": []}), 1),
        (Table({"col1": [], "col2": []}), 2),
    ],
    ids=["empty", "a column", "2 columns"],
)
def test_should_return_number_of_columns(table: Table, expected: int) -> None:
    assert table.number_of_columns == expected
