import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table.from_dict({}), 0),
        (Table.from_dict({"col1": []}), 1),
        (Table.from_dict({"col1": [], "col2": []}), 2),
    ],
)
def test_number_of_columns(table: Table, expected: int) -> None:
    assert table.number_of_columns == expected
