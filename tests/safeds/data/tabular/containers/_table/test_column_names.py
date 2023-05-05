import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table.from_dict({"col1": [1], "col2": [1]}), ["col1", "col2"]),
        (Table.from_dict({}), []),
    ],
    ids=["Integer", "empty"],
)
def test_should_compare_column_names(table: Table, expected: list) -> None:
    assert table.column_names == expected
