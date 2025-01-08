import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({}), []),
        (Table({"col1": []}), ["col1"]),
        (Table({"col1": [], "col2": []}), ["col1", "col2"]),
        (Table({"col1": [1], "col2": [1]}), ["col1", "col2"]),
    ],
    ids=["empty", "1 empty column", "2 empty columns", "non-empty columns"],
)
def test_should_return_column_names(table: Table, expected: list) -> None:
    assert table.column_names == expected
