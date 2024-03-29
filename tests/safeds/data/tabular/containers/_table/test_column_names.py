import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({"col1": [1], "col2": [1]}), ["col1", "col2"]),
        (Table({"col": [], "gg": []}), ["col", "gg"]),
        (Table(), []),
    ],
    ids=["Integer", "rowless", "empty"],
)
def test_should_compare_column_names(table: Table, expected: list) -> None:
    assert table.column_names == expected
