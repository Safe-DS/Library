import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "expected", "columns", "ignore_unknown_names", "should_raise"),
    [
        (Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table({"col1": [1, 2, 1]}), ["col2"], True, False),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table(), ["col1", "col2"], True, False),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), [], True, False),
        (Table(), Table(), [], True, False),
        (Table(), Table(), ["col1"], True, False),
        (Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table({"col1": [1, 2, 1]}), ["col2"], False, False),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table(), ["col1", "col2"], False, False),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), [], False, False),
        (Table(), Table(), [], False, False),
        (Table(), Table(), ["col1"], True, True),
        (Table(), Table(), ["col12"], False, True),
    ],
    ids=[
        "one column, ignore unknown names",
        "multiple columns, ignore unknown names",
        "no columns, ignore unknown names",
        "empty, ignore unknown names",
        "missing columns, ignore unknown names",
        "one column",
        "multiple columns",
        "no columns",
        "empty",
        "missing columns",
    ],
)
def test_should_remove_table_columns(table: Table, expected: Table, columns: list[str], ignore_unknown_names: bool, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ColumnNotFoundError):
            table.remove_columns(columns)
    else:
        table = table.remove_columns(columns, ignore_unknown_names=ignore_unknown_names)
        assert table.schema == expected.schema
        assert table == expected
        assert table.row_count == expected.row_count

