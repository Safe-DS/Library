import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


# Test cases where no exception is expected
@pytest.mark.parametrize(
    ("table", "expected", "columns", "ignore_unknown_names"),
    [
        (Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table({"col1": [1, 2, 1]}), ["col2"], True),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table(), ["col1", "col2"], True),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), [], True),
        (Table(), Table(), [], True),
        (Table(), Table(), ["col1"], True),
        (Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}), Table({"col1": [1, 2, 1]}), ["col2"], False),
        (Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}), Table(), ["col1", "col2"], False),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            [],
            False,
        ),
        (Table(), Table(), [], False),
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
    ],
)
def test_should_remove_table_columns_no_exception(
    table: Table,
    expected: Table,
    columns: list[str],
    ignore_unknown_names: bool,
) -> None:
    table = table.remove_columns(columns, ignore_unknown_names=ignore_unknown_names)
    assert table.schema == expected.schema
    assert table == expected
    assert table.row_count == expected.row_count


# Test cases where an exception is expected
@pytest.mark.parametrize(
    ("table", "columns", "ignore_unknown_names"),
    [
        (Table(), ["col1"], False),
        (Table(), ["col12"], False),
    ],
    ids=[
        "missing columns",
        "missing columns",
    ],
)
def test_should_raise_error_for_unknown_columns(
    table: Table,
    columns: list[str],
    ignore_unknown_names: bool,
) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.remove_columns(columns, ignore_unknown_names=ignore_unknown_names)
