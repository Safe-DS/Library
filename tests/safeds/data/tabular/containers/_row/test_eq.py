import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (Table({"col1": []}), Table({"col1": []}), True),
        (Table({"col1": [1, 2]}), Table({"col1": [1, 2]}), True),
        (Table({"col1": [1, 2]}), Table({"col1": [2, 3]}), False),
        (Table({"col1": [1, 2]}), Table({"col2": [1, 2]}), False),
        (Table({"col1": ["1", "2"]}), Table({"col1": [1, 2]}), False),
    ],
    ids=[
        "empty rows",
        "equal rows",
        "different values",
        "different columns",
        "different types",
    ],
)
def test_should_return_whether_two_rows_are_equal(table1: Table, table2: Table, expected: bool) -> None:
    row1 = _LazyVectorizedRow(table=table1)
    row2 = _LazyVectorizedRow(table=table2)
    assert (row1.__eq__(row2)) == expected


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({"col1": []}), True),
        (Table({"col1": [1, 2]}), True),
    ],
    ids=[
        "empty table",
        "filled table",
    ],
)
def test_should_return_true_if_rows_are_strict_equal(table: Table, expected: bool) -> None:
    row1 = _LazyVectorizedRow(table=table)
    assert (row1.__eq__(row1)) == expected


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (Table({"col1": []}), Table({"col1": []})),
        (Table({"col1": [1, 2]}), Table({"col1": [1, 2]})),
    ],
    ids=[
        "empty tables",
        "filled tables",
    ],
)
def test_should_return_false_if_object_is_other_type(table1: Table, table2: Table) -> None:
    row1 = _LazyVectorizedRow(table=table1)
    assert (row1.__eq__(table2)) == NotImplemented
