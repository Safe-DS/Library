import pytest
from safeds.data.tabular.containers import Row, Table
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
    row1: Row[any] = _LazyVectorizedRow(table=table1)
    row2: Row[any] = _LazyVectorizedRow(table=table2)
    assert (row1.__eq__(row2)) == expected
