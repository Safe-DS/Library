import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "column_name", "expected"),
    [
        (Table ({}), "A", False), 
        (Table({"A": [1, 2, 3]}), "A", True),
        (Table({"A": [1, 2, 3], "B": ["A", "A", "Bla"]}), "C", False),
        (Table({"col1": [1, 2, 3], "B": ["A", "A", "Bla"]}), 1, False),
    ],
    ids=[
        "empty row",
        "column exists",
        "column does not exist",
        "not a string",
    ],
)
def test_should_return_whether_the_row_has_the_column(table: Table, column_name: str, expected: bool) -> None:
    row: Row[any]=_LazyVectorizedRow(table=table)
    assert (column_name in row) == expected