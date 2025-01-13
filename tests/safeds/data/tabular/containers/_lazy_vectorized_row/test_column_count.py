import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({}), 0),
        (Table({"col1": []}), 1),
        (Table({"col1": [1], "col2": [1]}), 2),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
def test_should_return_number_of_columns(table: Table, expected: int) -> None:
    row = _LazyVectorizedRow(table)
    assert row.column_count == expected
