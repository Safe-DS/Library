import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), 0),
        (Table({"A": [1, 2, 3]}), 1),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_the_number_of_columns(table: Table, expected: int) -> None:
        row: Row[any] = _LazyVectorizedRow(table=table)
        assert row.column_count == expected
        


