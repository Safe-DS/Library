import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({}), []),
        (Table({"col1": []}), ["col1"]),
        (Table({"col1": [1], "col2": [1]}), ["col1", "col2"]),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
def test_should_return_column_names(table: Table, expected: list[str]) -> None:
    row = _LazyVectorizedRow(table)
    assert row.column_names == expected
