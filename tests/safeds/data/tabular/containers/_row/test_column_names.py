import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({}), []),
        (Table({"A": [1, 2, 3]}), ["A"]),
        (
            Table({"A": [1, 2, 3], "B": ["A", "A", "Bla"], "C": [True, True, False], "D": [1.0, 2.1, 4.5]}),
            ["A", "B", "C", "D"],
        ),
    ],
    ids=[
        "empty",
        "one-column",
        "four-column",
    ],
)
def test_should_return_the_column_names(table: Table, expected: list[str]) -> None:
    row = _LazyVectorizedRow(table=table)
    assert row.column_names == expected
