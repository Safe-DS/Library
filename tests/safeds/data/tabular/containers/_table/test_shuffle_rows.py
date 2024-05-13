import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table(), Table()),
        (Table({"col1": [1, 2, 3]}), Table({"col1": [3, 2, 1]})),
        (Table({"col1": [1, 2, 3], "col2": [4, 5, 6]}), Table({"col1": [3, 2, 1], "col2": [6, 5, 4]})),
    ],
    ids=[
        "empty",
        "one column",
        "multiple columns",
    ],
)
def test_should_shuffle_rows(table: Table, expected: Table) -> None:
    assert table.shuffle_rows() == expected
