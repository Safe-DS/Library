from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table", "comparator", "expected"),
    [
        # Check that it works with an empty table when https://github.com/Safe-DS/Stdlib/issues/75 is fixed.
        (
            Table({"col1": [3, 2, 1]}),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table({"col1": [1, 2, 3]}),
        ),
    ],
)
def test_should_return_sorted_table(
    table: Table,
    comparator: Callable[[Row, Row], int],
    expected: Table,
) -> None:
    assert table.sort_rows(comparator) == expected


@pytest.mark.parametrize(
    ("table", "comparator", "expected"),
    [
        # Check that it works with an empty table when https://github.com/Safe-DS/Stdlib/issues/75 is fixed.
        (
            Table({"col1": [3, 2, 1]}),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table({"col1": [3, 2, 1]}),
        ),
    ],
)
def test_should_not_modify_original_table(
    table: Table,
    comparator: Callable[[Row, Row], int],
    expected: Table,
) -> None:
    table.sort_rows(comparator)
    assert table == expected


def test_should_not_sort_anything_on_empty_table() -> None:
    assert Table() == Table().sort_rows(lambda row1, row2: row1["col1"] - row2["col1"])
