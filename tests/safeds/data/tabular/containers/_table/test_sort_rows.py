from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table", "comparator", "expected"),
    [
        (
            Table(),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table(),
        ),
        (
            Table({"col1": [3, 2, 1]}),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table({"col1": [1, 2, 3]}),
        ),
    ],

    ids=["empty", "3 rows"],

)
def test_should_return_sorted_table(
    table: Table,
    comparator: Callable[[Row, Row], int],
    expected: Table,
) -> None:
    assert table.sort_rows(comparator).schema == expected.schema
    assert table.sort_rows(comparator) == expected


@pytest.mark.parametrize(
    ("table", "comparator", "expected"),
    [
        (
            Table(),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table(),
        ),
        (
            Table({"col1": [3, 2, 1]}),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table({"col1": [3, 2, 1]}),
        ),
    ],

    ids=["empty", "3 rows"],

)
def test_should_not_modify_original_table(
    table: Table,
    comparator: Callable[[Row, Row], int],
    expected: Table,
) -> None:
    table.sort_rows(comparator)
    assert table.schema == expected.schema
    assert table == expected
