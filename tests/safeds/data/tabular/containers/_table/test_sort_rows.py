from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Cell, Row, Table


@pytest.mark.parametrize(
    ("table", "key_selector", "expected"),
    [
        (
            Table(),
            lambda row: row["col1"],
            Table(),
        ),
        (
            Table({"col1": [3, 2, 1]}),
            lambda row: row["col1"],
            Table({"col1": [1, 2, 3]}),
        ),
    ],
    ids=["empty", "3 rows"],
)
def test_should_return_sorted_table(
    table: Table,
    key_selector: Callable[[Row], Cell],
    expected: Table,
) -> None:
    assert table.sort_rows(key_selector).schema == expected.schema
    assert table.sort_rows(key_selector) == expected


@pytest.mark.parametrize(
    ("table", "key_selector", "expected"),
    [
        (
            Table(),
            lambda row: row["col1"],
            Table(),
        ),
        (
            Table({"col1": [3, 2, 1]}),
            lambda row: row["col1"],
            Table({"col1": [3, 2, 1]}),
        ),
    ],
    ids=["empty", "3 rows"],
)
def test_should_not_modify_original_table(
    table: Table,
    key_selector: Callable[[Row], Cell],
    expected: Table,
) -> None:
    table.sort_rows(key_selector)
    assert table.schema == expected.schema
    assert table == expected
