from typing import Any

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow
from safeds.exceptions import ColumnNotFoundError
from tests.helpers import assert_row_operation_works


@pytest.mark.parametrize(
    ("table", "name", "target", "expected"),
    [
        (
            Table({"A": [1, 2]}),
            "A",
            1,
            [True, False],
        ),
        (
            Table({"A": [1, 2], "B": [3, 4]}),
            "A",
            1,
            [True, False],
        ),
    ],
    ids=[
        "one column",
        "two columns",
    ],
)
def test_should_get_correct_item(
    table: Table,
    name: str,
    target: int,
    expected: list[Any],
) -> None:
    assert_row_operation_works(
        table,
        lambda row: row[name] == target,
        expected,
    )


@pytest.mark.parametrize(
    ("table", "name"),
    [
        (Table({}), "A"),
        (Table({"A": []}), "B"),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_raise_if_column_does_not_exist(table: Table, name: str) -> None:
    row = _LazyVectorizedRow(table)
    with pytest.raises(ColumnNotFoundError):
        _ignored = row[name]
