from typing import Any

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import ColumnLengthMismatchError
from safeds.data.tabular.typing import Integer, Schema


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {},
            Table([]),
        ),
        (
            {
                "a": [1],
                "b": [2],
            },
            Table([[1, 2]], schema=Schema({"a": Integer(), "b": Integer()})),
        ),
    ],
    ids=["empty", "with values"]
)
def test_should_create_table_from_dict(data: dict[str, list[Any]], expected: Table) -> None:
    assert Table.from_dict(data) == expected


def test_should_raise_error_if_columns_have_different_lengths() -> None:
    with pytest.raises(ColumnLengthMismatchError):
        Table.from_dict({"a": [1, 2], "b": [3]})
