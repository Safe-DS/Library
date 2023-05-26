from typing import Any

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnLengthMismatchError


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {},
            Table.from_dict({}),
        ),
        (
            {
                "a": [1],
                "b": [2],
            },
            Table.from_dict(
                {
                    "a": [1],
                    "b": [2],
                },
            ),
        ),
    ],
    ids=["empty", "with values"],
)
def test_should_create_table_from_dict(data: dict[str, list[Any]], expected: Table) -> None:
    assert Table.from_dict(data) == expected


def test_should_raise_error_if_columns_have_different_lengths() -> None:
    with pytest.raises(ColumnLengthMismatchError, match=r"The length of at least one column differs: \na: 2\nb: 1"):
        Table.from_dict({"a": [1, 2], "b": [3]})
