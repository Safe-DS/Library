from typing import Any

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import LengthMismatchError


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {},
            Table({}),
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
    ids=["empty", "non-empty"],
)
def test_should_create_table_from_dict(data: dict[str, list[Any]], expected: Table) -> None:
    actual = Table.from_dict(data)
    assert actual == expected


def test_should_raise_if_row_counts_differ() -> None:
    with pytest.raises(LengthMismatchError):
        Table.from_dict({"a": [1, 2], "b": [3]})
