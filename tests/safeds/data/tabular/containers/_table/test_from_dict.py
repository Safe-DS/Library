from typing import Any

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import RowCountMismatchError


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
    assert Table.from_dict(data).schema == expected.schema
    assert Table.from_dict(data) == expected


def test_should_raise_error_if_row_counts_differ() -> None:
    with pytest.raises(RowCountMismatchError):
        Table.from_dict({"a": [1, 2], "b": [3]})
