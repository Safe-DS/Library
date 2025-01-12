from typing import Any

import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table({}),
            {},
        ),
        (
            Table(
                {
                    "col1": [],
                    "col2": [],
                },
            ),
            {
                "col1": [],
                "col2": [],
            },
        ),
        (
            Table(
                {
                    "col1": [1, 2],
                    "col2": [3, 4],
                },
            ),
            {
                "col1": [1, 2],
                "col2": [3, 4],
            },
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
def test_should_return_dictionary(table: Table, expected: dict[str, list[Any]]) -> None:
    assert table.to_dict() == expected
