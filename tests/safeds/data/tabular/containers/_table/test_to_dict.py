from typing import Any

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Integer, Schema


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table([]),
            {},
        ),
        (
            Table([[1, 2]], schema=Schema({"a": Integer(), "b": Integer()})),
            {
                "a": [1],
                "b": [2],
            },
        ),
    ],
)
def test_should_return_dict_for_table(table: Table, expected: dict[str, list[Any]]) -> None:
    assert table.to_dict() == expected
