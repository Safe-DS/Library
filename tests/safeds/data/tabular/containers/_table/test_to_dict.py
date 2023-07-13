from typing import Any

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table(),
            {},
        ),
        (
            Table(
                {
                    "a": [1],
                    "b": [2],
                },
            ),
            {
                "a": [1],
                "b": [2],
            },
        ),
    ],
    ids=["Empty table", "Table with one row"],
)
def test_should_return_dict_for_table(table: Table, expected: dict[str, list[Any]]) -> None:
    assert table.to_dict() == expected
