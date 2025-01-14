from typing import Any

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        (Column("col1", []), 1, False),
        (Column("col1", [1]), 1, True),
        (Column("col1", [1]), 2, False),
        (Column("col1", [1]), "a", False),
    ],
    ids=[
        "empty",
        "value exists",
        "value does not exist",
        "different type",
    ],
)
def test_should_check_whether_the_value_exists(column: Column, value: Any, expected: bool) -> None:
    assert (value in column) == expected
