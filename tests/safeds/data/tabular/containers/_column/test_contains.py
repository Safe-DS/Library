from typing import Any

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        (Column("a", []), 1, False),
        (Column("a", [1, 2, 3]), 1, True),
        (Column("a", [1, 2, 3]), 4, False),
    ],
    ids=[
        "empty",
        "value exists",
        "value does not exist",
    ],
)
def test_should_check_whether_the_value_exists(column: Column, value: Any, expected: bool) -> None:
    assert (value in column) == expected
