import datetime

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            [],
            None,
        ),
        (
            [None, None],
            None,
        ),
        (
            [1, 2, None],
            1,
        ),
        (
            [datetime.time(1, 2, 3), datetime.time(4, 5, 6), None],
            datetime.time(1, 2, 3),
        ),
        (
            ["a", "b", None],
            "a",
        ),
        (
            [True, False, None],
            False,
        ),
    ],
    ids=[
        "empty",
        "null column",
        "numeric column",
        "temporal column",
        "string column",
        "boolean column",
    ],
)
def test_should_return_minimum(values: list, expected: int) -> None:
    column = Column("col1", values)
    assert column.min() == expected
