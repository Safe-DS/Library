import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError


@pytest.mark.parametrize(
    ("table", "start", "length", "expected"),
    [
        (
            Table(),
            0,
            None,
            Table(),
        ),
        (
            Table({"col1": []}),
            0,
            None,
            Table({"col1": []}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            0,
            None,
            Table({"col1": [1, 2, 3]}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            1,
            None,
            Table({"col1": [2, 3]}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            10,
            None,
            Table({"col1": []}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            -1,
            None,
            Table({"col1": [3]}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            -10,
            None,
            Table({"col1": [1, 2, 3]}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            0,
            1,
            Table({"col1": [1]}),
        ),
        (
            Table({"col1": [1, 2, 3]}),
            0,
            10,
            Table({"col1": [1, 2, 3]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "full table",
        "positive start in bounds",
        "positive start out of bounds",
        "negative start in bounds",
        "negative start out of bounds",
        "positive length in bounds",
        "positive length out of bounds",
    ],
)
def test_should_slice_rows(table: Table, start: int, length: int | None, expected: Table) -> None:
    assert table.slice_rows(start, length) == expected


def test_should_raise_for_negative_length() -> None:
    table: Table = Table()
    with pytest.raises(OutOfBoundsError):
        table.slice_rows(0, -1)
