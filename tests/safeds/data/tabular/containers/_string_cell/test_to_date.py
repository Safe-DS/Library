import datetime

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", None),
        ("2022-01-09", datetime.date(2022, 1, 9)),
        ("abc", None),
    ],
    ids=[
        "empty",
        "ISO date",
        "invalid string",
    ],
)
def test_should_parse_date(string: str, expected: bool) -> None:
    assert_cell_operation_works(string, lambda cell: cell.string.to_date(), expected)
