import pytest
from safeds.exceptions import OutOfBoundsError

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "start", "length", "expected"),
    [
        ("", 0, None, ""),
        ("abc", 0, None, "abc"),
        ("abc", 1, None, "bc"),
        ("abc", 10, None, ""),
        ("abc", -1, None, "c"),
        ("abc", -10, None, "abc"),
        ("abc", 0, 1, "a"),
        ("abc", 0, 10, "abc"),
    ],
    ids=[
        "empty",
        "full string",
        "positive start in bounds",
        "positive start out of bounds",
        "negative start in bounds",
        "negative start out of bounds",
        "positive length in bounds",
        "positive length out of bounds",
    ],
)
def test_should_return_substring(string: str, start: int, length: int | None, expected: str) -> None:
    assert_cell_operation_works(string, lambda cell: cell.str.substring(start, length), expected)


def test_should_raise_if_length_is_negative() -> None:
    with pytest.raises(OutOfBoundsError):
        assert_cell_operation_works("abc", lambda cell: cell.str.substring(length=-1), None)
