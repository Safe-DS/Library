import pytest
from helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "substring", "expected"),
    [
        ("", "a", False),
        ("abc", "", True),
        ("abc", "a", True),
        ("abc", "d", False),
    ],
    ids=[
        "empty string",
        "empty substring",
        "contained",
        "not contained",
    ],
)
def test_should_check_whether_string_contains_substring(string: str, substring: str, expected: bool) -> None:
    assert_cell_operation_works([string], lambda cell: cell.string.contains(substring), [expected])
