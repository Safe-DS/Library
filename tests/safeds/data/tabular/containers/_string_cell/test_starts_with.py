import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "prefix", "expected"),
    [
        ("", "a", False),
        ("abc", "", True),
        ("abc", "a", True),
        ("abc", "c", False),
    ],
    ids=[
        "empty string",
        "empty prefix",
        "starts with",
        "does not start with",
    ],
)
def test_should_check_whether_string_start_with_prefix(string: str, prefix: str, expected: bool) -> None:
    assert_cell_operation_works(string, lambda cell: cell.string.starts_with(prefix), expected)
