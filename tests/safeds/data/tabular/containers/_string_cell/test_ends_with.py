import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "suffix", "expected"),
    [
        ("", "a", False),
        ("abc", "", True),
        ("abc", "c", True),
        ("abc", "a", False),
    ],
    ids=[
        "empty string",
        "empty suffix",
        "ends with",
        "does not end with",
    ],
)
def test_should_check_whether_string_ends_with_prefix(string: str, suffix: str, expected: bool) -> None:
    assert_cell_operation_works(string, lambda cell: cell.string.ends_with(suffix), expected)
