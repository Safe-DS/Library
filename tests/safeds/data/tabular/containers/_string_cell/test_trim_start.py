import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", ""),
        ("abc", "abc"),
        (" abc", "abc"),
        ("abc ", "abc "),
        (" abc ", "abc "),
    ],
    ids=[
        "empty",
        "non-empty",
        "whitespace start",
        "whitespace end",
        "whitespace start and end",
    ],
)
def test_should_remove_whitespace_prefix(string: str, expected: str) -> None:
    assert_cell_operation_works(string, lambda cell: cell.str.trim_start(), expected)
