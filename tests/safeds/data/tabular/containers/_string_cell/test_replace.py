import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "old", "new", "expected"),
    [
        ("", "a", "b", ""),
        ("abc", "", "d", "dadbdcd"),
        ("abc", "a", "", "bc"),
        ("abc", "d", "e", "abc"),
        ("aba", "a", "d", "dbd"),
    ],
    ids=[
        "empty string",
        "empty old",
        "empty new",
        "no occurrences",
        "replace all occurrences",
    ],
)
def test_should_replace_all_occurrences_of_old_with_new(string: str, old: str, new: str, expected: str) -> None:
    assert_cell_operation_works(string, lambda cell: cell.str.replace(old, new), expected)
