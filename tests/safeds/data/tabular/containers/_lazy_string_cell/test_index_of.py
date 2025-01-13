import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "substring", "expected"),
    [
        ("", "a", None),
        ("abc", "", 0),
        ("abc", "b", 1),
        ("abc", "d", None),
    ],
    ids=[
        "empty string",
        "empty substring",
        "contained",
        "not contained",
    ],
)
def test_should_return_index_of_first_occurrence_of_substring(string: str, substring: str, expected: bool) -> None:
    assert_cell_operation_works(string, lambda cell: cell.str.index_of(substring), expected)
