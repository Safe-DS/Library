import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", ""),
        ("AbC", "ABC"),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_uppercase_a_string(string: str, expected: str) -> None:
    assert_cell_operation_works(string, lambda cell: cell.string.to_uppercase(), expected)
