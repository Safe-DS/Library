import pytest
from helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", ""),
        ("AbC", "abc"),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_lowercase_a_string(string: str, expected: str) -> None:
    assert_cell_operation_works([string], lambda cell: cell.string.to_lowercase(), [expected])
