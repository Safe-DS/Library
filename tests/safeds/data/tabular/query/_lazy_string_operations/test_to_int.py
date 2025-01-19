import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "base", "expected"),
    [
        ("", 10, None),
        ("11", 10, 11),
        ("11", 2, 3),
        ("abc", 10, None),
    ],
    ids=[
        "empty",
        "11 base 10",
        "11 base 2",
        "invalid string",
    ],
)
def test_should_parse_integer(string: str, base: int, expected: bool) -> None:
    assert_cell_operation_works(
        string,
        lambda cell: cell.str.to_int(base=base),
        expected,
    )
