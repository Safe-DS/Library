import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", None),
        ("11", 11),
        ("11.5", 11.5),
        ("10e-1", 1.0),
        ("abc", None),
    ],
    ids=[
        "empty",
        "integer",
        "float",
        "scientific notation",
        "invalid string",
    ],
)
def test_should_parse_float(string: str, expected: bool) -> None:
    assert_cell_operation_works(string, lambda cell: cell.string.to_float(), expected)
