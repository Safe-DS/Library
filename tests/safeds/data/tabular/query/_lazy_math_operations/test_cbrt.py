import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, -1),
        (0, 0),
        (1, 1),
        (3.375, 1.5),
        (8, 2),
        (None, None),
    ],
    ids=[
        "-1",
        "0",
        "1",
        "cube of float",
        "cube of int",
        "None",
    ],
)
def test_should_return_cube_root(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.cbrt(), expected)
