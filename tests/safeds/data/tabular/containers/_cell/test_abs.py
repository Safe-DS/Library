import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (0.0, 0.0),
        (10, 10),
        (10.5, 10.5),
        (-10, 10),
        (-10.5, 10.5),
    ],
    ids=[
        "zero int",
        "zero float",
        "positive int",
        "positive float",
        "negative int",
        "negative float",
    ],
)
class TestShouldReturnAbsoluteValueOfCell:
    def test_dunder_method(self, value: float, expected: float):
        assert_cell_operation_works(value, lambda cell: abs(cell), expected)

    def test_named_method(self, value: float, expected: float):
        assert_cell_operation_works(value, lambda cell: cell.abs(), expected)
