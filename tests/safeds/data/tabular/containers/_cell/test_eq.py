import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (3, 3, True),
        (3, 1.5, False),
        (1.5, 3, False),
        (1.5, 1.5, True),
    ],
    ids=[
        "int - int",
        "int - float",
        "float - int",
        "float - float",
    ],
)
class TestShouldComputeEquality:
    def test_dunder_method(self, value1: float, value2: float, expected: bool):
        assert_cell_operation_works(value1, lambda cell: cell == value2, expected)

    def test_dunder_method_inverted_order(self, value1: float, value2: float, expected: bool):
        assert_cell_operation_works(value1, lambda cell: value2 == cell, expected)

    def test_named_method(self, value1: float, value2: float, expected: bool):
        assert_cell_operation_works(value1, lambda cell: cell.eq(value2), expected)
