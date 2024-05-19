import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (3, 2, 1),
        (3, 1.6, 1),
        (1.5, 3, 0),
        (1.5, 1.4, 1),
    ],
    ids=[
        "int - int",
        "int - float",
        "float - int",
        "float - float",
    ],
)
class TestShouldComputeDivision:
    def test_dunder_method(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value1, lambda cell: cell // value2, expected)

    def test_dunder_method_inverted_order(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value2, lambda cell: value1 // cell, expected)
