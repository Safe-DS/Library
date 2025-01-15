import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (0.0, 0.0),
        (10, -10),
        (10.5, -10.5),
        (-10, 10),
        (-10.5, 10.5),
        (None, None),
    ],
    ids=[
        "zero int",
        "zero float",
        "positive int",
        "positive float",
        "negative int",
        "negative float",
        "None",
    ],
)
class TestShouldNegateValue:
    def test_dunder_method(self, value: float | None, expected: float | None) -> None:
        assert_cell_operation_works(value, lambda cell: -cell, expected, type_if_none=ColumnType.float64())

    def test_named_method(self, value: float | None, expected: float | None) -> None:
        assert_cell_operation_works(value, lambda cell: cell.neg(), expected, type_if_none=ColumnType.float64())
