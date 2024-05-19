from typing import Any

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
        (0, False, False),
        (0, True, False),
        (1, False, False),
        (1, True, True),
    ],
    ids=[
        "false - false",
        "false - true",
        "true - false",
        "true - true",
        "falsy int - false",
        "falsy int - true",
        "truthy int - false",
        "truthy int - true",
    ],
)
class TestShouldComputeConjunction:
    def test_dunder_method(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell & value2, expected)

    def test_dunder_method_inverted_order(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value2, lambda cell: value1 & cell, expected)

    def test_named_method(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.and_(value2), expected)
