from typing import Any

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
        (0, False, False),
        (0, True, True),
        (1, False, True),
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
class TestShouldComputeDisjunction:
    def test_dunder_method(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell | value2, expected)

    def test_dunder_method_inverted_order(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value2, lambda cell: value1 | cell, expected)

    def test_named_method(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.or_(value2), expected)
