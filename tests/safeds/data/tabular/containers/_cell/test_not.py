from typing import Any

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, True),
        (True, False),
        (0, True),
        (1, False),
    ],
    ids=[
        "false",
        "true",
        "falsy int",
        "truthy int",
    ],
)
class TestShouldInvertValueOfCell:
    def test_dunder_method(self, value: Any, expected: bool):
        assert_cell_operation_works(value, lambda cell: ~cell, expected)

    def test_named_method(self, value: Any, expected: bool):
        assert_cell_operation_works(value, lambda cell: cell.not_(), expected)
