from typing import Any

import pytest

from safeds.data.tabular.containers import Cell
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
    ],
    ids=[
        "None",
        "int",
    ],
)
def test_should_return_constant_value(value: Any) -> None:
    assert_cell_operation_works(None, lambda _: Cell.constant(value), value)
