import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import OutOfBoundsError
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "decimal_places", "expected"),
    [
        # Zero
        (0, 1, 0),
        (0.0, 1, 0),
        # Zero decimal places
        (0.1, 0, 0),
        (1, 0, 1),
        (1.1, 0, 1),
        # Rounding down
        (0.14, 1, 0.1),
        (0.104, 2, 0.1),
        # Rounding up
        (0.15, 1, 0.2),
        (0.105, 2, 0.11),
        # Overflow
        (9.99, 1, 10),
        (9.99, 2, 9.99),
        # None
        (None, 1, None),
    ],
    ids=[
        # Zero
        "0",
        "0.0",
        # Zero decimal places
        "0.1 (0 decimal places)",
        "1 (0 decimal places)",
        "1.1 (0 decimal places)",
        # Rounding down
        "0.14 (1 decimal places)",
        "0.104 (2 decimal places)",
        # Rounding up
        "0.15 (1 decimal places)",
        "0.105 (2 decimal places)",
        # Overflow
        "9.99 (1 decimal places)",
        "9.99 (2 decimal places)",
        # None
        "None",
    ],
)
def test_should_round_to_decimal_places(
    value: float | None,
    decimal_places: int,
    expected: float | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.math.round_to_decimal_places(decimal_places),
        expected,
        type_if_none=ColumnType.float64(),
    )


def test_should_raise_if_parameter_is_out_of_bounds() -> None:
    column = Column("a", [1])
    with pytest.raises(OutOfBoundsError):
        column.transform(lambda cell: cell.math.round_to_decimal_places(-1))
