import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import OutOfBoundsError
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "significant_figures", "expected"),
    [
        # Zero
        (0, 1, 0),
        (0.0, 1, 0),
        # (0, 0.1)
        (0.05, 1, 0.05),
        (0.05, 2, 0.05),
        (0.054, 1, 0.05),
        (0.054, 2, 0.054),
        (0.055, 1, 0.06),
        (0.055, 2, 0.055),
        # [0.1, 1)
        (0.5, 1, 0.5),
        (0.5, 2, 0.5),
        (0.54, 1, 0.5),
        (0.54, 2, 0.54),
        (0.55, 1, 0.6),
        (0.55, 2, 0.55),
        # [1, 10)
        (5, 1, 5),
        (5, 2, 5),
        (5.4, 1, 5),
        (5.4, 2, 5.4),
        (5.5, 1, 6),
        (5.5, 2, 5.5),
        # [10, 100)
        (50, 1, 50),
        (50, 2, 50),
        (54, 1, 50),
        (54, 2, 54),
        (55, 1, 60),
        (55, 2, 55),
        # Overflow
        (9.99, 1, 10),
        (9.99, 2, 10),
        # None
        (None, 1, None),
    ],
    ids=[
        # Zero
        "0",
        "0.0",
        # (0, 0.1)
        "0.05 (1 sig fig)",
        "0.05 (2 sig fig)",
        "0.054 (1 sig fig)",
        "0.054 (2 sig fig)",
        "0.055 (1 sig fig)",
        "0.055 (2 sig fig)",
        # [0.1, 1)
        "0.5 (1 sig fig)",
        "0.5 (2 sig fig)",
        "0.54 (1 sig fig)",
        "0.54 (2 sig fig)",
        "0.55 (1 sig fig)",
        "0.55 (2 sig fig)",
        # [1, 10)
        "5 (1 sig fig)",
        "5 (2 sig fig)",
        "5.4 (1 sig fig)",
        "5.4 (2 sig fig)",
        "5.5 (1 sig fig)",
        "5.5 (2 sig fig)",
        # [10, 100)
        "50 (1 sig fig)",
        "50 (2 sig fig)",
        "54 (1 sig fig)",
        "54 (2 sig fig)",
        "55 (1 sig fig)",
        "55 (2 sig fig)",
        # Overflow
        "9.99 (1 sig fig)",
        "9.99 (2 sig fig)",
        # None
        "None",
    ],
)
def test_should_round_to_significant_figures(
    value: float | None,
    significant_figures: int,
    expected: float | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.math.round_to_significant_figures(significant_figures),
        expected,
        type_if_none=ColumnType.float64(),
    )


@pytest.mark.parametrize(
    "significant_figures",
    [
        -1,
        0,
    ],
    ids=[
        "negative",
        "zero",
    ],
)
def test_should_raise_if_parameter_is_out_of_bounds(significant_figures: int) -> None:
    column = Column("a", [1])
    with pytest.raises(OutOfBoundsError):
        column.transform(lambda cell: cell.math.round_to_significant_figures(significant_figures))
