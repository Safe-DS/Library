import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import NonNumericColumnError
from syrupy import SnapshotAssertion


def test_should_return_table(snapshot_png_image: SnapshotAssertion) -> None:
    col = Column(
        "target",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    lag_plot = col.plot_lagplot(1)
    assert lag_plot == snapshot_png_image


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = Column(
        "target",
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThis time series target"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_lagplot(2)
