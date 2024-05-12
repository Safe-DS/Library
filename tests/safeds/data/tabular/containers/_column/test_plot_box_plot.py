import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import NonNumericColumnError
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [0]),
        Column("a", [0, 1]),
    ],
    ids=[
        "empty",
        "one row",
        "multiple rows",
    ],
)
def test_should_match_snapshot(column: Column, snapshot_png_image: SnapshotAssertion) -> None:
    box_plot = column.plot.box_plot()
    assert box_plot == snapshot_png_image


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    column = Column("a", ["A", "B", "C"])
    with pytest.raises(NonNumericColumnError):
        column.plot.box_plot()
