import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import ColumnTypeError
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
    violin_plot = column.plot.violin_plot()
    assert violin_plot == snapshot_png_image


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
def test_should_match_dark_snapshot(column: Column, snapshot_png_image: SnapshotAssertion) -> None:
    violin_plot = column.plot.violin_plot(theme="dark")
    assert violin_plot == snapshot_png_image



def test_should_raise_if_column_contains_non_numerical_values() -> None:
    column = Column("a", ["A", "B", "C"])
    with pytest.raises(ColumnTypeError):
        column.plot.violin_plot()
