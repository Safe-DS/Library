import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError
from syrupy import SnapshotAssertion


def test_should_match_snapshot(snapshot_png_image: SnapshotAssertion) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    lineplot = table.plot.line_plot("A", "B")
    assert lineplot == snapshot_png_image


@pytest.mark.parametrize(
    ("table", "x", "y"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "C"),
        (Table(), "x", "y"),
    ],
    ids=["x column", "y column", "empty"],
)
def test_should_raise_if_column_does_not_exist(table: Table, x: str, y: str) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(ColumnNotFoundError):
        table.plot.line_plot(x, y)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ("C", "A"),
        ("A", "C"),
        ("C", "D"),
    ],
    ids=["x column", "y column", "x and y column"],
)
def test_should_raise_if_column_does_not_exist_error_message(x: str, y: str) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(ColumnNotFoundError):
        table.plot.line_plot(x, y)
