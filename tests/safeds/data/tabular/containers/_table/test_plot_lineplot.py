import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    ("table", "x_name", "y_name"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "B"),
        (Table({"A": [1, 1, 2, 2], "B": [2, 4, 6, 8]}), "A", "B"),
        (Table({"A": [2, 1, 3, 3, 1, 2], "B": [6, 2, 5, 5, 4, 8]}), "A", "B"),
    ],
    ids=[
        "functional",
        "sorted grouped",
        "unsorted grouped",
    ],
)
def test_should_match_snapshot(table: Table, x_name: str, y_name: str, snapshot_png_image: SnapshotAssertion) -> None:
    line_plot = table.plot.line_plot(x_name, y_name)
    assert line_plot == snapshot_png_image


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
