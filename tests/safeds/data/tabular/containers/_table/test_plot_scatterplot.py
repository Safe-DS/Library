import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    ("table", "x_name", "y_name"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "B"),
        (
            Table(
                {
                    "A": [1, 0.99, 0.99, 2],
                    "B": [1, 0.99, 1.01, 2],
                 }),
            "A",
            "B",
        ),
    ],
    ids=[
        "functional",
        "overlapping",
    ],
)
def test_should_match_snapshot(table: Table, x_name: str, y_name: str, snapshot_png_image: SnapshotAssertion) -> None:
    scatterplot = table.plot.scatter_plot(x_name, y_name)
    assert scatterplot == snapshot_png_image


@pytest.mark.parametrize(
    ("table", "col1", "col2"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "B", "C"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "D"),
        (Table(), "C", "D"),
    ],
    ids=["First argument doesn't exist", "Second argument doesn't exist", "Both arguments do not exist", "empty"],
)
def test_should_raise_if_column_does_not_exist(table: Table, col1: str, col2: str) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.plot.scatter_plot(col1, col2)

@pytest.mark.parametrize(
    ("table", "x_name", "y_name"),
    [
        (Table({"A": ["a", "b", "c"], "B": [2, 4, 7]}), "A", "B"),
        (Table({"A": [1, 2, 3], "B": ["a", "b", "c"]}), "A", "B"),
    ],
)
def test_should_raise_if_columns_are_not_numeric(table: Table, x_name: str, y_name: str) -> None:
    with pytest.raises(ColumnTypeError):
        table.plot.scatter_plot(x_name, y_name)
