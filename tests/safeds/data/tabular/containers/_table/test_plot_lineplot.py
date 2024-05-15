import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError
from syrupy import SnapshotAssertion

from tests.helpers import  resolve_resource_path


def test_should_match_snapshot(snapshot_png_image: SnapshotAssertion) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    lineplot = table.plot.line_plot("A", "B")

    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    table = table.replace_column("date", [table.get_column("date").from_str_to_temporal("%Y-%m-%d")])
    lineplot_2 = table.plot.line_plot("date", "value")
    assert lineplot == snapshot_png_image
    assert lineplot_2 == snapshot_png_image


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
