import pytest
from safeds.data.tabular.containers import Table
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    "table",
    [
        Table({"time": [0, 1, 1, 2], "A": [1, 2, 1, 3.5], "B": [0.2, 4, 1, 5]}),
    ],
    ids=["normal"],
)
def test_should_match_snapshot(table: Table, snapshot_png_image: SnapshotAssertion) -> None:
    correlation_heatmap = table.plot.line_plot("time", ["A", "B"])
    assert correlation_heatmap == snapshot_png_image
