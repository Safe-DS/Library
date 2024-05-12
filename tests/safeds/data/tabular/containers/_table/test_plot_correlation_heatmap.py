import pytest
from safeds.data.tabular.containers import Table
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    "table",
    [
        Table({"A": [1, 2, 3.5], "B": [0.2, 4, 77]}),
    ],
    ids=["normal"],
)
def test_should_match_snapshot(table: Table, snapshot_png_image: SnapshotAssertion) -> None:
    correlation_heatmap = table.plot.correlation_heatmap()
    assert correlation_heatmap == snapshot_png_image


def test_should_warn_about_empty_table() -> None:
    with pytest.warns(
        UserWarning,
        match=r"An empty table has been used. A correlation heatmap on an empty table will show nothing.",
    ):
        Table().plot.correlation_heatmap()
