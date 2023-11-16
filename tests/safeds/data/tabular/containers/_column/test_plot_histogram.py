from safeds.data.tabular.containers import Table
from syrupy import SnapshotAssertion


def test_should_match_snapshot_numeric(snapshot_png: SnapshotAssertion) -> None:
    table = Table({"A": [1, 2, 3]})
    histogram = table.get_column("A").plot_histogram()
    assert histogram == snapshot_png


def test_should_match_snapshot_str(snapshot_png: SnapshotAssertion) -> None:
    table = Table({"A": ["A", "B", "Apple"]})
    histogram = table.get_column("A").plot_histogram()
    assert histogram == snapshot_png
