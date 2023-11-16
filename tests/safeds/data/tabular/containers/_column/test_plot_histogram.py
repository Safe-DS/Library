from safeds.data.tabular.containers import Table


def test_should_match_snapshot_numeric(snapshot_png) -> None:
    table = Table({"A": [1, 2, 3]})
    histogram = table.get_column("A").plot_histogram()
    assert histogram == snapshot_png


def test_should_match_snapshot_str(snapshot_png) -> None:
    table = Table({"A": ["A", "B", "Apple"]})
    histogram = table.get_column("A").plot_histogram()
    assert histogram == snapshot_png
