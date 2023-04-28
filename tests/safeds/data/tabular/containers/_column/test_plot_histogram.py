import _pytest
import matplotlib.pyplot as plt
from safeds.data.tabular.containers import Table
from safeds.data.image.containers import Image
from tests.helpers import resolve_resource_path


def test_plot_histogram(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3]})
    table.get_column("A").plot_histogram()


def test_plot_histogram_legacy_check_str() -> None:
    table = Table.from_dict({"A": ["A", "B", "Apple"]})
    #table.get_column("A").plot_histogram().to_png_file(resolve_resource_path("./image/snapshot_histogram_str.png"))
    current = table.get_column("A").plot_histogram()
    legacy = Image.from_png_file(resolve_resource_path("./image/snapshot_histogram_str.png"))
    assert legacy._image.tobytes() == current._image.tobytes()

def test_plot_histogram_legacy_check_numeric() -> None:
    table = Table.from_dict({"A": [1, 2, 3]})
    #table.get_column("A").plot_histogram().to_png_file(resolve_resource_path("./image/snapshot_histogram_numeric.png"))
    current = table.get_column("A").plot_histogram()
    legacy = Image.from_png_file(resolve_resource_path("./image/snapshot_histogram_numeric.png"))
    assert legacy._image.tobytes() == current._image.tobytes()
