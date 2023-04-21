import _pytest
import matplotlib.pyplot as plt

from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from tests.helpers import resolve_resource_path


def test_plot_correlation_heatmap_non_numeric(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, "A"], "B": [1, 2, 3]})
    table.plot_correlation_heatmap()


def test_plot_correlation_heatmap(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3.5], "B": [2, 4, 7]})
    table.plot_correlation_heatmap()

def test_plot_heatmap_legacy_check() -> None:
    table = Table.from_dict({"A": [1, 2, 3.5], "B": [0.2, 4, 77]})
    #table.plot_correlation_heatmap().to_png_file(resolve_resource_path("./image/legacy_heatmap.png"))
    current = table.plot_correlation_heatmap()
    legacy = Image.from_png_file(resolve_resource_path("./image/legacy_heatmap.png"))
    assert legacy._image.tobytes() == current._image.tobytes()
