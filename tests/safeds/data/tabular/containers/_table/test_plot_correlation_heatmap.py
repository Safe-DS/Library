import _pytest
import matplotlib.pyplot as plt
from safeds.data.tabular.containers import Table


def test_plot_correlation_heatmap_non_numeric(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, "A"], "B": [1, 2, 3]})
    table.plot_correlation_heatmap()


def test_plot_correlation_heatmap(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3.5], "B": [2, 4, 7]})
    table.plot_correlation_heatmap()
