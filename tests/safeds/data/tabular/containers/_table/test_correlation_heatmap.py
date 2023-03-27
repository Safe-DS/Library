import _pytest
import matplotlib.pyplot as plt
import pandas as pd

from safeds.data.tabular.containers import Table


def test_correlation_heatmap_non_numeric(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, "A"], "B": [1, 2, 3]}))
    table.correlation_heatmap()


def test_correlation_heatmap(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3.5], "B": [2, 4, 7]}))
    table.correlation_heatmap()
