import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safe_ds import plotting
from safe_ds.data import Table
from safe_ds.exceptions import NonNumericColumnError


def test_plot_correlation_heatmap_non_numeric() -> None:
    with pytest.raises(NonNumericColumnError):
        table = Table(pd.DataFrame(data={"A": [1, 2, "A"], "B": [1, 2, "A"]}))
        plotting.plot_correlation_heatmap(table)


def test_plot_correlation_heatmap(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3.5], "B": [2, 4, 7]}))
    plotting.plot_correlation_heatmap(table)
