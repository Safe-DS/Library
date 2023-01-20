import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safe_ds import plotting
from safe_ds.data import Table
from safe_ds.exceptions import UnknownColumnNameError


def test_plot_scatterplot(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3], "B": [2, 4, 7]}))
    plotting.plot_scatterplot(table, "A", "B")


def test_plot_scatterplot_wrong_column_name() -> None:
    with pytest.raises(UnknownColumnNameError):
        table = Table(pd.DataFrame(data={"A": [1, 2, 3], "B": [2, 4, 7]}))
        plotting.plot_scatterplot(table, "C", "A")
