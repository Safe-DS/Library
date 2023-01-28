import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safeds import plotting
from safeds.data import Table
from safeds.exceptions import UnknownColumnNameError


def test_plot_lineplot(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3], "B": [2, 4, 7]}))
    plotting.plot_lineplot(table, "A", "B")


def test_plot_lineplot_wrong_column_name() -> None:
    with pytest.raises(UnknownColumnNameError):
        table = Table(pd.DataFrame(data={"A": [1, 2, 3], "B": [2, 4, 7]}))
        plotting.plot_lineplot(table, "C", "A")
