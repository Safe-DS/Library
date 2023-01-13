import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safe_ds import plotting
from safe_ds.data import Table
from safe_ds.exceptions import NonNumericColumnError


def test_plot_boxplot_complex() -> None:
    with pytest.raises(TypeError):
        table = Table(pd.DataFrame(data={"A": [1, 2, complex(1, -2)]}))
        plotting.plot_boxplot(table.get_column("A"))


def test_plot_boxplot_non_numeric() -> None:
    with pytest.raises(NonNumericColumnError):
        table = Table(pd.DataFrame(data={"A": [1, 2, "A"]}))
        plotting.plot_boxplot(table.get_column("A"))


def test_plot_boxplot_float(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3.5]}))
    plotting.plot_boxplot(table.get_column("A"))


def test_plot_boxplot_int(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3]}))
    plotting.plot_boxplot(table.get_column("A"))
