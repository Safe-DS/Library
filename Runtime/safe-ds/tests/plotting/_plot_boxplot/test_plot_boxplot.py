import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safe_ds import plotting
from safe_ds.data import Table


def test_plot_boxplot_complex():
    with pytest.raises(TypeError):
        table = Table(pd.DataFrame(data={"A": [1, 2, complex(1, -2)]}))
        plotting.plot_boxplot(table.get_column_by_name("A"))


def test_plot_boxplot_non_numeric():
    with pytest.raises(TypeError):
        table = Table(pd.DataFrame(data={"A": [1, 2, "A"]}))
        plotting.plot_boxplot(table.get_column_by_name("A"))


def test_plot_boxplot_float(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3.5]}))
    plotting.plot_boxplot(table.get_column_by_name("A"))


def test_plot_boxplot_int(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3]}))
    plotting.plot_boxplot(table.get_column_by_name("A"))
