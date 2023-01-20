import _pytest
import matplotlib.pyplot as plt
import pandas as pd
from safe_ds import plotting
from safe_ds.data import Table


def test_plot_histogram(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3]}))
    plotting.plot_histogram(table.get_column("A"))
