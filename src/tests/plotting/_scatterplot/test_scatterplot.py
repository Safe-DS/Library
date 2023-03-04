import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safeds import plotting
from safeds.data.tabular import Table
from safeds.exceptions import UnknownColumnNameError


def test_scatterplot(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3], "B": [2, 4, 7]}))
    plotting.scatterplot(table, "A", "B")


def test_scatterplot_wrong_column_name() -> None:
    with pytest.raises(UnknownColumnNameError):
        table = Table(pd.DataFrame(data={"A": [1, 2, 3], "B": [2, 4, 7]}))
        plotting.scatterplot(table, "C", "A")
