import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from safeds import plotting
from safeds.data.tabular import Table
from safeds.exceptions import NonNumericColumnError


def test_correlation_heatmap_non_numeric() -> None:
    with pytest.raises(NonNumericColumnError):
        table = Table(pd.DataFrame(data={"A": [1, 2, "A"], "B": [1, 2, "A"]}))
        plotting.correlation_heatmap(table)


def test_correlation_heatmap(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table(pd.DataFrame(data={"A": [1, 2, 3.5], "B": [2, 4, 7]}))
    plotting.correlation_heatmap(table)
