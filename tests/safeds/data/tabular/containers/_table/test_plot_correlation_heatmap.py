import _pytest
import matplotlib.pyplot as plt
import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table.from_dict({"A": [1, 2, "A"], "B": [1, 2, 3]}),
        Table.from_dict({"A": [1, 2, 3.5], "B": [2, 4, 7]}),
    ],
    ids=["non numerical", "numerical"]
)
def test_should_plot_correlation_heatmap(table: Table, monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table.plot_correlation_heatmap()
