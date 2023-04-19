import _pytest
import matplotlib.pyplot as plt
from safeds.data.tabular.containers import Table


def test_plot_histogram(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3]})
    table.get_column("A").plot_histogram()
