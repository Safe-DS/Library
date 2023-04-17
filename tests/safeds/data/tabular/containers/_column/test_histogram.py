import _pytest
import matplotlib.pyplot as plt
import pandas as pd
from safeds.data.tabular.containers import Table


def test_histogram(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3]})
    table.get_column("A").histogram()
