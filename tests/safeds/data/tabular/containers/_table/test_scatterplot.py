import _pytest
import matplotlib.pyplot as plt
import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


def test_scatterplot(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    table.scatterplot("A", "B")


def test_scatterplot_wrong_column_name() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.scatterplot("C", "A")
