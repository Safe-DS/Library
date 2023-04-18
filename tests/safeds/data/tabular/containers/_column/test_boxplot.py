import _pytest
import matplotlib.pyplot as plt
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import NonNumericColumnError


def test_boxplot_complex() -> None:
    with pytest.raises(NotImplementedError):  # noqa: PT012
        table = Table.from_dict({"A": [1, 2, complex(1, -2)]})
        table.get_column("A").boxplot()


def test_boxplot_non_numeric() -> None:
    table = Table.from_dict({"A": [1, 2, "A"]})
    with pytest.raises(NonNumericColumnError):
        table.get_column("A").boxplot()


def test_boxplot_float(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3.5]})
    table.get_column("A").boxplot()


def test_boxplot_int(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3]})
    table.get_column("A").boxplot()
