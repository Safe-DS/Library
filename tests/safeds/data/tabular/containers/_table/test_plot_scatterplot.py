import _pytest
import matplotlib.pyplot as plt
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    "table",
    [
        Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]}),
    ],
    ids=["numerical"]
)
def test_should_plot_scatter_plot(table: Table, monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table.plot_scatterplot("A", "B")


def test_raise_error_if_column_name_unknown() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.plot_scatterplot("C", "A")
