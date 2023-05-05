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
def test_should_plot_lineplot(table: Table, monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table.plot_lineplot("A", "B")


def test_should_raise_unknown_column_name_error() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.plot_lineplot("C", "A")
