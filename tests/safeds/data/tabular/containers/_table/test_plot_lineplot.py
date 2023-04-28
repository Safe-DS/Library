import _pytest
import matplotlib.pyplot as plt
import pytest

from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError
from tests.helpers import resolve_resource_path


def test_plot_lineplot(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    table.plot_lineplot("A", "B")


def test_plot_lineplot_wrong_column_name() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.plot_lineplot("C", "A")

def test_plot_lineplot_legacy_check() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    #table.plot_lineplot("A", "B").to_png_file(resolve_resource_path("./image/snapshot_lineplot.png"))
    current = table.plot_lineplot("A", "B")
    legacy = Image.from_png_file(resolve_resource_path("./image/snapshot_lineplot.png"))
    assert legacy._image.tobytes() == current._image.tobytes()
