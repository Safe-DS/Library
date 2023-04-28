import _pytest
import matplotlib.pyplot as plt
import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_plot_scatterplot(monkeypatch: _pytest.monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    table.plot_scatterplot("A", "B")


def test_plot_scatterplot_wrong_column_name() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.plot_scatterplot("C", "A")


def test_plot_scatterplot_legacy_check() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    current = table.plot_scatterplot("A", "B")
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_scatterplot.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()
