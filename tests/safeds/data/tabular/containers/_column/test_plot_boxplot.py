import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import NonNumericColumnError

from tests.helpers import resolve_resource_path


def test_plot_boxplot_complex() -> None:
    with pytest.raises(NotImplementedError):  # noqa: PT012
        table = Table.from_dict({"A": [1, 2, complex(1, -2)]})
        table.get_column("A").plot_boxplot()


def test_plot_boxplot_non_numeric() -> None:
    table = Table.from_dict({"A": [1, 2, "A"]})
    with pytest.raises(NonNumericColumnError):
        table.get_column("A").plot_boxplot()


def test_should_match_snapshot() -> None:
    table = Table.from_dict({"A": [1, 2, 3]})
    table.get_column("A").plot_boxplot()
    current = table.get_column("A").plot_boxplot()
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_boxplot.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()
