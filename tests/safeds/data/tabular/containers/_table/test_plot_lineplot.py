import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_should_raise_UnknownColumnNameError() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.plot_lineplot("C", "A")


def test_should_match_snapshot() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    current = table.plot_lineplot("A", "B")
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_lineplot.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()
