import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import NonNumericColumnError

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table.from_dict({"A": [1, 2, 3]})
    current = table.get_column("A").plot_boxplot()
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_boxplot.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = Table.from_dict({"A": [1, 2, "A"]})
    with pytest.raises(NonNumericColumnError):
        table.get_column("A").plot_boxplot()
