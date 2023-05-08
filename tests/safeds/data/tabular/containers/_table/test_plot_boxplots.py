import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    "table",
    [
        Table.from_dict({"A": [1, 2, 3.5], "B": [0.2, 4, 77]}),
        Table.from_dict({"A": [1, 2, 3.5], "A_2": ["A", "B", "C"], "B": [0.2, 4, 77]}),
    ],
    ids=["only non-numerical columns", "remove all non-numerical columns"],
)
def test_should_match_snapshot(table: Table) -> None:
    current = table.plot_boxplots()
    legacy = Image.from_png_file(resolve_resource_path("./image/snapshot_boxplots.png"))
    assert legacy._image.tobytes() == current._image.tobytes()


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = Table.from_dict({"A": ["1", "2", "3.5"], "B": ["0.2", "4", "77"]})
    with pytest.raises(NonNumericColumnError):
        table.plot_boxplots()
