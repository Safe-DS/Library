import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]})
    current = table.plot_scatterplot("A", "B")
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_scatterplot.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()


@pytest.mark.parametrize(
    ("table", "col1", "col2"),
    [
        (Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A"),
        (Table.from_dict({"A": [1, 2, 3], "B": [2, 4, 7]}), "B", "C"),
    ],
    ids=["First argument doesn't exist", "Second argument doesn't exist"],
)
def test_should_raise_if_column_does_not_exist(table:Table, col1:str, col2:str) -> None:
    with pytest.raises(UnknownColumnNameError):
        table.plot_scatterplot(col1, col2)
