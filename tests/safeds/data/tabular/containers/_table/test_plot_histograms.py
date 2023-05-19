import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("table", "path"),
    [
        (Table({"A": [1, 2, 3]}), "./image/snapshot_histograms/one_column.png"),
        (
            Table({"A": [1, 2, 3], "B": ["A", "A", "Bla"], "C": [True, True, False], "D": [1.0, 2.1, 4.5]}),
            "./image/snapshot_histograms/four_columns.png",
        ),
    ],
    ids=["one column", "four columns"],
)
def test_should_match_snapshot(table: Table, path: str) -> None:
    current = table.plot_histograms()
    snapshot = Image.from_png_file(resolve_resource_path(path))

    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot._image.tobytes() == current._image.tobytes()
    assert assertion

def test_should_fail_on_empty_table() -> None:
    with pytest.raises(Exception):
        Table().plot_histograms()
