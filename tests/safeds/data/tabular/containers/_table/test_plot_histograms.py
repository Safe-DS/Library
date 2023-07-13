import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("table", "path"),
    [
        (Table({"A": [1, 2, 3]}), "./image/snapshot_histograms/one_column.png"),
        (
            Table(
                {
                    "A": [1, 2, 3, 3, 2, 4, 2],
                    "B": ["a", "b", "b", "b", "b", "b", "a"],
                    "C": [True, True, False, True, False, None, True],
                    "D": [1.0, 2.1, 2.1, 2.1, 2.1, 3.0, 3.0],
                },
            ),
            "./image/snapshot_histograms/four_columns.png",
        ),
    ],
    ids=["one column", "four columns"],
)
def test_should_match_snapshot(table: Table, path: str) -> None:
    current = table.plot_histograms()
    snapshot = Image.from_png_file(resolve_resource_path(path))
    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot == current
    assert assertion


def test_should_fail_on_empty_table() -> None:
    with pytest.raises(ZeroDivisionError):
        Table().plot_histograms()
