import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("table", "path"),
    [
        (Table({"A": [1, 2, 3.5], "B": [0.2, 4, 77]}), "./image/snapshot_heatmap.png"),
        (Table(), "./image/snapshot_empty_heatmap.png"),
    ],
    ids=["normal", "empty"],
)
def test_should_match_snapshot(table: Table, path: str) -> None:
    if table.number_of_rows == 0:
        with pytest.warns(
            UserWarning,
            match=r"An empty table has been used. A correlation heatmap on an empty table will show nothing.",
        ):
            current = table.plot_correlation_heatmap()
    else:
        current = table.plot_correlation_heatmap()
    snapshot = Image.from_png_file(resolve_resource_path(path))

    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot == current
    assert assertion
