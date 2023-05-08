from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table({"A": [1, 2, 3.5], "B": [0.2, 4, 77]})
    current = table.plot_correlation_heatmap()
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_heatmap.png"))

    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot._image.tobytes() == current._image.tobytes()
    assert assertion
