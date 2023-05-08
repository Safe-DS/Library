from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table({"A": [1, 2, 3], "B": ["A", "A", "Apple"]})
    current = table.plot_histograms()
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_histograms.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()
