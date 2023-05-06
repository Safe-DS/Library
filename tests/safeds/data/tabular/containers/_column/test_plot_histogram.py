from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_should_match_snapshot_numeric() -> None:
    table = Table({"A": [1, 2, 3]})
    current = table.get_column("A").plot_histogram()
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_histogram_numeric.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()


def test_should_match_snapshot_str() -> None:
    table = Table({"A": ["A", "B", "Apple"]})
    current = table.get_column("A").plot_histogram()
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_histogram_str.png"))
    assert snapshot._image.tobytes() == current._image.tobytes()
