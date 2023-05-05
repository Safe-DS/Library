from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table.from_dict({"A": [1, 2, 3.5], "B": [0.2, 4, 77]})
    current = table.plot_correlation_heatmap()
    legacy = Image.from_png_file(resolve_resource_path("./image/snapshot_heatmap.png"))
    assert legacy._image.tobytes() == current._image.tobytes()
