import pytest
from safeds.data.tabular.containers import Table
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    "table",
    [
        Table({"A": [1, 2, 3]}),
        Table(
            {
                "A": [1, 2, 3, 3, 2, 4, 2],
                "B": ["a", "b", "b", "b", "b", "b", "a"],
                "C": [True, True, False, True, False, None, True],
                "D": [1.0, 2.1, 2.1, 2.1, 2.1, 3.0, 3.0],
            },
        ),
    ],
    ids=["one column", "four columns"],
)
def test_should_match_snapshot(table: Table, snapshot_png_image: SnapshotAssertion) -> None:
    histograms = table.plot_histograms()
    assert histograms == snapshot_png_image


def test_should_fail_on_empty_table() -> None:
    with pytest.raises(ZeroDivisionError):
        Table().plot_histograms()
