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
        Table(
            {
                "A": [
                    3.8,
                    1.8,
                    3.2,
                    2.2,
                    1.0,
                    2.4,
                    3.5,
                    3.9,
                    1.9,
                    4.0,
                    1.4,
                    4.2,
                    4.5,
                    4.5,
                    1.4,
                    2.5,
                    2.8,
                    2.8,
                    1.9,
                    4.3,
                ],
                "B": [
                    "a",
                    "b",
                    "b",
                    "c",
                    "d",
                    "f",
                    "a",
                    "f",
                    "e",
                    "a",
                    "b",
                    "b",
                    "k",
                    "j",
                    "b",
                    "i",
                    "h",
                    "g",
                    "g",
                    "a",
                ],
            }
        ),
    ],
    ids=["one column", "four columns", "two columns with compressed visualization"],
)
def test_should_match_snapshot(table: Table, snapshot_png_image: SnapshotAssertion) -> None:
    histograms = table.plot.histograms()
    assert histograms == snapshot_png_image


def test_should_fail_on_empty_table() -> None:
    with pytest.raises(ZeroDivisionError):
        Table().plot.histograms()
