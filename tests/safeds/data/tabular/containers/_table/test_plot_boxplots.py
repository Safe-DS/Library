import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    ("table", "path"),
    [
        (Table({"A": [1, 2, 3]}), "./image/snapshot_boxplots/one_column.png"),
        (
            Table({"A": [1, 2, 3], "B": ["A", "A", "Bla"], "C": [True, True, False], "D": [1.0, 2.1, 4.5]}),
            "./image/snapshot_boxplots/four_columns_some_non_numeric.png",
        ),
        (
            Table({"A": [1, 2, 3], "B": [1.0, 2.1, 4.5], "C": [1, 2, 3], "D": [1.0, 2.1, 4.5]}),
            "./image/snapshot_boxplots/four_columns_all_numeric.png",
        ),
    ],
    ids=["one column", "four columns (some non-numeric)", "four columns (all numeric)"],
)
def test_should_match_snapshot(table: Table, path: str) -> None:
    current = table.plot_boxplots()
    current.to_png_file(resolve_resource_path(path))
    snapshot = Image.from_png_file(resolve_resource_path(path))

    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot._image.tobytes() == current._image.tobytes()
    assert assertion

def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = Table.from_dict({"A": ["1", "2", "3.5"], "B": ["0.2", "4", "77"]})
    with pytest.raises(NonNumericColumnError):
        table.plot_boxplots()
