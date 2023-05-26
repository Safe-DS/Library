import pytest
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    current = table.plot_scatterplot("A", "B")
    snapshot = Image.from_png_file(resolve_resource_path("./image/snapshot_scatterplot.png"))

    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot._image.tobytes() == current._image.tobytes()
    assert assertion


@pytest.mark.parametrize(
    ("table", "col1", "col2", "error_message"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A", r"Could not find column\(s\) 'C'"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "B", "C", r"Could not find column\(s\) 'C'"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "D", r"Could not find column\(s\) 'C, D'"),
    ],
    ids=["First argument doesn't exist", "Second argument doesn't exist", "Both arguments do not exist"],
)
def test_should_raise_if_column_does_not_exist(table: Table, col1: str, col2: str, error_message: str) -> None:
    with pytest.raises(UnknownColumnNameError, match=error_message):
        table.plot_scatterplot(col1, col2)
