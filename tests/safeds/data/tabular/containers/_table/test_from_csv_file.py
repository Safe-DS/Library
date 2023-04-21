from pathlib import Path
import pytest
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    "path",
    ["table.csv", Path("table.csv")],
)
def test_from_csv_file_valid(path: str | Path) -> None:
    table = Table.from_csv_file(resolve_resource_path(path))
    assert table.get_column("A").get_value(0) == 1
    assert table.get_column("B").get_value(0) == 2


@pytest.mark.parametrize(
    "path",
    ["test_table_from_csv_file_invalid.csv", Path("test_table_from_csv_file_invalid.csv")],
)
def test_from_csv_file_invalid(path: str | Path) -> None:
    with pytest.raises(FileNotFoundError):
        Table.from_csv_file(resolve_resource_path(path))
