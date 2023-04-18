import pytest
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


@pytest.mark.parametrize(
    "path",
    [
        "test_table_duplicate_rows_duplicates.csv",
        "test_table_duplicate_rows_no_duplicates.csv",
    ],
)
def test_remove_duplicate_rows(path: str) -> None:
    expected_table = Table.from_dict({"A": [1, 4], "B": [2, 5]})
    table = Table.from_csv_file(resolve_resource_path(path))
    result_table = table.remove_duplicate_rows()
    assert expected_table == result_table
