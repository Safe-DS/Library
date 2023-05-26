from pathlib import Path

from _pytest.python_api import approx

from src.safeds.data.tabular.containers import Table

_resources_root = Path(__file__).parent / ".." / "resources"


def resolve_resource_path(resource_path: str | Path) -> str:
    """
    Resolve a path relative to the `resources` directory to an absolute path.

    Parameters
    ----------
    resource_path : str | Path
        The path to the resource relative to the `resources` directory.

    Returns
    -------
    absolute_path : str
        The absolute path to the resource.
    """
    return str(_resources_root / resource_path)


def check_that_tables_are_close(table1: Table, table2: Table) -> None:
    """
    Check that two tables are almost equal.

    Parameters
    ----------
    table1: Table
        The first table.
    table2: Table
        The table to compare the first table to.
    """
    assert table1.schema == table2.schema
    for column_name in table1.column_names:
        assert table1.get_column(column_name).type == table2.get_column(column_name).type
        assert table1.get_column(column_name).type.is_numeric()
        assert table2.get_column(column_name).type.is_numeric()
        for i in range(table1.number_of_rows):
            entry_1 = table1.get_column(column_name).get_value(i)
            entry_2 = table2.get_column(column_name).get_value(i)
            assert entry_1 == approx(entry_2)
