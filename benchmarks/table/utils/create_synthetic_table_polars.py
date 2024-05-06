from random import randrange

from safeds.data.tabular.containers import ExperimentalPolarsTable


def create_synthetic_table_polars(
    number_of_rows: int,
    number_of_columns: int,
    *,
    min_value: int = 0,
    max_value: int = 1000,
) -> ExperimentalPolarsTable:
    """Create a synthetic Table with random numerical data.

    Parameters
    ----------
    number_of_rows:
        Number of rows in the Table.
    number_of_columns:
        Number of columns in the Table.
    min_value:
        Minimum value of the random data.
    max_value:
        Maximum value of the random data.

    Returns
    -------
    Table
        A Table with random numerical data.
    """
    return ExperimentalPolarsTable(
        {
            f"column_{i}": [randrange(min_value, max_value) for _ in range(number_of_rows)]
            for i in range(number_of_columns)
        }
    )
