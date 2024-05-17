from random import randrange

from safeds.data.tabular.containers import Table


def create_synthetic_table(
    row_count: int,
    column_count: int,
    *,
    min_value: int = 0,
    max_value: int = 1000,
) -> Table:
    """
    Create a synthetic Table with random numerical data.

    Parameters
    ----------
    row_count:
        Number of rows in the Table.
    column_count:
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
    return Table(
        {
            f"column_{i}": [randrange(min_value, max_value) for _ in range(row_count)]
            for i in range(column_count)
        },
    )
