from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import RealNumber, Schema


def test_remove_rows_with_outliers_no_outliers() -> None:
    table = Table.from_dict(
        {
            "col1": ["A", "B", "C"],
            "col2": [1.0, 2.0, 3.0],
            "col3": [2, 3, 1],
        },
    )
    names = table.column_names
    result = table.remove_rows_with_outliers()
    assert result.n_rows == 3
    assert result.n_columns == 3
    assert names == table.column_names


def test_remove_rows_with_outliers_with_outliers() -> None:
    input_ = Table.from_dict(
        {
            "col1": [
                "A",
                "B",
                "C",
                "outlier",
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
            ],
            "col2": [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
            "col3": [2, 3, 1, 1_000_000_000, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    )
    result = input_.remove_rows_with_outliers()

    expected = Table.from_dict(
        {
            "col1": ["A", "B", "C", "a", "a", "a", "a", "a", "a", "a", "a"],
            "col2": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
            "col3": [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    )

    assert result == expected


def test_remove_rows_with_outliers_no_rows() -> None:
    table = Table([], Schema({"col1": RealNumber()}))
    result = table.remove_rows_with_outliers()
    assert result.n_rows == 0
    assert result.n_columns == 1
