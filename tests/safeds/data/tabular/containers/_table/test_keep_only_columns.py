import pytest

from safeds.data.tabular.containers import Table, Column
from safeds.data.tabular.exceptions import UnknownColumnNameError


class TestKeepOnlyColumns:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table.from_columns([Column("A", [1]), Column("B", [2])]),
                [],
                Table.from_columns([]),
            ),
            (
                Table.from_columns([Column("A", [1]), Column("B", [2])]),
                ["A"],
                Table.from_columns([Column("A", [1])]),
            ),
            (
                Table.from_columns([Column("A", [1]), Column("B", [2])]),
                ["B"],
                Table.from_columns([Column("B", [2])]),
            ),
            (
                Table.from_columns([Column("A", [1]), Column("B", [2])]),
                ["A", "B"],
                Table.from_columns([Column("A", [1]), Column("B", [2])]),
            ),
            # Related to https://github.com/Safe-DS/Stdlib/issues/115
            (
                Table.from_columns([Column("A", [1]), Column("B", [2]), Column("C", [3])]),
                ["C", "A"],
                Table.from_columns([Column("C", [3]), Column("A", [1])]),
            ),
        ]
    )
    def test_should_keep_only_listed_columns(self, table: Table, column_names: list[str], expected: Table):
        transformed_table = table.keep_only_columns(column_names)
        assert transformed_table == expected

    def test_raise_if_column_does_no_exist(self):
        table = Table.from_columns([Column("A", [1]), Column("B", [2])])
        with pytest.raises(UnknownColumnNameError):
            table.keep_only_columns(["C"])
