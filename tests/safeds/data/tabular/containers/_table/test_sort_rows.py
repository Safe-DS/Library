from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Row, Table


class TestSortRows:
    @pytest.mark.parametrize(
        ("table", "comparator", "expected"),
        [
            # Check that it works with an empty table when https://github.com/Safe-DS/Stdlib/issues/75 is fixed.
            (
                Table.from_dict({"col1": [3, 2, 1]}),
                lambda row1, row2: row1["col1"] - row2["col1"],
                Table.from_dict({"col1": [1, 2, 3]}),
            ),
        ],
    )
    def test_should_return_sorted_table(
        self,
        table: Table,
        comparator: Callable[[Row, Row], int],
        expected: Table,
    ) -> None:
        assert table.sort_rows(comparator) == expected

    @pytest.mark.parametrize(
        ("table", "comparator", "expected"),
        [
            # Activate when https://github.com/Safe-DS/Stdlib/issues/75 is fixed.
            # ),
            (
                Table.from_dict({"col1": [3, 2, 1]}),
                lambda row1, row2: row1["col1"] - row2["col1"],
                Table.from_dict({"col1": [3, 2, 1]}),
            ),
        ],
    )
    def test_should_not_modify_original_table(
        self,
        table: Table,
        comparator: Callable[[Row, Row], int],
        expected: Table,
    ) -> None:
        table.sort_rows(comparator)
        assert table == expected
