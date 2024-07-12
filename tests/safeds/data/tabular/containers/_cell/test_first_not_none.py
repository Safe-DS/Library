from datetime import date, time

import pytest
from safeds.data.tabular.containers._cell import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell


class TestFirstNotNone:
    def test_should_return_none(self) -> None:
        to_eval: list[Cell] = [_LazyCell(None) for i in range(5)]
        res = Cell.first_not_none(to_eval)
        assert res.eq(_LazyCell(None))

    @pytest.mark.parametrize(
        ("list_of_cells", "expected"),
        [
            ([_LazyCell(None), _LazyCell(1), _LazyCell(None), _LazyCell(4)], _LazyCell(1)),
            ([_LazyCell(i) for i in range(5)], _LazyCell(1)),
            (
                [_LazyCell(None), _LazyCell(None), _LazyCell("Hello, World!"), _LazyCell("Not returned")],
                _LazyCell("Hello, World!"),
            ),
            ([_LazyCell(i) for i in ["a", "b", "c", "d"]], _LazyCell("a")),
            ([_LazyCell(i) for i in [None, time(0, 0, 0, 0), None, time(1, 1, 1, 1)]], _LazyCell(time(0, 0, 0, 0))),
            (
                [_LazyCell(i) for i in [time(0, 0, 0, 0), time(1, 1, 1, 1), time(2, 2, 2, 2), time(3, 3, 3, 3)]],
                _LazyCell(time(0, 0, 0, 0)),
            ),
            ([_LazyCell(i) for i in [None, date(2000, 1, 1), date(1098, 3, 4), None]], _LazyCell(date(2000, 1, 1))),
            ([_LazyCell(date(2000, 3, i)) for i in range(1, 5)], _LazyCell(date(2000, 3, 1))),
            ([_LazyCell(i) for i in [None, "a", 1, time(0, 0, 0, 0)]], _LazyCell("a")),
            ([_LazyCell(i) for i in [time(1, 1, 1, 1), 0, "c", date(2020, 1, 7)]], _LazyCell(time(1, 1, 1, 1))),
        ],
        ids=[
            "numeric_with_null",
            "numeric_no_null",
            "strings_with_null",
            "strings_no_null",
            "times_with_null",
            "times_no_null",
            "dates_with_null",
            "dates_no_null",
            "mixed_with_null",
            "mixed_no_null",
        ],
    )
    def test_should_not_return_none(self, list_of_cells: list[Cell], expected: Cell) -> None:
        res = Cell.first_not_none(list_of_cells)
        assert res.eq(expected)

    def test_given_empty_list(self) -> None:
        to_eval: list[Cell] = []
        res = Cell.first_not_none(to_eval)
        assert res.eq(_LazyCell(None))
