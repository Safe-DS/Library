import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "old", "new", "expected"),
    [
        # all empty
        ("", "", "", ""),
        # empty value
        ("", "a", "z", ""),
        # empty old
        ("abc", "", "z", "zazbzcz"),
        # empty new
        ("abc", "a", "", "bc"),
        # no matches
        ("abc", "d", "z", "abc"),
        # one match
        ("abc", "a", "z", "zbc"),
        # many matches
        ("abcabc", "a", "z", "zbczbc"),
        # full match
        ("abc", "abc", "z", "z"),
        # None value
        (None, "a", "z", None),
        # None old
        pytest.param("abc", None, "z", None, marks=pytest.mark.xfail(reason="Not supported by polars.")),
        # None new
        pytest.param("abc", "a", None, None, marks=pytest.mark.xfail(reason="Not supported by polars.")),
    ],
    ids=[
        "all empty",
        "empty value",
        "empty old",
        "empty new",
        "no matches",
        "one match",
        "many matches",
        "full match",
        "None value",
        "None old",
        "None new",
    ],
)
class TestShouldReplaceAllOccurrencesOfOldWithNew:
    def test_plain_arguments(self, value: str | None, old: str | None, new: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.replace_all(old, new),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(
        self,
        value: str | None,
        old: str | None,
        new: str | None,
        expected: bool | None,
    ) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.replace_all(
                Cell.constant(old),
                Cell.constant(new),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )
