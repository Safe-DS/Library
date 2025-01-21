import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "prefix", "expected"),
    [
        ("", " ", ""),
        ("~ a ~", "", "~ a ~"),
        ("~ a ~", "~ ", "a ~"),
        ("~ a ~", " ~", "~ a ~"),
        (None, " ", None),
        ("~ a ~", None, None),
        (None, None, None),
    ],
    ids=[
        "empty",
        "empty prefix",
        "non-empty (has prefix)",
        "non-empty (does not have prefix)",
        "None as string",
        "None as prefix",
        "None as both",
    ],
)
def test_should_remove_prefix(value: str | None, prefix: str | None, expected: bool | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.remove_prefix(prefix),
        expected,
        type_if_none=ColumnType.string(),
    )
