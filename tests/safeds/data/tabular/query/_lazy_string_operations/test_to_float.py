import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", None),
        ("abc", None),
        ("1", 1.0),
        ("1.5", 1.5),
        ("-1.5", -1.5),
        ("1e3", 1000),
        (None, None),
    ],
    ids=[
        "empty",
        "invalid",
        "int",
        "positive float",
        "negative float",
        "exponential",
        "None",
    ],
)
def test_should_convert_string_to_float(value: str | None, expected: float | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_float(),
        expected,
        type_if_none=ColumnType.string(),
    )
