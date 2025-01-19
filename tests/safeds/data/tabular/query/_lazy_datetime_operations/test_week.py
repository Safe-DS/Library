from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # datetime - first day is Monday
        (datetime(2024, 1, 1, tzinfo=UTC), 1),
        # datetime - first day is Tuesday
        (datetime(2030, 1, 1, tzinfo=UTC), 1),
        # datetime - first day is Wednesday
        (datetime(2025, 1, 1, tzinfo=UTC), 1),
        # datetime - first day is Thursday
        (datetime(2026, 1, 1, tzinfo=UTC), 1),
        # datetime - first day is Friday
        (datetime(2027, 1, 1, tzinfo=UTC), 53),
        # datetime - first day is Saturday
        (datetime(2028, 1, 1, tzinfo=UTC), 52),
        # datetime - first day is Sunday
        (datetime(2023, 1, 1, tzinfo=UTC), 52),
        # datetime - last day is Monday
        (datetime(2029, 12, 31, tzinfo=UTC), 1),
        # datetime - last day is Tuesday
        (datetime(2024, 12, 31, tzinfo=UTC), 1),
        # datetime - last day is Wednesday
        (datetime(2025, 12, 31, tzinfo=UTC), 1),
        # datetime - last day is Thursday
        (datetime(2026, 12, 31, tzinfo=UTC), 53),
        # datetime - last day is Friday
        (datetime(2027, 12, 31, tzinfo=UTC), 52),
        # datetime - last day is Saturday
        (datetime(2022, 12, 31, tzinfo=UTC), 52),
        # datetime - last day is Sunday
        (datetime(2023, 12, 31, tzinfo=UTC), 52),
        # date - first day is Monday
        (date(2024, 1, 1), 1),
        # date - first day is Tuesday
        (date(2030, 1, 1), 1),
        # date - first day is Wednesday
        (date(2025, 1, 1), 1),
        # date - first day is Thursday
        (date(2026, 1, 1), 1),
        # date - first day is Friday
        (date(2027, 1, 1), 53),
        # date - first day is Saturday
        (date(2028, 1, 1), 52),
        # date - first day is Sunday
        (date(2023, 1, 1), 52),
        # date - last day is Monday
        (date(2029, 12, 31), 1),
        # date - last day is Tuesday
        (date(2024, 12, 31), 1),
        # date - last day is Wednesday
        (date(2025, 12, 31), 1),
        # date - last day is Thursday
        (date(2026, 12, 31), 53),
        # date - last day is Friday
        (date(2027, 12, 31), 52),
        # date - last day is Saturday
        (date(2022, 12, 31), 52),
        # date - last day is Sunday
        (date(2023, 12, 31), 52),
        # None
        (None, None),
    ],
    ids=[
        # datetime
        "datetime - first day is Monday",
        "datetime - first day is Tuesday",
        "datetime - first day is Wednesday",
        "datetime - first day is Thursday",
        "datetime - first day is Friday",
        "datetime - first day is Saturday",
        "datetime - first day is Sunday",
        "datetime - last day is Monday",
        "datetime - last day is Tuesday",
        "datetime - last day is Wednesday",
        "datetime - last day is Thursday",
        "datetime - last day is Friday",
        "datetime - last day is Saturday",
        "datetime - last day is Sunday",
        # date
        "date - first day is Monday",
        "date - first day is Tuesday",
        "date - first day is Wednesday",
        "date - first day is Thursday",
        "date - first day is Friday",
        "date - first day is Saturday",
        "date - first day is Sunday",
        "date - last day is Monday",
        "date - last day is Tuesday",
        "date - last day is Wednesday",
        "date - last day is Thursday",
        "date - last day is Friday",
        "date - last day is Saturday",
        "date - last day is Sunday",
        # None
        "None",
    ],
)
def test_should_extract_week(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.week(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
