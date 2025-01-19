import pytest

from safeds.data.tabular.typing import ColumnType


def test_should_raise_if_time_zone_is_invalid():
    with pytest.raises(ValueError, match="Invalid time zone"):
        ColumnType.datetime(time_zone="invalid")
