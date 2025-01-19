import zoneinfo

from safeds._utils import _get_similar_strings

_VALID_TZ_IDENTIFIERS = zoneinfo.available_timezones()


def _check_time_zone(time_zone: str | None) -> None:
    """
    Check if the time zone is valid.

    Parameters
    ----------
    time_zone:
        The time zone to check.

    Raises
    ------
    ValueError
        If the time zone is invalid.
    """
    if time_zone is not None and time_zone not in _VALID_TZ_IDENTIFIERS:
        message = _build_error_message(time_zone)
        raise ValueError(message)


def _build_error_message(time_zone: str) -> str:
    result = f"Invalid time zone '{time_zone}'."

    similar_time_zones = _get_similar_strings(time_zone, _VALID_TZ_IDENTIFIERS)
    if similar_time_zones:  # pragma: no cover
        result += f" Did you mean one of {similar_time_zones}?"

    return result
