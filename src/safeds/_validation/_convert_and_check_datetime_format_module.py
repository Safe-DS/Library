from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from safeds._utils import _get_similar_strings

if TYPE_CHECKING:
    from collections.abc import Iterable

_DATE_REPLACEMENTS = {
    # Year
    "Y": "Y",
    "_Y": "_Y",
    "^Y": "-Y",
    "Y99": "y",
    "_Y99": "_y",
    "^Y99": "-y",
    # Month
    "M": "m",
    "_M": "_m",
    "^M": "-m",
    "M-full": "B",
    "M-short": "b",
    # Week
    "W": "V",
    "_W": "_V",
    "^W": "-V",
    # Day
    "D": "d",
    "_D": "_d",
    "^D": "-d",
    "DOW": "u",
    "DOW-full": "A",
    "DOW-short": "a",
    "DOY": "j",
    "_DOY": "_j",
    "^DOY": "-j",
}

_TIME_REPLACEMENTS = {
    # Hour
    "h": "H",
    "_h": "_H",
    "^h": "-H",
    "h12": "I",
    "_h12": "_I",
    "^h12": "-I",
    # Minute
    "m": "M",
    "_m": "_M",
    "^m": "-M",
    # Second
    "s": "S",
    "_s": "_S",
    "^s": "-S",
    # Fractional seconds
    ".f": ".f",
    "ms": "3f",
    "us": "6f",
    "ns": "9f",
    # AM/PM
    "AM/PM": "p",
    "am/pm": "P",
}

_DATETIME_REPLACEMENTS = {
    # Date and time replacements are also valid for datetime
    **_DATE_REPLACEMENTS,
    **_TIME_REPLACEMENTS,
    # Timezone
    "z": "z",
    ":z": ":z",
    # UNIX timestamp
    "u": "s",
}

_DATETIME_REPLACEMENTS_WHEN_PARSING = {
    **_DATETIME_REPLACEMENTS,
    # Allow omission of minutes for the timezone offset
    "z": "#z",
    ":z": "#z",
}


def _convert_and_check_datetime_format(
    format_: str,
    type_: Literal["datetime", "date", "time"],
    used_for_parsing: bool,
) -> str:
    """
    Convert our datetime format string to a format string understood by polars and check for errors.

    Parameters
    ----------
    format_:
        The datetime format to convert.
    type_:
        Whether format is for a datetime, date, or time.
    used_for_parsing:
        Whether the format is used for parsing.

    Returns
    -------
    converted_format:
        The converted datetime format.

    Raises
    ------
    ValueError
        If the format is invalid.
    """
    replacements = _get_replacements(type_, used_for_parsing)
    converted_format = ""
    index = 0

    while index < len(format_):
        char = format_[index]

        # Escaped characters
        if char == "\\" and char_at(format_, index + 1) == "\\":
            converted_format += "\\"
            index += 2
        elif char == "\\" and char_at(format_, index + 1) == "{":
            converted_format += "{"
            index += 2
        # Characters that need to be escaped for rust's chrono crate
        elif char == "\n":
            converted_format += "%n"
            index += 1
        elif char == "\t":
            converted_format += "%t"
            index += 1
        elif char == "%":
            converted_format += "%%"
            index += 1
        # Template expression
        elif char == "{":
            end_index = format_.find("}", index)
            if end_index == -1:
                message = f"Unclosed specifier at index {index}."
                raise ValueError(message)

            expression = format_[index + 1 : end_index]
            converted_format += _convert_and_check_template_expression(expression, type_, replacements)
            index = end_index + 1
        # Regular characters
        else:
            converted_format += char
            index += 1

    return converted_format


def _get_replacements(
    type_: Literal["datetime", "date", "time"],
    used_for_parsing: bool,
) -> dict[str, str]:
    if type_ == "datetime":
        return _DATETIME_REPLACEMENTS_WHEN_PARSING if used_for_parsing else _DATETIME_REPLACEMENTS
    elif type_ == "date":
        return _DATE_REPLACEMENTS
    else:
        return _TIME_REPLACEMENTS


def char_at(string: str, i: int) -> str | None:
    if i >= len(string):
        return None
    return string[i]


def _convert_and_check_template_expression(
    expression: str,
    type_: str,
    replacements: dict[str, str],
) -> str:
    if expression in replacements:
        return "%" + replacements[expression]

    # Unknown specifier
    message = _build_error_message(expression, type_, replacements.keys())
    raise ValueError(message)


def _build_error_message(
    expression: str,
    type_: str,
    valid_expressions: Iterable[str],
) -> str:
    result = f"Invalid specifier '{expression}' for type {type_}."

    similar_expressions = _get_similar_strings(expression, valid_expressions)
    if similar_expressions:  # pragma: no cover
        result += f" Did you mean one of {similar_expressions}?"

    return result
