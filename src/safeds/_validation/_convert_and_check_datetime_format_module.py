from __future__ import annotations

from typing import Literal

from safeds._utils import _get_similar_strings

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
    "M-abbr": "b",
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
    "DOW-abbr": "a",
    "DOY": "j",
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
    format: str,
    type_: Literal["datetime", "date", "time"],
    used_for_parsing: bool,
) -> str:
    replacements = _get_replacements(type_, used_for_parsing)
    converted_format = ""
    index = 0

    while index < len(format):
        char = char_at(format, index)

        # Escaped characters
        if char == "\\" and char_at(format, index + 1) == "\\":
            converted_format += "\\"
            index += 2
        if char == "\\" and char_at(format, index + 1) == "{":
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
            end_index = format.find("}", index)
            if end_index == -1:
                raise ValueError(f"Unclosed template expression at index {index}.")

            expression = format[index + 1 : end_index]
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

    # Unknown template expression
    message = _build_error_message(expression, type_, list(replacements.keys()))
    raise ValueError(message)


def _build_error_message(
    expression: str,
    type_: str,
    valid_expressions: list[str],
) -> str:
    result = f"Invalid template expression '{expression}' for type {type_}."

    similar_expressions = _get_similar_strings(expression, valid_expressions)
    if similar_expressions:
        result += f" Did you mean one of {similar_expressions}?"

    return result
