def _convert_and_check_datetime_format(
    format: str,
    used_for_parsing: bool,
) -> str:
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
            # Find the closing curly brace
            closing_brace_index = format.find("}", index)
            if closing_brace_index == -1:
                raise ValueError(f"Unclosed template expression at index {index}.")

            expression = format[index + 1 : closing_brace_index]
            converted_format += _convert_and_check_template_expression(expression, used_for_parsing)
            index = closing_brace_index + 1
        # Regular characters
        else:
            converted_format += char
            index += 1

    return converted_format


def char_at(string: str, i: int) -> str | None:
    if i >= len(string):
        return None
    return string[i]


def _convert_and_check_template_expression(
    expression: str,
    used_for_parsing: bool,
) -> str:
    converted_expression = expression

    return converted_expression


# def _check_format_string(format_string: str) -> bool:
#     valid_format_codes = {
#         "F": "the standard",
#         "a": "abbreviated weekday name",
#         "A": "full weekday name",
#         "w": "weekday as a decimal number",
#         "d": "day of the month as a zero-padded decimal number",
#         "b": "abbreviated month name",
#         "B": "full month name",
#         "m": "month as a zero-padded decimal number",
#         "y": "year without century as a zero-padded decimal number",
#         "Y": "year with century as a decimal number",
#         "H": "hour (24-hour clock) as a zero-padded decimal number",
#         "I": "hour (12-hour clock) as a zero-padded decimal number",
#         "p": "locale's equivalent of either AM or PM",
#         "M": "minute as a zero-padded decimal number",
#         "S": "second as a zero-padded decimal number",
#         "f": "microsecond as a zero-padded decimal number",
#         "z": "UTC offset in the form ±HHMM[SS[.ffffff]]",
#         "Z": "time zone name",
#         "j": "day of the year as a zero-padded decimal number",
#         "U": "week number of the year (Sunday as the first day of the week)",
#         "W": "week number of the year (Monday as the first day of the week)",
#         "c": "locale's appropriate date and time representation",
#         "x": "locale's appropriate date representation",
#         "X": "locale's appropriate time representation",
#         "%": "a literal '%' character",
#     }
#
#     # Keep track of the positions in the string
#     i = 0
#     n = len(format_string)
#
#     # Iterate over each character in the format string
#     while i < n:
#         if format_string[i] == "%":
#             # Make sure there's at least one character following the '%'
#             if i + 1 < n:
#                 code = format_string[i + 1]
#                 # Check if the following character is a valid format code
#                 if code not in valid_format_codes:
#                     return False
#                 i += 2  # Skip ahead past the format code
#             else:
#                 # '%' is at the end of the string with no following format code
#                 return False
#         else:
#             i += 1  # Continue to the next character
#
#     return True
