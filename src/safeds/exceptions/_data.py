from __future__ import annotations


class NonNumericColumnError(TypeError):
    """Exception raised for trying to do numerical operations on a non-numerical column."""

    def __init__(self, column_info: str, help_msg: str | None = None) -> None:
        line_break = "\n"
        super().__init__(
            "Tried to do a numerical operation on one or multiple non-numerical columns:"
            f" \n{column_info}{line_break + help_msg if help_msg is not None else ''}",
        )


class MissingValuesColumnError(Exception):
    """Exception raised for trying to do operations on columns containing missing values."""

    def __init__(self, column_info: str, help_msg: str | None = None) -> None:
        line_break = "\n"
        super().__init__(
            "Tried to do an operation on one or multiple columns containing missing values:"
            f" \n{column_info}{line_break + help_msg if help_msg is not None else ''}",
        )


class DuplicateIndexError(IndexError):
    """
    Exception raised for trying to add an element with an index that does already exist in the underlying data.

    Parameters
    ----------
    index:
        The wrongly added index.
    """

    def __init__(self, index: int):
        super().__init__(f"The index '{index}' is already in use.")


class OutputLengthMismatchError(Exception):
    """Exception raised when the lengths of the input and output container does not match."""

    def __init__(self, output_info: str):
        super().__init__(f"The length of the output container differs: \n{output_info}")


class ValueNotPresentWhenFittedError(Exception):
    """Exception raised when attempting to one-hot-encode a table containing values not present in the fitting phase."""

    def __init__(self, values: list[tuple[str, str]]) -> None:
        values_info = [f"{value} in column {column}" for value, column in values]
        line_break = "\n"
        super().__init__(
            "Value(s) not present in the table the transformer was fitted on:"
            f" {line_break}{line_break.join(values_info)}",
        )


class IllegalFormatError(Exception):
    """Exception raised when a format is not legal."""

    def __init__(self, formats: list[str] | str) -> None:
        super().__init__(f"This format is illegal. Use one of the following formats: {formats}")
