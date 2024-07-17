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


class DuplicateColumnError(ValueError):
    """
    Exception raised for trying to modify a table resulting in a duplicate column name.

    Parameters
    ----------
    column_name:
        The name of the column that resulted in a duplicate.
    """

    def __init__(self, column_name: str):
        super().__init__(f"Column '{column_name}' already exists.")


class IndexOutOfBoundsError(IndexError):
    """
    Exception raised for trying to access an element by an index that does not exist in the underlying data.

    Parameters
    ----------
    index:
        The wrongly used index.
    """

    def __init__(self, index: int | list[int] | slice):
        if isinstance(index, list):
            if len(index) == 1:
                index = index[0]
            else:
                super().__init__(f"There are no elements at indices {index}.")
                return
        if isinstance(index, int):
            super().__init__(f"There is no element at index '{index}'.")
        else:
            super().__init__(f"There is no element in the range [{index.start}, {index.stop}].")


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


class ColumnSizeError(Exception):
    """
    Exception raised for trying to use a column of unsupported size.

    Parameters
    ----------
    expected_size:
        The expected size of the column as an expression (e.g. 2, >0, !=0).
    actual_size:
        The actual size of the column as an expression (e.g. 2, >0, !=0).
    """

    def __init__(self, expected_size: str, actual_size: str):
        super().__init__(f"Expected a column of size {expected_size} but got column of size {actual_size}.")


class ColumnLengthMismatchError(ValueError):
    """Exception raised when the lengths of two or more columns do not match."""

    def __init__(self, column_info: str):
        super().__init__(f"The length of at least one column differs: \n{column_info}")


class OutputLengthMismatchError(Exception):
    """Exception raised when the lengths of the input and output container does not match."""

    def __init__(self, output_info: str):
        super().__init__(f"The length of the output container differs: \n{output_info}")


class TransformerNotFittedError(Exception):
    """Raised when a transformer is used before fitting it."""

    def __init__(self) -> None:
        super().__init__("The transformer has not been fitted yet.")


class TransformerNotInvertibleError(Exception):
    """Raised when a function tries to invert a non-invertible transformer."""

    def __init__(self, transformer_type: str) -> None:
        super().__init__(f"{transformer_type} is not invertible.")


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
