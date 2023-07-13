from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class UnknownColumnNameError(KeyError):
    """
    Exception raised for trying to access an invalid column name.

    Parameters
    ----------
    column_names : list[str]
        The name of the column that was tried to be accessed.
    """

    def __init__(self, column_names: list[str], similar_columns: list[str] | None = None):
        class _UnknownColumnNameErrorMessage(
            str,
        ):  # This class is necessary for the newline character in a KeyError exception. See https://stackoverflow.com/a/70114007
            def __repr__(self) -> str:
                return str(self)

        error_message = f"Could not find column(s) '{', '.join(column_names)}'."

        if similar_columns is not None and len(similar_columns) > 0:
            error_message += f"\nDid you mean '{similar_columns}'?"

        super().__init__(_UnknownColumnNameErrorMessage(error_message))


class NonNumericColumnError(Exception):
    """Exception raised for trying to do numerical operations on a non-numerical column."""

    def __init__(self, column_info: str, help_msg: str | None = None) -> None:
        line_break = "\n"
        super().__init__(
            (
                "Tried to do a numerical operation on one or multiple non-numerical columns:"
                f" \n{column_info}{line_break + help_msg if help_msg is not None else ''}"
            ),
        )


class MissingValuesColumnError(Exception):
    """Exception raised for trying to do operations on columns containing missing values."""

    def __init__(self, column_info: str, help_msg: str | None = None) -> None:
        line_break = "\n"
        super().__init__(
            (
                "Tried to do an operation on one or multiple columns containing missing values:"
                f" \n{column_info}{line_break + help_msg if help_msg is not None else ''}"
            ),
        )


class DuplicateColumnNameError(Exception):
    """
    Exception raised for trying to modify a table resulting in a duplicate column name.

    Parameters
    ----------
    column_name : str
        The name of the column that resulted in a duplicate.
    """

    def __init__(self, column_name: str):
        super().__init__(f"Column '{column_name}' already exists.")


class IndexOutOfBoundsError(IndexError):
    """
    Exception raised for trying to access an element by an index that does not exist in the underlying data.

    Parameters
    ----------
    index : int | slice
        The wrongly used index.
    """

    def __init__(self, index: int | slice):
        if isinstance(index, int):
            super().__init__(f"There is no element at index '{index}'.")
        else:
            super().__init__(f"There is no element in the range [{index.start}, {index.stop}]")


class ColumnSizeError(Exception):
    """
    Exception raised for trying to use a column of unsupported size.

    Parameters
    ----------
    expected_size : str
        The expected size of the column as an expression (e.g. 2, >0, !=0).
    actual_size : str
        The actual size of the column as an expression (e.g. 2, >0, !=0).
    """

    def __init__(self, expected_size: str, actual_size: str):
        super().__init__(f"Expected a column of size {expected_size} but got column of size {actual_size}.")


class ColumnLengthMismatchError(Exception):
    """Exception raised when the lengths of two or more columns do not match."""

    def __init__(self, column_info: str):
        super().__init__(f"The length of at least one column differs: \n{column_info}")


class TransformerNotFittedError(Exception):
    """Raised when a transformer is used before fitting it."""

    def __init__(self) -> None:
        super().__init__("The transformer has not been fitted yet.")


class ValueNotPresentWhenFittedError(Exception):
    """Exception raised when attempting to one-hot-encode a table containing values not present in the fitting phase."""

    def __init__(self, values: list[tuple[str, str]]) -> None:
        values_info = [f"{value} in column {column}" for value, column in values]
        line_break = "\n"
        super().__init__(
            (
                "Value(s) not present in the table the transformer was fitted on:"
                f" {line_break}{line_break.join(values_info)}"
            ),
        )


class WrongFileExtensionError(Exception):
    """Exception raised when the file has the wrong file extension."""

    def __init__(self, file: str | Path, file_extension: str | list[str]) -> None:
        super().__init__(
            (
                f"The file {file} has a wrong file extension. Please provide a file with the following extension(s):"
                f" {file_extension}"
            ),
        )


class IllegalSchemaModificationError(Exception):
    """Exception raised when modifying a schema in a way that is inconsistent with the subclass's requirements."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"Illegal schema modification: {msg}")


class ColumnIsTargetError(IllegalSchemaModificationError):
    """Exception raised in overriden methods of the Table class when removing tagged Columns from a TaggedTable."""

    def __init__(self, column_name: str) -> None:
        super().__init__(f'Column "{column_name}" is the target column and cannot be removed.')
