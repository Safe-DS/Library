class UnknownColumnNameError(KeyError):
    """
    Exception raised for trying to access an invalid column name.

    Parameters
    ----------
    column_names : list[str]
        The name of the column that was tried to be accessed.
    """

    def __init__(self, column_names: list[str]):
        super().__init__(f"Could not find column(s) '{', '.join(column_names)}'")


class NonNumericColumnError(Exception):
    """
    Exception raised for trying to do numerical operations on a non-numerical column.
    """

    def __init__(self, column_info: str) -> None:
        super().__init__(
            f"Tried to do a numerical operation on one or multiple non numerical Columns: \n{column_info}"
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
    index : int
        The wrongly used index.
    """

    def __init__(self, index: int):
        super().__init__(f"There is no element at index '{index}'.")


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
        super().__init__(
            f"Expected a column of size {expected_size} but got column of size {actual_size}."
        )


class SchemaMismatchError(Exception):
    """
    Exception raised when schemas are unequal.
    """

    def __init__(self) -> None:
        super().__init__("Failed because at least two schemas didn't match.")


class MissingSchemaError(Exception):
    """
    Exception raised when a required schema is missing.
    """

    def __init__(self) -> None:
        super().__init__("Failed because a required schema is missing.")


class ColumnLengthMismatchError(Exception):
    """
    Exception raised when the lengths of two or more columns do not match.
    """

    def __init__(self, column_info: str):
        super().__init__(f"The length of at least one column differs: \n{column_info}")


class MissingDataError(Exception):
    """
    Exception raised if a function is not given enough data to succeed.
    """

    def __init__(self, missing_data_info: str):
        super().__init__(f"The function is missing data: \n{missing_data_info}")
