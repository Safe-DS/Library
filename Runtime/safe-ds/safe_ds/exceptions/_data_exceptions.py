class ColumnNameError(Exception):
    """Exception raised for trying to access an invalid column name.

    Parameters
    ----------
    column_names: list[str]
        Name of the column that was tried to be accessed
    """

    def __init__(self, column_names: list[str]):
        super().__init__(f"Could not find column(s) '{', '.join(column_names)}'")


class ColumnNameDuplicateError(Exception):
    """Exception raised for trying to modify a table, resulting in a duplicate column name.

    Parameters
    ----------
    column_name: str
        Name of the column that resulted in a duplicate
    """

    def __init__(self, column_name: str):
        super().__init__(f"Column '{column_name}' already exists.")
