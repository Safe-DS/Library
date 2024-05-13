from abc import ABC


class ColumnNotFoundError(KeyError, ABC):
    """Exception raised when trying to access an invalid column name."""

class _ColumnNotFoundError(ColumnNotFoundError):


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
