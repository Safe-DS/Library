import pytest
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("column_names", "similar_columns", "expected_error_message"),
    [
        (["column1", "column2"], [], "Could not find column(s) 'column1, column2'."),
    ],
    ids=["empty_list"]
)
def test_empty_similar_columns(column_names, similar_columns, expected_error_message):
    error = UnknownColumnNameError(column_names, similar_columns)
    assert str(error) == expected_error_message

