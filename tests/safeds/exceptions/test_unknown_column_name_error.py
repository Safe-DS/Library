import pytest
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("column_names", "similar_columns", "expected_error_message"),
    [
        (["column1", "column2"], [], r"Could not find column\(s\) 'column1, column2'."),
    ],
    ids=["empty_list"],
)
def test_empty_similar_columns(
    column_names: list[str],
    similar_columns: list[str],
    expected_error_message: str,
) -> None:
    with pytest.raises(UnknownColumnNameError, match=expected_error_message):
        raise UnknownColumnNameError(column_names, similar_columns)
