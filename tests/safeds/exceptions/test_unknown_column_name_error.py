# TODO: move to validation tests
# import pytest
# from safeds.exceptions import ColumnNotFoundError
#
#
# @pytest.mark.parametrize(
#     ("column_names", "similar_columns", "expected_error_message"),
#     [
#         (["column1"], [], r"Could not find column\(s\) 'column1'\."),
#         (["column1", "column2"], [], r"Could not find column\(s\) 'column1, column2'\."),
#         (["column1"], ["column_a"], r"Could not find column\(s\) 'column1'\.\nDid you mean '\['column_a'\]'\?"),
#         (
#             ["column1", "column2"],
#             ["column_a"],
#             r"Could not find column\(s\) 'column1, column2'\.\nDid you mean '\['column_a'\]'\?",
#         ),
#         (
#             ["column1"],
#             ["column_a", "column_b"],
#             r"Could not find column\(s\) 'column1'\.\nDid you mean '\['column_a', 'column_b'\]'\?",
#         ),
#         (
#             ["column1", "column2"],
#             ["column_a", "column_b"],
#             r"Could not find column\(s\) 'column1, column2'\.\nDid you mean '\['column_a', 'column_b'\]'\?",
#         ),
#     ],
#     ids=[
#         "one_unknown_no_suggestions",
#         "two_unknown_no_suggestions",
#         "one_unknown_one_suggestion",
#         "two_unknown_one_suggestion",
#         "one_unknown_two_suggestions",
#         "two_unknown_two_suggestions",
#     ],
# )
# def test_empty_similar_columns(
#     column_names: list[str],
#     similar_columns: list[str],
#     expected_error_message: str,
# ) -> None:
#     with pytest.raises(ColumnNotFoundError, match=expected_error_message):
#         raise ColumnNotFoundError(column_names, similar_columns)
