import pytest

from safeds._utils import _get_similar_strings


@pytest.mark.parametrize(
    ("string", "valid_strings", "expected"),
    [
        (
            "column1",
            [],
            [],
        ),
        (
            "column1",
            ["column1", "column2"],
            ["column1"],
        ),
        (
            "dissimilar",
            ["column1", "column2", "column3"],
            [],
        ),
        (
            "cilumn1",
            ["column1", "x", "y"],
            ["column1"],
        ),
        (
            "cilumn1",
            ["column1", "column2", "y"],
            ["column1", "column2"],
        ),
    ],
    ids=["empty", "exact match", "no similar", "one similar", "multiple similar"],
)
def test_should_get_similar_strings(string: str, valid_strings: list[str], expected: list[str]) -> None:
    assert _get_similar_strings(string, valid_strings) == expected
