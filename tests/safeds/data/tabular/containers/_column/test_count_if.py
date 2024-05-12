import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 0),
        ([1], 1),
        ([2], 0),
        ([None], 0),
        ([1, None], 1),
        ([2, None], 0),
        ([1, 2], 1),
        ([1, 2, None], 1),
    ],
    ids=[
        "empty",
        "always true",
        "always false",
        "always unknown",
        "true and unknown",
        "false and unknown",
        "true and false",
        "true and false and unknown",
    ],
)
def test_should_handle_boolean_logic(
    values: list,
    expected: int,
) -> None:
    column = Column("a", values)
    assert column.count_if(lambda value: value < 2) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 0),
        ([1], 1),
        ([2], 0),
        ([None], None),
        ([1, None], None),
        ([2, None], None),
        ([1, 2], 1),
        ([1, 2, None], None),
    ],
    ids=[
        "empty",
        "always true",
        "always false",
        "always unknown",
        "true and unknown",
        "false and unknown",
        "true and false",
        "true and false and unknown",
    ],
)
def test_should_handle_kleene_logic(
    values: list,
    expected: int | None,
) -> None:
    column = Column("a", values)
    assert column.count_if(lambda value: value < 2, ignore_unknown=False) == expected
