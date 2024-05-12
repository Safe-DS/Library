import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], True),
        ([1], False),
        ([2], True),
        ([None], True),
        ([1, None], False),
        ([2, None], True),
        ([1, 2], False),
        ([1, 2, None], False),
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
    expected: bool,
) -> None:
    column = Column("a", values)
    assert column.none(lambda value: value < 2) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], True),
        ([1], False),
        ([2], True),
        ([None], None),
        ([1, None], False),
        ([2, None], None),
        ([1, 2], False),
        ([1, 2, None], False),
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
    expected: bool | None,
) -> None:
    column = Column("a", values)
    assert column.none(lambda value: value < 2, ignore_unknown=False) == expected
