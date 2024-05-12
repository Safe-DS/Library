import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], True),
        ([1], True),
        ([2], False),
        ([None], True),
        ([1, None], True),
        ([2, None], False),
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
    expected: bool | None,
) -> None:
    column = Column("a", values)
    assert column.all(lambda value: value < 2) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], True),
        ([1], True),
        ([2], False),
        ([None], None),
        ([1, None], None),
        ([2, None], False),
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
    assert column.all(lambda value: value < 2, ignore_unknown=False) == expected
