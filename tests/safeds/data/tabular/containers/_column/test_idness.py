import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "result"),
    [
        ([], 1),
        (["A", "B"], 1),
        (["A", "A", "A", "B"], 0.5),
        (["A", "A", "A", "A"], 0.25),
        (["A", "A", "A", None], 0.5),
    ],
    ids=[
        "empty",
        "all unique values",
        "some unique values",
        "all same values",
        "with missing values",
    ],
)
def test_should_return_idness_of_column(values: list[str], result: float) -> None:
    column = Column("A", values)
    assert column.idness() == result
