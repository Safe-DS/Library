from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler, TableTransformer
from safeds.exceptions import NotFittedError


# We test the behavior of each transformer in their own test file.
@pytest.mark.parametrize(
    ("table_factory", "transformer", "expected"),
    [
        (
            lambda: Table({"col1": [1, 2, 3]}),
            RangeScaler(),
            Table({"col1": [0.0, 0.5, 1.0]}),
        ),
    ],
    ids=[
        "with data",
    ],
)
class TestHappyPath:
    def test_should_return_transformed_table(
        self,
        table_factory: Callable[[], Table],
        transformer: TableTransformer,
        expected: Table,
    ) -> None:
        table = table_factory()
        fitted_transformer = transformer.fit(table)
        actual = table.transform_table(fitted_transformer)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        transformer: TableTransformer,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        fitted_transformer = transformer.fit(original)
        original.transform_table(fitted_transformer)
        assert original == table_factory()


def test_should_raise_if_not_fitted() -> None:
    table = Table({})
    transformer = RangeScaler()

    with pytest.raises(NotFittedError):
        table.transform_table(transformer)
