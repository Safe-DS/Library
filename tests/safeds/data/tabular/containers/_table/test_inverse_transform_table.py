from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import InvertibleTableTransformer, RangeScaler
from safeds.exceptions import NotFittedError


# We test the behavior of each transformer in their own test file.
@pytest.mark.parametrize(
    ("table_factory", "transformer"),
    [
        (
            lambda: Table({"col1": [1, 2, 3]}),
            RangeScaler(),
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
        transformer: InvertibleTableTransformer,
    ) -> None:
        original = table_factory()
        fitted_transformer, transformed_table = transformer.fit_and_transform(original)
        restored = transformed_table.inverse_transform_table(fitted_transformer)
        assert restored == original

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        transformer: InvertibleTableTransformer,
    ) -> None:
        original = table_factory()
        fitted_transformer = transformer.fit(original)
        original.inverse_transform_table(fitted_transformer)
        assert original == table_factory()


def test_should_raise_if_not_fitted() -> None:
    table = Table({})
    transformer = RangeScaler()

    with pytest.raises(NotFittedError):
        table.inverse_transform_table(transformer)
