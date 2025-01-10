import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import (
    Discretizer,
    LabelEncoder,
    OneHotEncoder,
    SequentialTableTransformer,
    SimpleImputer,
    StandardScaler,
    TableTransformer,
)
from safeds.exceptions import NotFittedError, TransformerNotInvertibleError

from tests.helpers import assert_tables_are_equal


class TestInit:
    def test_should_warn_on_empty_list(self) -> None:
        with pytest.warns(UserWarning, match=("transformers should contain at least 1 transformer")):
            SequentialTableTransformer(transformers=[])  # type: ignore[attr-defined]


class TestFit:
    def test_should_raise_value_error_on_empty_table(self) -> None:
        test_table = Table(
            {
                "col1": [],
                "col2": [],
            },
        )
        sequential_table_transformer = SequentialTableTransformer([SimpleImputer(SimpleImputer.Strategy.constant(0))])
        with pytest.raises(
            ValueError,
            match=("The SequentialTableTransformer cannot be fitted because the table contains 0 rows."),
        ):
            sequential_table_transformer.fit(test_table)

    def test_should_not_change_original_transformer(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformer_list = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1, 2, None],
                "col2": ["a", "b", "a"],
            },
        )
        sequential_table_transformer = SequentialTableTransformer(transformers=transformer_list)
        old_hash = hash(sequential_table_transformer)
        sequential_table_transformer.fit(test_table)
        assert old_hash == hash(sequential_table_transformer)


class TestTransform:
    def test_should_raise_if_not_fitted(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1, 2, None],
                "col2": ["a", "b", "a"],
            },
        )
        sequential_table_transformer = SequentialTableTransformer(transformers)
        with pytest.raises(NotFittedError, match=r"The transformer has not been fitted yet."):
            sequential_table_transformer.transform(test_table)

    @pytest.mark.parametrize(
        "transformer",
        [
            OneHotEncoder(),
            SimpleImputer(SimpleImputer.Strategy.constant(0)),
            LabelEncoder(),
            SimpleImputer(SimpleImputer.Strategy.mean()),
        ],
        ids=["OneHotEncoder", "Imputer with Constant", "LabelEncoder", "Mean Imputer"],
    )
    def test_should_do_same_as_transformer_with_single_transformer(self, transformer: TableTransformer) -> None:
        sequential_transformer = SequentialTableTransformer([transformer])
        test_table = Table(
            {
                "col1": [1, 2, None],
                "col2": ["a", "b", "a"],
            },
        )
        sequential_transformer = sequential_transformer.fit(test_table)
        transformer = transformer.fit(test_table)
        test_table_normal = transformer.transform(test_table)
        test_table_sequential = sequential_transformer.transform(test_table)
        assert_tables_are_equal(test_table_normal, test_table_sequential)

    def test_should_transform_with_multiple_transformers(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1, 2, None],
                "col2": ["a", "b", "a"],
            },
        )
        sequential_table_transformer = SequentialTableTransformer(transformers)
        fitted_sequential_table_transformer = sequential_table_transformer.fit(test_table)
        transformed_table_sequential = fitted_sequential_table_transformer.transform(test_table)

        one_hot = one_hot.fit(test_table)
        transformed_table_individual = one_hot.transform(test_table)
        imputer = imputer.fit(transformed_table_individual)
        transformed_table_individual = imputer.transform(transformed_table_individual)

        assert_tables_are_equal(transformed_table_sequential, transformed_table_individual)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        sequential_table_transformer = SequentialTableTransformer(transformers)
        assert sequential_table_transformer.is_fitted is False

    def test_should_return_true_after_fitting(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1, 2, None],
                "col2": ["a", "b", "a"],
            },
        )
        sequential_table_transformer = SequentialTableTransformer(transformers)
        sequential_table_transformer = sequential_table_transformer.fit(test_table)
        assert sequential_table_transformer.is_fitted is True


class TestInverseTransform:
    @pytest.mark.parametrize(
        "transformers",
        [
            [Discretizer(bin_count=3, column_names="col1")],
            [SimpleImputer(SimpleImputer.Strategy.constant(0))],
            [SimpleImputer(SimpleImputer.Strategy.constant(0)), Discretizer(bin_count=3)],
            [
                LabelEncoder(column_names="col2", partial_order=["a", "b", "c"]),
                SimpleImputer(SimpleImputer.Strategy.mean()),
            ],
        ],
        ids=["Discretizer", "SimpleImputer", "Multiple non-invertible", "invertible and non-invertible"],
    )
    def test_should_raise_transformer_not_invertible_error_on_non_invertible_transformers(
        self,
        transformers: list[TableTransformer],
    ) -> None:
        test_table = Table(
            {
                "col1": [0.1, 0.113, 0.232, 1.199, 2.33, 2.01, 2.99],
                "col2": ["a", "a", "c", "b", "a", "a", "c"],
                "col3": [1, 1, None, 3, 14, None, 7],
            },
        )
        sequential_table_transformer = SequentialTableTransformer(transformers)
        sequential_table_transformer = sequential_table_transformer.fit(test_table)
        transformed_table = sequential_table_transformer.transform(test_table)
        with pytest.raises(TransformerNotInvertibleError, match=r".*is not invertible."):
            sequential_table_transformer.inverse_transform(transformed_table)

    @pytest.mark.parametrize(
        "transformers",
        [
            [OneHotEncoder()],
            [OneHotEncoder(), StandardScaler(column_names=["col1", "col3"])],
            [
                LabelEncoder(column_names="col2", partial_order=["a", "b", "c"]),
                OneHotEncoder(),
                StandardScaler(column_names=["col1", "col3"]),
            ],
            [LabelEncoder(), LabelEncoder()],
        ],
        ids=["1 Transformer", "2 Transformers", "3 Transformers", "Duplicate Transformers"],
    )
    def test_should_return_original_table(self, transformers: list[TableTransformer]) -> None:
        test_table = Table(
            {
                "col1": [0.1, 0.113, 0.232, 1.199, 2.33, 2.01, 2.99],
                "col2": ["a", "a", "c", "b", "a", "a", "c"],
                "col3": [1.0, 1.0, 0.0, 3.0, 14.0, 0.0, 7.0],
                "col4": ["one", "two", "one", "two", "one", "two", "one"],
            },
        )
        sequential_table_transformer = SequentialTableTransformer(transformers)
        sequential_table_transformer = sequential_table_transformer.fit(test_table)
        transformed_table = sequential_table_transformer.transform(test_table)
        inverse_transformed_table = sequential_table_transformer.inverse_transform(transformed_table)
        assert_tables_are_equal(test_table, inverse_transformed_table, ignore_column_order=True, ignore_types=True)

    def test_should_raise_transformer_not_fitted_error_if_not_fited(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        sequential_table_transformer = SequentialTableTransformer(transformers)
        test_table = Table(
            {
                "col1": [1, 2, None],
                "col2": ["a", "b", "a"],
            },
        )
        with pytest.raises(NotFittedError, match=r"The transformer has not been fitted yet."):
            sequential_table_transformer.inverse_transform(test_table)
