import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import *
from safeds.exceptions import TransformerNotFittedError
from safeds.exceptions import TransformerNotInvertableError

from tests.helpers import assert_tables_equal


class TestInit:
    def test_should_raise_value_error_on_none(self) -> None:
        with pytest.raises(ValueError, match=("transformers must contain at least 1 transformer")):
            SequentialTableTransformer(transformers = None)
    
    def test_should_raise_value_error_on_empty_list(self) -> None:
        with pytest.raises(ValueError, match=("transformers must contain at least 1 transformer")):
            SequentialTableTransformer(transformers = [])

class TestFit:
    def test_should_raise_value_error_on_empty_table(self) -> None:
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [],
                "col2": [],
            },
        )
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        with pytest.raises(ValueError, match=("The SequentialTableTransformer cannot be fitted because the table contains 0 rows.")):
            sequentialTableTransformer.fit(test_table)
    
    def test_fit_does_not_change_original_transformer(self):
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformer_list = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1,2,None],
                "col2": ["a", "b", "a"],
            },
        )
        sequentialTableTransformer = SequentialTableTransformer(transformers=transformer_list)
        old_hash = hash(sequentialTableTransformer)
        sequentialTableTransformer.fit(test_table)
        assert old_hash == hash(sequentialTableTransformer)

class TestTransform:
    def test_should_raise_if_not_fitted(self):
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1,2,None],
                "col2": ["a", "b", "a"],
            },
        )
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            sequentialTableTransformer.transform(test_table)
    
    @pytest.mark.parametrize(
            "transformers",[
                OneHotEncoder,
                SimpleImputer(SimpleImputer.Strategy.constant(0)),
                LabelEncoder(),
                SimpleImputer(SimpleImputer.Strategy.mean())],
            ids=["OneHotEncoder", "Imputer with Constant", "LabelEncoder", "Mean Imputer"],
    )
    def test_should_do_same_as_transformer_with_single_transformer(self, transformer: TableTransformer):
        sequential_transformer = SequentialTableTransformer([transformer])
        test_table = Table(
            {
                "col1": [1,2,None],
                "col2": ["a", "b", "a"],
            },
        )
        sequential_transformer = sequential_transformer.fit(test_table)
        transformer = transformer.fit(test_table)
        test_table_normal = transformer.transform(test_table)
        test_table_sequential = sequential_transformer.transform(test_table)
        assert_tables_equal(test_table_normal, test_table_sequential)

    def test_transforms_correctly_with_multiple_transformers(self):
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1,2,None],
                "col2": ["a", "b", "a"],
            },
        )
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        fitted_sequentialTableTransformer = sequentialTableTransformer.fit(test_table)
        transfromed_table_sequential = fitted_sequentialTableTransformer.transform(test_table)

        one_hot = one_hot.fit(test_table)
        transormed_table_individual = one_hot.transform(test_table)
        imputer = imputer.fit(transormed_table_individual)
        transormed_table_individual = imputer.transform(transormed_table_individual)

        assert_tables_equal(transfromed_table_sequential, transormed_table_individual)

class TestIsFitted:
    def test_should_return_false_before_fiting(self):
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        assert sequentialTableTransformer.is_fitted() == False
    
    def test_should_return_true_after_fiting(self):
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        test_table = Table(
            {
                "col1": [1,2,None],
                "col2": ["a", "b", "a"],
            },
        )
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        sequentialTableTransformer = sequentialTableTransformer.fit(test_table)
        assert sequentialTableTransformer.is_fitted() == True

class TestInverseTransform:

    @pytest.mark.parametrize(
            "transformers",[
                [Discretizer(bin_count=3, column_names="col1")],
                [SimpleImputer(SimpleImputer.Strategy.constant(0))],
                [SimpleImputer(SimpleImputer.Strategy.constant(0)), Discretizer(bin_count=3)],
                [LabelEncoder(column_names="col2", partial_order=["a","b","c"]), SimpleImputer(SimpleImputer.Strategy.mean())],
                ],
            ids=["Discretizer", "SimpleImputer", "Multiple non-invertable", "invertable and non-invertable"],
    )
    def test_should_raise_TransformerNotInvertableError_on_non_invertable_transformers(self,transformers):
        test_table = Table(
            {
                "col1": [0.1,0.113,0.232,1.199,2.33,2.01,2.99],
                "col2": ["a","a","c","b","a","a","c"],
                "col3": [1,1,None,3,14,None,7],
            },
        )
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        sequentialTableTransformer = sequentialTableTransformer.fit(test_table)
        transformed_table = sequentialTableTransformer.transform(test_table)
        with pytest.raises(TransformerNotInvertableError, match=r".*is not invertable."):
            sequentialTableTransformer.inverse_transform(transformed_table)

    # Currently doesn't work as StandardScaler changes int to float and OneHotEncoder changes column order.
    # @pytest.mark.parametrize(
    #         "transformers",[
    #             [OneHotEncoder()],
    #             [OneHotEncoder(),StandardScaler()],
    #             [LabelEncoder(column_names="col2", partial_order=["a","b","c"]), OneHotEncoder(), StandardScaler()],
    #             [LabelEncoder(),LabelEncoder()],
    #             ],
    #         ids=["1 Transformer", "2 Transformers", "3 Transformers", "Duplicate Transformers"],
    # )
    # def test_should_return_original_table(self,transformers):
    #     test_table = Table(
    #         {
    #             "col1": [0.1,0.113,0.232,1.199,2.33,2.01,2.99],
    #             "col2": ["a","a","c","b","a","a","c"],
    #             "col3": [1,1,0,3,14,0,7],
    #             "col4": ["one", "two", "one", "two", "one", "two", "one"],
    #         },
    #     )
    #     sequentialTableTransformer = SequentialTableTransformer(transformers)
    #     sequentialTableTransformer = sequentialTableTransformer.fit(test_table)
    #     transformed_table = sequentialTableTransformer.transform(test_table)
    #     inverse_transformed_table = sequentialTableTransformer.inverse_transform(transformed_table)
    #     assert_tables_equal(test_table, inverse_transformed_table)

    def test_should_raise_TransformerNotFittedError_if_not_fited(self):
        one_hot = OneHotEncoder()
        imputer = SimpleImputer(SimpleImputer.Strategy.constant(0))
        transformers = [one_hot, imputer]
        sequentialTableTransformer = SequentialTableTransformer(transformers)
        test_table = Table(
            {
                "col1": [1,2,None],
                "col2": ["a", "b", "a"],
            }
        )
        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            sequentialTableTransformer.inverse_transform(test_table)

