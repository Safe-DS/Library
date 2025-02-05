import itertools

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import (
    Discretizer,
    FunctionalTableTransformer,
    KNearestNeighborsImputer,
    LabelEncoder,
    OneHotEncoder,
    RangeScaler,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
    TableTransformer,
)


def transformers_numeric() -> list[TableTransformer]:
    """
    Return the list of numeric transformers to test.

    After you implemented a new numeric transformer, add it to this list to ensure its `__hash__` method works as
    expected. Place tests of methods that are specific to your transformer in a separate test file.

    Returns
    -------
    classifiers : list[TableTransformer]
        The list of numeric transformers to test.
    """
    return [
        StandardScaler(selector="col1"),
        RangeScaler(selector="col1"),
        Discretizer(selector="col1"),
        RobustScaler(selector="col1"),
    ]


def transformers_non_numeric() -> list[TableTransformer]:
    """
    Return the list of non-numeric transformers to test.

    After you implemented a new non-numeric transformer, add it to this list to ensure its `__hash__` method works as
    expected. Place tests of methods that are specific to your transformer in a separate test file.

    Returns
    -------
    classifiers : list[TableTransformer]
        The list of non-numeric transformers to test.
    """
    return [
        OneHotEncoder(selector="col1"),
        LabelEncoder(selector="col1"),
    ]


def valid_callable_for_functional_table_transformer(table: Table) -> Table:
    return table.remove_columns(["col1"])


def transformers() -> list[TableTransformer]:
    """
    Return the list of all transformers to test.

    After you implemented a new transformer (which is either applicable to both numeric and non-numeric data or none), add it to one of the three lists to ensure its `__hash__` method works as
    expected. Place tests of methods that are specific to your in a separate test file.

    Returns
    -------
    classifiers : list[TableTransformer]
        The list of all transformers to test.
    """
    return (
        transformers_numeric()
        + transformers_non_numeric()
        + [
            SimpleImputer(strategy=SimpleImputer.Strategy.mode()),
            KNearestNeighborsImputer(neighbor_count=3, value_to_replace=None),
            FunctionalTableTransformer(valid_callable_for_functional_table_transformer),
        ]
    )


@pytest.fixture
def valid_data_numeric() -> Table:
    return Table(
        {
            "col1": [0.0, 5.0, 10.0],
        },
    )


@pytest.fixture
def valid_data_non_numeric() -> Table:
    return Table(
        {
            "col1": ["a", "b", "c"],
        },
    )


@pytest.fixture
def valid_data_imputer() -> Table:
    return Table(
        {
            "col1": [1, 1, None],
        },
    )


class TestHash:
    @pytest.mark.parametrize(
        ("transformer1", "transformer2"),
        ([(x, y) for x in transformers() for y in transformers() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_transformer(
        self,
        transformer1: TableTransformer,
        transformer2: TableTransformer,
    ) -> None:
        assert hash(transformer1) == hash(transformer2)

    @pytest.mark.parametrize(
        ("transformer1", "transformer2"),
        ([(x, y) for x in transformers() for y in transformers() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_transformer(
        self,
        transformer1: TableTransformer,
        transformer2: TableTransformer,
    ) -> None:
        assert hash(transformer1) != hash(transformer2)

    @pytest.mark.parametrize("transformer1", transformers_numeric(), ids=lambda x: x.__class__.__name__)
    def test_should_return_different_hash_for_same_numeric_transformer_fit(
        self,
        transformer1: TableTransformer,
        valid_data_numeric: Table,
    ) -> None:
        transformer1_fit = transformer1.fit(valid_data_numeric)
        assert hash(transformer1) != hash(transformer1_fit)

    @pytest.mark.parametrize("transformer1", transformers_non_numeric(), ids=lambda x: x.__class__.__name__)
    def test_should_return_different_hash_for_same_non_numeric_transformer_fit(
        self,
        transformer1: TableTransformer,
        valid_data_non_numeric: Table,
    ) -> None:
        transformer1_fit = transformer1.fit(valid_data_non_numeric)
        assert hash(transformer1) != hash(transformer1_fit)

    @pytest.mark.parametrize(
        ("transformer1", "transformer2"),
        (list(itertools.product(transformers_numeric(), transformers()))),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_numeric_transformer_fit(
        self,
        transformer1: TableTransformer,
        transformer2: TableTransformer,
        valid_data_numeric: Table,
    ) -> None:
        transformer1_fit = transformer1.fit(valid_data_numeric)
        assert hash(transformer2) != hash(transformer1_fit)

    @pytest.mark.parametrize(
        ("transformer1", "transformer2"),
        (list(itertools.product(transformers_non_numeric(), transformers()))),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_non_numeric_transformer_fit(
        self,
        transformer1: TableTransformer,
        transformer2: TableTransformer,
        valid_data_non_numeric: Table,
    ) -> None:
        transformer1_fit = transformer1.fit(valid_data_non_numeric)
        assert hash(transformer2) != hash(transformer1_fit)

    @pytest.mark.parametrize("transformer2", transformers(), ids=lambda x: x.__class__.__name__)
    def test_should_return_different_hash_for_imputer_fit(
        self,
        transformer2: TableTransformer,
        valid_data_imputer: Table,
    ) -> None:
        transformer1 = SimpleImputer(strategy=SimpleImputer.Strategy.mode())
        transformer1_fit = transformer1.fit(valid_data_imputer)
        assert hash(transformer2) != hash(transformer1_fit)
