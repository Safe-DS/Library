import sys
import warnings

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import SimpleImputer
from safeds.data.tabular.transformation._simple_imputer import _Mode
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, TransformerNotFittedError


def strategies() -> list[SimpleImputer.Strategy]:
    """
    Return the list of imputer strategies to test.

    After you implemented a new imputer strategy, add it to this list to ensure
    the tests run as expected with the new strategy.

    Returns
    -------
    strategies : list[Imputer.Strategy]
        The list of classifiers to test.
    """
    return [
        SimpleImputer.Strategy.constant(2),
        SimpleImputer.Strategy.mean(),
        SimpleImputer.Strategy.median(),
        SimpleImputer.Strategy.mode(),
    ]


class TestStrategyClass:
    def test_should_be_able_to_get_value_of_constant_strategy(self) -> None:
        assert SimpleImputer.Strategy.constant(1).value == 1  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("strategy", "type_", "expected"),
        [
            (SimpleImputer.Strategy.constant(0), SimpleImputer.Strategy.constant, True),
            (SimpleImputer.Strategy.mean(), SimpleImputer.Strategy.mean, True),
            (SimpleImputer.Strategy.median(), SimpleImputer.Strategy.median, True),
            (SimpleImputer.Strategy.mode(), SimpleImputer.Strategy.mode, True),
            (SimpleImputer.Strategy.mode(), SimpleImputer.Strategy.mean, False),
        ],
    )
    def test_should_be_able_to_use_strategy_in_isinstance(
        self,
        strategy: SimpleImputer.Strategy,
        type_: type,
        expected: bool,
    ) -> None:
        assert isinstance(strategy, type_) == expected

    class TestEq:
        @pytest.mark.parametrize(
            ("strategy1", "strategy2"),
            ([(x, y) for x in strategies() for y in strategies() if x.__class__ == y.__class__]),
            ids=lambda x: x.__class__.__name__,
        )
        def test_equal_strategy(
            self,
            strategy1: SimpleImputer.Strategy,
            strategy2: SimpleImputer.Strategy,
        ) -> None:
            assert strategy1 == strategy2

        @pytest.mark.parametrize(
            "strategy",
            ([x for x in strategies() if x.__class__]),
            ids=lambda x: x.__class__.__name__,
        )
        def test_equal_identity_strategy(
            self,
            strategy: SimpleImputer.Strategy,
        ) -> None:
            assert strategy == strategy  # noqa: PLR0124

        @pytest.mark.parametrize(
            ("strategy1", "strategy2"),
            ([(x, y) for x in strategies() for y in strategies() if x.__class__ != y.__class__]),
            ids=lambda x: x.__class__.__name__,
        )
        def test_unequal_strategy(
            self,
            strategy1: SimpleImputer.Strategy,
            strategy2: SimpleImputer.Strategy,
        ) -> None:
            assert strategy1 != strategy2

    class TestHash:
        @pytest.mark.parametrize(
            ("strategy1", "strategy2"),
            ([(x, y) for x in strategies() for y in strategies() if x.__class__ == y.__class__]),
            ids=lambda x: x.__class__.__name__,
        )
        def test_should_return_same_hash_for_equal_strategy(
            self,
            strategy1: SimpleImputer.Strategy,
            strategy2: SimpleImputer.Strategy,
        ) -> None:
            assert hash(strategy1) == hash(strategy2)

        @pytest.mark.parametrize(
            ("strategy1", "strategy2"),
            ([(x, y) for x in strategies() for y in strategies() if x.__class__ != y.__class__]),
            ids=lambda x: x.__class__.__name__,
        )
        def test_should_return_different_hash_for_unequal_strategy(
            self,
            strategy1: SimpleImputer.Strategy,
            strategy2: SimpleImputer.Strategy,
        ) -> None:
            assert hash(strategy1) != hash(strategy2)

    class TestSizeof:
        @pytest.mark.parametrize(
            "strategy",
            ([SimpleImputer.Strategy.constant(1)]),
            ids=lambda x: x.__class__.__name__,
        )
        def test_sizeof_strategy(
            self,
            strategy: SimpleImputer.Strategy,
        ) -> None:
            assert sys.getsizeof(strategy) > sys.getsizeof(object())

    class TestStr:
        @pytest.mark.parametrize(
            ("strategy", "expected"),
            [
                (SimpleImputer.Strategy.constant(0), "Constant(0)"),
                (SimpleImputer.Strategy.mean(), "Mean"),
                (SimpleImputer.Strategy.median(), "Median"),
                (SimpleImputer.Strategy.mode(), "Mode"),
            ],
            ids=lambda x: x.__class__.__name__,
        )
        def test_should_return_correct_string_representation(
            self,
            strategy: SimpleImputer.Strategy,
            expected: str,
        ) -> None:
            assert str(strategy) == expected


class TestStrategyProperty:
    @pytest.mark.parametrize(
        "strategy",
        strategies(),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_correct_strategy(self, strategy: SimpleImputer.Strategy) -> None:
        assert SimpleImputer(strategy).strategy == strategy


class TestValueToReplaceProperty:
    @pytest.mark.parametrize(
        "value_to_replace",
        [0],
    )
    def test_should_return_correct_value_to_replace(self, value_to_replace: float | str | None) -> None:
        assert (
            SimpleImputer(SimpleImputer.Strategy.mode(), value_to_replace=value_to_replace).value_to_replace
            == value_to_replace
        )


class TestFit:
    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_column_not_found(self, strategy: SimpleImputer.Strategy) -> None:
        table = Table(
            {
                "a": [1, 3, None],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            SimpleImputer(strategy).fit(table, ["b", "c"])

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_table_contains_no_rows(self, strategy: SimpleImputer.Strategy) -> None:
        with pytest.raises(ValueError, match=r"The SimpleImputer cannot be fitted because the table contains 0 rows"):
            SimpleImputer(strategy).fit(Table({"col1": []}), None)

    @pytest.mark.parametrize(
        ("table", "col_names", "strategy"),
        [
            (Table({"col1": [1, None, "ok"], "col2": [1, 2, "3"]}), ["col1", "col2"], SimpleImputer.Strategy.mean()),
            (Table({"col1": [1, None, "ok"], "col2": [1, 2, "3"]}), ["col1", "col2"], SimpleImputer.Strategy.median()),
        ],
        ids=["Strategy Mean", "Strategy Median"],
    )
    def test_should_raise_if_table_contains_non_numerical_data(
        self,
        table: Table,
        col_names: list[str],
        strategy: SimpleImputer.Strategy,
    ) -> None:
        with pytest.raises(ColumnTypeError):
            SimpleImputer(strategy).fit(table, col_names)

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_not_change_original_transformer(self, strategy: SimpleImputer.Strategy) -> None:
        table = Table(
            {
                "a": [1, 3, 3, None],
            },
        )

        transformer = SimpleImputer(strategy)
        transformer.fit(table, None)

        assert transformer._column_names is None
        assert transformer._replacement is None


class TestTransform:
    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_column_not_found(self, strategy: SimpleImputer.Strategy) -> None:
        table_to_fit = Table(
            {
                "a": [1, 3, 3, None],
                "b": [1, 2, 3, 4],
            },
        )

        if isinstance(strategy, _Mode):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message=r"There are multiple most frequent values in a column given to the Imputer\..*",
                    category=UserWarning,
                )
                transformer = SimpleImputer(strategy).fit(table_to_fit, None)
        else:
            transformer = SimpleImputer(strategy).fit(table_to_fit, None)

        table_to_transform = Table(
            {
                "c": [1, 3, 3, None],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            transformer.transform(table_to_transform)

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_not_fitted(self, strategy: SimpleImputer.Strategy) -> None:
        table = Table(
            {
                "a": [1, 3, None],
            },
        )

        transformer = SimpleImputer(strategy)

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)


class TestIsFitted:
    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_return_false_before_fitting(self, strategy: SimpleImputer.Strategy) -> None:
        transformer = SimpleImputer(strategy)
        assert not transformer.is_fitted

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_return_true_after_fitting(self, strategy: SimpleImputer.Strategy) -> None:
        table = Table(
            {
                "a": [1, 3, 3, None],
            },
        )

        transformer = SimpleImputer(strategy)
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "strategy", "value_to_replace", "expected"),
        [
            (
                Table(
                    {
                        "a": [1.0, 3.0, None],
                    },
                ),
                None,
                SimpleImputer.Strategy.constant(0.0),
                None,
                Table(
                    {
                        "a": [1.0, 3.0, 0.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 3.0, None],
                    },
                ),
                None,
                SimpleImputer.Strategy.mean(),
                None,
                Table(
                    {
                        "a": [1.0, 3.0, 2.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 3.0, 1.0, None],
                    },
                ),
                None,
                SimpleImputer.Strategy.median(),
                None,
                Table(
                    {
                        "a": [1.0, 3.0, 1.0, 1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 3.0, 3.0, None],
                    },
                ),
                None,
                SimpleImputer.Strategy.mode(),
                None,
                Table(
                    {
                        "a": [1.0, 3.0, 3.0, 3.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 3.0, None],
                        "b": [1.0, 3.0, None],
                    },
                ),
                ["a"],
                SimpleImputer.Strategy.constant(0.0),
                None,
                Table(
                    {
                        "a": [1.0, 3.0, 0.0],
                        "b": [1.0, 3.0, None],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 1.0, 2.0, 2.0, None],
                    },
                ),
                ["a"],
                SimpleImputer.Strategy.mode(),
                None,
                Table({"a": [1.0, 1.0, 2.0, 2.0, 1.0]}),
            ),
            (
                Table(
                    {
                        "a": [0.0, 1.0, 2.0],
                    },
                ),
                None,
                SimpleImputer.Strategy.constant(1.0),
                0.0,
                Table(
                    {
                        "a": [1.0, 1.0, 2.0],
                    },
                ),
            ),
        ],
        ids=[
            "constant strategy",
            "mean strategy",
            "median strategy",
            "mode strategy",
            "constant strategy multiple columns",
            "mode strategy multiple most frequent values",
            "other value to replace",
        ],
    )
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        strategy: SimpleImputer.Strategy,
        value_to_replace: float | str | None,
        expected: Table,
    ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=r"There are multiple most frequent values in a column given to the Imputer\..*",
                category=UserWarning,
            )
            fitted_transformer, transformed_table = SimpleImputer(
                strategy,
                value_to_replace=value_to_replace,
            ).fit_and_transform(table, column_names)

        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_not_change_original_table(self, strategy: SimpleImputer.Strategy) -> None:
        table = Table(
            {
                "a": [1, None, None],
            },
        )

        SimpleImputer(strategy=strategy).fit_and_transform(table)

        expected = Table(
            {
                "a": [1, None, None],
            },
        )

        assert table == expected
