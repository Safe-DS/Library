import sys
import warnings

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import Imputer
from safeds.data.tabular.typing import ImputerStrategy
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError


def strategies() -> list[ImputerStrategy]:
    """
    Return the list of imputer strategies to test.

    After you implemented a new imputer strategy, add it to this list to ensure
    the tests run as expected with the new strategy.

    Returns
    -------
    strategies : list[ImputerStrategy]
        The list of classifiers to test.
    """
    return [Imputer.Strategy.Constant(2), Imputer.Strategy.Mean(), Imputer.Strategy.Median(), Imputer.Strategy.Mode()]


class TestStrategy:
    class TestStr:
        @pytest.mark.parametrize(
            ("strategy", "expected"),
            [
                (Imputer.Strategy.Constant(0), "Constant(0)"),
                (Imputer.Strategy.Mean(), "Mean"),
                (Imputer.Strategy.Median(), "Median"),
                (Imputer.Strategy.Mode(), "Mode"),
            ],
            ids=["Constant", "Mean", "Median", "Mode"],
        )
        def test_should_return_correct_string_representation(self, strategy: ImputerStrategy, expected: str) -> None:
            assert str(strategy) == expected


class TestFit:
    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_column_not_found(self, strategy: ImputerStrategy) -> None:
        table = Table(
            {
                "a": [1, 3, None],
            },
        )

        with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'b, c'"):
            Imputer(strategy).fit(table, ["b", "c"])

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_table_contains_no_rows(self, strategy: ImputerStrategy) -> None:
        with pytest.raises(ValueError, match=r"The Imputer cannot be fitted because the table contains 0 rows"):
            Imputer(strategy).fit(Table({"col1": []}), ["col1"])

    @pytest.mark.parametrize(
        ("table", "col_names", "strategy"),
        [
            (Table({"col1": [1, None, "ok"], "col2": [1, 2, "3"]}), ["col1", "col2"], Imputer.Strategy.Mean()),
            (Table({"col1": [1, None, "ok"], "col2": [1, 2, "3"]}), ["col1", "col2"], Imputer.Strategy.Median()),
        ],
        ids=["Strategy Mean", "Strategy Median"],
    )
    def test_should_raise_if_table_contains_non_numerical_data(
        self,
        table: Table,
        col_names: list[str],
        strategy: ImputerStrategy,
    ) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            Imputer(strategy).fit(table, col_names)

    @pytest.mark.parametrize(
        ("table", "most_frequent"),
        [
            (Table({"col1": [1, 2, 2, 1, 3]}), r"{'col1': \[1, 2\]}"),
            (Table({"col1": ["a1", "a2", "a2", "a1", "a3"]}), r"{'col1': \['a1', 'a2'\]}"),
            (
                Table({"col1": ["a1", "a2", "a2", "a1", "a3"], "col2": [1, 1, 2, 3, 3]}),
                r"{'col1': \['a1', 'a2'\], 'col2': \[1, 3\]}",
            ),
        ],
        ids=["integers", "strings", "multiple columns"],
    )
    def test_should_warn_if_multiple_mode_values(self, table: Table, most_frequent: str) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                r"There are multiple most frequent values in a column given to the Imputer.\nThe lowest values are"
                r" being chosen in this cases. The following columns have multiple most frequent"
                rf" values:\n{most_frequent}"
            ),
        ):
            Imputer(Imputer.Strategy.Mode()).fit(table, None)

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_not_change_original_transformer(self, strategy: ImputerStrategy) -> None:
        table = Table(
            {
                "a": [1, 3, 3, None],
            },
        )

        transformer = Imputer(strategy)
        transformer.fit(table, None)

        assert transformer._wrapped_transformer is None
        assert transformer._column_names is None


class TestTransform:
    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_column_not_found(self, strategy: ImputerStrategy) -> None:
        table_to_fit = Table(
            {
                "a": [1, 3, 3, None],
                "b": [1, 2, 3, 4],
            },
        )

        if isinstance(strategy, Imputer.Strategy.Mode):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message=r"There are multiple most frequent values in a column given to the Imputer\..*",
                    category=UserWarning,
                )
                transformer = Imputer(strategy).fit(table_to_fit, None)
        else:
            transformer = Imputer(strategy).fit(table_to_fit, None)

        table_to_transform = Table(
            {
                "c": [1, 3, 3, None],
            },
        )

        with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'a, b'"):
            transformer.transform(table_to_transform)

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_table_contains_no_rows(self, strategy: ImputerStrategy) -> None:
        with pytest.raises(ValueError, match=r"The Imputer cannot transform the table because it contains 0 rows"):
            Imputer(strategy).fit(Table({"col1": [1, 2, 2]}), ["col1"]).transform(Table({"col1": []}))

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_raise_if_not_fitted(self, strategy: ImputerStrategy) -> None:
        table = Table(
            {
                "a": [1, 3, None],
            },
        )

        transformer = Imputer(strategy)

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)


class TestIsFitted:
    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_return_false_before_fitting(self, strategy: ImputerStrategy) -> None:
        transformer = Imputer(strategy)
        assert not transformer.is_fitted()

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_return_true_after_fitting(self, strategy: ImputerStrategy) -> None:
        table = Table(
            {
                "a": [1, 3, 3, None],
            },
        )

        transformer = Imputer(strategy)
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted()


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "strategy", "expected"),
        [
            (
                Table(
                    {
                        "a": [1.0, 3.0, None],
                    },
                ),
                None,
                Imputer.Strategy.Constant(0.0),
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
                Imputer.Strategy.Mean(),
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
                Imputer.Strategy.Median(),
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
                Imputer.Strategy.Mode(),
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
                Imputer.Strategy.Constant(0.0),
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
                Imputer.Strategy.Mode(),
                Table({"a": [1.0, 1.0, 2.0, 2.0, 1.0]}),
            ),
        ],
        ids=[
            "constant strategy",
            "mean strategy",
            "median strategy",
            "mode strategy",
            "constant strategy multiple columns",
            "mode strategy multiple most frequent values",
        ],
    )
    def test_should_return_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        strategy: ImputerStrategy,
        expected: Table,
    ) -> None:
        if isinstance(strategy, Imputer.Strategy.Mode):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message=r"There are multiple most frequent values in a column given to the Imputer\..*",
                    category=UserWarning,
                )
                assert Imputer(strategy).fit_and_transform(table, column_names) == expected
        else:
            assert Imputer(strategy).fit_and_transform(table, column_names) == expected

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_should_not_change_original_table(self, strategy: ImputerStrategy) -> None:
        table = Table(
            {
                "a": [1, None, None],
            },
        )

        Imputer(strategy=strategy).fit_and_transform(table)

        expected = Table(
            {
                "a": [1, None, None],
            },
        )

        assert table == expected

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_get_names_of_added_columns(self, strategy: ImputerStrategy) -> None:
        transformer = Imputer(strategy=strategy)
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_added_columns()

        table = Table(
            {
                "a": [1, None],
                "b": [1, 1],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_added_columns() == []

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_get_names_of_changed_columns(self, strategy: ImputerStrategy) -> None:
        transformer = Imputer(strategy=strategy)
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_changed_columns()
        table = Table(
            {
                "a": [1, None],
                "b": [1, 1],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_changed_columns() == ["a", "b"]

    @pytest.mark.parametrize("strategy", strategies(), ids=lambda x: x.__class__.__name__)
    def test_get_names_of_removed_columns(self, strategy: ImputerStrategy) -> None:
        transformer = Imputer(strategy=strategy)
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_removed_columns()

        table = Table(
            {
                "a": [1, None],
                "b": [1, 1],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_removed_columns() == []


class TestHash:
    @pytest.mark.parametrize(
        ("strategy1", "strategy2"),
        ([(x, y) for x in strategies() for y in strategies() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_strategy(
        self,
        strategy1: ImputerStrategy,
        strategy2: ImputerStrategy,
    ) -> None:
        assert hash(strategy1) == hash(strategy2)

    @pytest.mark.parametrize(
        ("strategy1", "strategy2"),
        ([(x, y) for x in strategies() for y in strategies() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_strategy(
        self,
        strategy1: ImputerStrategy,
        strategy2: ImputerStrategy,
    ) -> None:
        assert hash(strategy1) != hash(strategy2)


class TestEq:

    @pytest.mark.parametrize(
        ("strategy1", "strategy2"),
        ([(x, y) for x in strategies() for y in strategies() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_equal_strategy(
        self,
        strategy1: ImputerStrategy,
        strategy2: ImputerStrategy,
    ) -> None:
        assert strategy1 == strategy2

    @pytest.mark.parametrize(
        "strategy",
        ([x for x in strategies() if x.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_equal_identity_strategy(
        self,
        strategy: ImputerStrategy,
    ) -> None:
        assert strategy == strategy  # noqa: PLR0124

    @pytest.mark.parametrize(
        ("strategy1", "strategy2"),
        ([(x, y) for x in strategies() for y in strategies() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_unequal_strategy(
        self,
        strategy1: ImputerStrategy,
        strategy2: ImputerStrategy,
    ) -> None:
        assert strategy1 != strategy2


class TestSizeof:
    @pytest.mark.parametrize(
        "strategy",
        ([Imputer.Strategy.Constant(1)]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_sizeof_strategy(
        self,
        strategy: ImputerStrategy,
    ) -> None:
        assert sys.getsizeof(strategy) > sys.getsizeof(object())
