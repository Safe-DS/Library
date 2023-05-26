import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._discretizer import Discretizer
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Parameter 'number_of_bins' must be >= 2."):
            _ = Discretizer(1)


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(UnknownColumnNameError):
            Discretizer().fit(table, ["col2"])

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()
        transformer.fit(table, None)

        assert transformer._wrapped_transformer is None
        assert transformer._column_names is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer().fit(table_to_fit, None)

        table_to_transform = Table(
            {
                "col2": ["a", "b", "c"],
            },
        )

        with pytest.raises(UnknownColumnNameError):
            transformer.transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = Discretizer()
        assert not transformer.is_fitted()

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted()


class TestFitAndTransform:
    # the return value has a formate with "(0,int)\t1.0" and I don't understand how that happens and how the test should look like...
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": ["(0, 0)\t1.0", "(0, 2)\t1.0","(0, 2)\t1.0", "(0, 3)\t1.0"],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
            ),
        ],
        ids=["None", "col1"]
    )
    def test_should_return_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:

        assert Discretizer().fit_and_transform(table, column_names) == expected

    @pytest.mark.parametrize(
        ("table", "number_of_bins", "expected"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                2,
                Table(
                    {
                        "col1": [0, 1.0, 1.0, 1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                10,
                Table(
                    {
                        "col1": [0.0, 4.0, 4.0, 7.0],
                    },
                ),
            ),
        ],
        ids=["2", "10"]
    )
    def test_should_return_transformed_table_with_correct_number_of_bins(
        self,
        table: Table,
        number_of_bins: int,
        expected: Table,
    ) -> None:
        print(Discretizer(number_of_bins).fit_and_transform(table, ["col1"]))
        assert Discretizer(number_of_bins).fit_and_transform(table, ["col1"]) == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        Discretizer().fit_and_transform(table)

        expected = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        assert table == expected

    def test_get_names_of_added_columns(self) -> None:
        transformer = Discretizer()
        with pytest.warns(
            UserWarning,
            match="Discretizer only changes data within columns, but does not add any columns.",
        ), pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_added_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        with pytest.warns(
            UserWarning,
            match="Discretizer only changes data within columns, but does not add any columns.",
        ):
            assert transformer.get_names_of_added_columns() == []

    def test_get_names_of_changed_columns(self) -> None:
        transformer = Discretizer()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_changed_columns()
        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_changed_columns() == ["a"]

    def test_get_names_of_removed_columns(self) -> None:
        transformer = Discretizer()
        with pytest.warns(
            UserWarning,
            match="Discretizer only changes data within columns, but does not remove any columns.",
        ), pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_removed_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        with pytest.warns(
            UserWarning,
            match="Discretizer only changes data within columns, but does not remove any columns.",
        ):
            assert transformer.get_names_of_removed_columns() == []


class TestInverseTransform:
    # inverse_transform doesn't regenerate the old table, it just transforms the discretized data back to original feature space
    # need to test if the features are the same
    @pytest.mark.parametrize(
        "table",
        [
            Table(
                {
                    "col1": [0.0, 5.0, 5.0, 10.0],
                },
            ),
        ],
    )
    def test_should_return_original_table(self, table: Table) -> None:
        transformer = Discretizer().fit(table, None)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        transformer = Discretizer().fit(table, None)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        assert transformed_table == expected

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()

        with pytest.raises(TransformerNotFittedError):
            transformer.inverse_transform(table)
