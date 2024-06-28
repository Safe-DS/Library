import numpy as np
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import KNearestNeighborsImputer
from safeds.exceptions import (
    ColumnNotFoundError,
    OutOfBoundsError,
    TransformerNotFittedError,
)


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(OutOfBoundsError):
            KNearestNeighborsImputer(neighbor_count=0)

    def test_neighbor_count(self) -> None:
        knn = KNearestNeighborsImputer(neighbor_count=5)
        assert knn.neighbor_count == 5


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            KNearestNeighborsImputer(column_names=["col2", "col3"]).fit(table)

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The KNearestNeighborsImputer cannot be fitted because the table contains 0 rows",
        ):
            KNearestNeighborsImputer().fit(Table({"col1": []}))

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = KNearestNeighborsImputer()
        transformer.fit(table)

        assert transformer._column_names is None
        assert transformer._wrapped_transformer is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col2": [5.0, 50.0, 100.0],
            },
        )

        transformer = KNearestNeighborsImputer()

        table_to_transform = Table(
            {
                "col3": ["a", "b", "c"],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            transformer.fit(table_to_fit).transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = KNearestNeighborsImputer()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = KNearestNeighborsImputer()
        assert not transformer.is_fitted

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = KNearestNeighborsImputer()
        fitted_transformer = transformer.fit(table)
        assert fitted_transformer.is_fitted


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [1, 2, np.nan],
                        "col2": [1, 2, 3],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": [1, 2, 2],  # Assuming k=1, the nearest neighbor for the missing value is 2.
                        "col2": [1, 2, 3],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": [1, 2, np.nan, 4],
                        "col2": [1, 2, 3, 4],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": [1, 2, 2, 4],  # Assuming k=1, the nearest neighbor for the missing value is 2.
                        "col2": [1, 2, 3, 4],
                    },
                ),
            ),
        ],
        ids=["one_column", "two_columns"],
    )
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,  # noqa: ARG002
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = KNearestNeighborsImputer(
            neighbor_count=1,
            column_names=None,
            value_to_replace=np.nan,
        ).fit_and_transform(table)
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [1, 2, np.nan, 4],
                        "col2": [1, np.nan, 3, 4],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": [1, 2, 7 / 3, 4],  # Assuming k=1, the nearest neighbor for the missing value is 2.
                        "col2": [1, 8 / 3, 3, 4],
                    },
                ),
            ),
        ],
        ids=["two_columns"],
    )
    def test_should_return_fitted_transformer_and_transformed_table_with_correct_values(
        self,
        table: Table,
        column_names: list[str] | None,  # noqa: ARG002
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = KNearestNeighborsImputer(
            neighbor_count=3,
            value_to_replace=np.nan,
        ).fit_and_transform(table)
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        KNearestNeighborsImputer().fit_and_transform(table)

        expected = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        assert table == expected
