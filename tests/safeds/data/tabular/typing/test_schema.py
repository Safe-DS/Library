from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest
from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing import Boolean, ColumnType, Integer, RealNumber, Schema, String

if TYPE_CHECKING:
    from typing import Any


class TestFromPandasDataFrame:
    @pytest.mark.parametrize(
        ("dataframe", "expected"),
        [
            (
                pd.DataFrame({"A": [True, False, True]}),
                Schema({"A": Boolean()}),
            ),
            (
                pd.DataFrame({"A": [1, 2, 3]}),
                Schema({"A": Integer()}),
            ),
            (
                pd.DataFrame({"A": [1.0, 2.0, 3.0]}),
                Schema({"A": RealNumber()}),
            ),
            (
                pd.DataFrame({"A": ["a", "b", "c"]}),
                Schema({"A": String()}),
            ),
            (
                pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]}),
                Schema({"A": Integer(), "B": String()}),
            ),
        ],
        ids=[
            "integer",
            "real number",
            "string",
            "boolean",
            "multiple columns",
        ],
    )
    def test_should_create_schema_from_pandas_dataframe(self, dataframe: pd.DataFrame, expected: Schema) -> None:
        assert Schema._from_pandas_dataframe(dataframe) == expected


class TestFromPolarsDataFrame:
    @pytest.mark.parametrize(
        ("dataframe", "expected"),
        [
            (
                pl.DataFrame({"A": [True, False, True]}),
                Schema({"A": Boolean()}),
            ),
            (
                pl.DataFrame({"A": [1, 2, 3]}),
                Schema({"A": Integer()}),
            ),
            (
                pl.DataFrame({"A": [1.0, 2.0, 3.0]}),
                Schema({"A": RealNumber()}),
            ),
            (
                pl.DataFrame({"A": ["a", "b", "c"]}),
                Schema({"A": String()}),
            ),
            (
                pl.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]}),
                Schema({"A": Integer(), "B": String()}),
            ),
        ],
        ids=[
            "integer",
            "real number",
            "string",
            "boolean",
            "multiple columns",
        ],
    )
    def test_should_create_schema_from_polars_dataframe(self, dataframe: pl.DataFrame, expected: Schema) -> None:
        assert Schema._from_polars_dataframe(dataframe) == expected


class TestStr:
    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            (Schema({}), "{}"),
            (Schema({"A": Integer()}), "{'A': Integer}"),
            (Schema({"A": Integer(), "B": String()}), "{\n    'A': Integer,\n    'B': String\n}"),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_create_a_printable_representation(self, schema: Schema, expected: str) -> None:
        assert str(schema) == expected


class TestEq:
    @pytest.mark.parametrize(
        ("schema1", "schema2", "expected"),
        [
            (Schema({}), Schema({}), True),
            (Schema({"col1": Integer()}), Schema({"col1": Integer()}), True),
            (Schema({"col1": Integer()}), Schema({"col1": String()}), False),
            (Schema({"col1": Integer()}), Schema({"col2": Integer()}), False),
            (
                Schema({"col1": Integer(), "col2": String()}),
                Schema({"col2": String(), "col1": Integer()}),
                True,
            ),
        ],
        ids=[
            "empty",
            "same name and type",
            "same name but different type",
            "different name but same type",
            "flipped columns",
        ],
    )
    def test_should_return_whether_two_schema_are_equal(self, schema1: Schema, schema2: Schema, expected: bool) -> None:
        assert (schema1.__eq__(schema2)) == expected

    @pytest.mark.parametrize(
        ("schema", "other"),
        [
            (Schema({"col1": Integer()}), None),
            (Schema({"col1": Integer()}), {"col1": Integer()}),
        ],
    )
    def test_should_return_not_implemented_if_other_is_not_schema(self, schema: Schema, other: Any) -> None:
        assert (schema.__eq__(other)) is NotImplemented


class TestHash:
    @pytest.mark.parametrize(
        ("schema1", "schema2"),
        [
            (Schema({}), Schema({})),
            (Schema({"col1": Integer()}), Schema({"col1": Integer()})),
        ],
        ids=[
            "empty",
            "one column",
        ],
    )
    def test_should_return_same_hash_for_equal_schemas(self, schema1: Schema, schema2: Schema) -> None:
        assert hash(schema1) == hash(schema2)

    @pytest.mark.parametrize(
        ("schema1", "schema2"),
        [
            (Schema({"col1": Integer()}), Schema({"col1": String()})),
            (Schema({"col1": Integer()}), Schema({"col2": Integer()})),
        ],
        ids=[
            "same name but different type",
            "different name but same type",
        ],
    )
    def test_should_return_different_hash_for_unequal_schemas(self, schema1: Schema, schema2: Schema) -> None:
        assert hash(schema1) != hash(schema2)


class TestHasColumn:
    @pytest.mark.parametrize(
        ("schema", "column_name", "expected"),
        [
            (Schema({}), "A", False),
            (Schema({"A": Integer()}), "A", True),
            (Schema({"A": Integer()}), "B", False),
        ],
        ids=[
            "empty",
            "column exists",
            "column does not exist",
        ],
    )
    def test_should_return_whether_column_exists(self, schema: Schema, column_name: str, expected: bool) -> None:
        assert schema.has_column(column_name) == expected


class TestGetTypeOfColumn:
    @pytest.mark.parametrize(
        ("schema", "column_name", "expected"),
        [
            (Schema({"A": Integer()}), "A", Integer()),
            (Schema({"A": Integer(), "B": String()}), "B", String()),
        ],
        ids=[
            "one column",
            "two columns",
        ],
    )
    def test_should_return_type_of_existing_column(
        self,
        schema: Schema,
        column_name: str,
        expected: ColumnType,
    ) -> None:
        assert schema.get_column_type(column_name) == expected

    def test_should_raise_if_column_does_not_exist(self) -> None:
        schema = Schema({"A": Integer()})
        with pytest.raises(UnknownColumnNameError):
            schema.get_column_type("B")


class TestGetColumnNames:
    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            (Schema({}), []),
            (Schema({"A": Integer()}), ["A"]),
            (Schema({"A": Integer(), "B": RealNumber()}), ["A", "B"]),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_return_column_names(self, schema: Schema, expected: list[str]) -> None:
        assert schema.column_names == expected


class TestGetColumnIndex:
    @pytest.mark.parametrize(
        ("schema", "column_name", "expected"),
        [
            (Schema({"A": Integer()}), "A", 0),
            (Schema({"A": Integer(), "B": RealNumber()}), "B", 1),
        ],
        ids=[
            "single column",
            "multiple columns",
        ],
    )
    def test_should_return_column_index(self, schema: Schema, column_name: str, expected: int) -> None:
        assert schema._get_column_index(column_name) == expected

    def test_should_raise_if_column_does_not_exist(self) -> None:
        schema = Schema({"A": Integer()})
        with pytest.raises(UnknownColumnNameError):
            schema._get_column_index("B")
