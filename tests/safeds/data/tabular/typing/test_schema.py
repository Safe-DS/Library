import pytest

from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing import Integer, RealNumber, Schema, ColumnType, String


# TODO: test _from_pandas_dataframe
# TODO: test _from_polars_dataframe

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


# TODO: test __eq__
# TODO: test __hash__

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
    def test_should_return_type_of_existing_column(self, schema: Schema, column_name: str,
                                                   expected: ColumnType) -> None:
        assert schema.get_type_of_column(column_name) == expected

    def test_should_raise_if_column_does_not_exist(self) -> None:
        schema = Schema({"A": Integer()})
        with pytest.raises(UnknownColumnNameError):
            schema.get_type_of_column("B")


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
        assert schema.get_column_names() == expected


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
