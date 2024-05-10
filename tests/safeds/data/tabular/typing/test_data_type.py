# TODO
# from collections.abc import Iterable
# from typing import Any
#
# import numpy as np
# import pandas as pd
# import pytest
# from safeds.data.tabular.typing import (
#     Anything,
#     Boolean,
#     ColumnType,
#     Integer,
#     Nothing,
#     RealNumber,
#     String,
# )
#
#
# class TestDataType:
#     @pytest.mark.parametrize(
#         ("data", "expected"),
#         [
#             ([1, 2, 3], Integer(is_nullable=False)),
#             ([1.0, 2.0, 3.0], Integer(is_nullable=False)),
#             ([1.0, 2.5, 3.0], RealNumber(is_nullable=False)),
#             ([True, False, True], Boolean(is_nullable=False)),
#             (["a", "b", "c"], String(is_nullable=False)),
#             (["a", 1, 2.0], Anything(is_nullable=False)),
#             ([None, None, None], Nothing()),
#             ([None, 1, 2], Integer(is_nullable=True)),
#             ([1.0, 2.0, None], Integer(is_nullable=True)),
#             ([1.0, 2.5, None], RealNumber(is_nullable=True)),
#             ([True, False, None], Boolean(is_nullable=True)),
#             (["a", None, "b"], String(is_nullable=True)),
#         ],
#         ids=[
#             "Integer",
#             "Real number .0",
#             "Real number",
#             "Boolean",
#             "String",
#             "Mixed",
#             "None",
#             "Nullable integer",
#             "Nullable RealNumber .0",
#             "Nullable RealNumber",
#             "Nullable Boolean",
#             "Nullable String",
#         ],
#     )
#     def test_should_return_the_data_type(self, data: Iterable, expected: ColumnType) -> None:
#         assert ColumnType._data_type(pd.Series(data)) == expected
#
#     @pytest.mark.parametrize(
#         ("data", "error_message"),
#         [(np.array([1, 2, 3], dtype=np.int16), "Unsupported numpy data type '<class 'numpy.int16'>'.")],
#         ids=["int16 not supported"],
#     )
#     def test_should_throw_not_implemented_error_when_type_is_not_supported(self, data: Any, error_message: str) -> None:
#         with pytest.raises(NotImplementedError, match=error_message):
#             ColumnType._data_type(data)
#
#
# class TestRepr:
#     @pytest.mark.parametrize(
#         ("column_type", "expected"),
#         [
#             (Anything(is_nullable=False), "Anything"),
#             (Anything(is_nullable=True), "Anything?"),
#             (Boolean(is_nullable=False), "Boolean"),
#             (Boolean(is_nullable=True), "Boolean?"),
#             (RealNumber(is_nullable=False), "RealNumber"),
#             (RealNumber(is_nullable=True), "RealNumber?"),
#             (Integer(is_nullable=False), "Integer"),
#             (Integer(is_nullable=True), "Integer?"),
#             (String(is_nullable=False), "String"),
#             (String(is_nullable=True), "String?"),
#         ],
#         ids=repr,
#     )
#     def test_should_create_a_printable_representation(self, column_type: ColumnType, expected: str) -> None:
#         assert repr(column_type) == expected
#
#
# class TestIsNullable:
#     @pytest.mark.parametrize(
#         ("column_type", "expected"),
#         [
#             (Anything(is_nullable=False), False),
#             (Anything(is_nullable=True), True),
#             (Boolean(is_nullable=False), False),
#             (Boolean(is_nullable=True), True),
#             (RealNumber(is_nullable=False), False),
#             (RealNumber(is_nullable=True), True),
#             (Integer(is_nullable=False), False),
#             (Integer(is_nullable=True), True),
#             (String(is_nullable=False), False),
#             (String(is_nullable=True), True),
#         ],
#         ids=repr,
#     )
#     def test_should_return_whether_the_column_type_is_nullable(self, column_type: ColumnType, expected: bool) -> None:
#         assert column_type.is_nullable() == expected
#
#
# class TestIsNumeric:
#     @pytest.mark.parametrize(
#         ("column_type", "expected"),
#         [
#             (Anything(is_nullable=False), False),
#             (Anything(is_nullable=True), False),
#             (Boolean(is_nullable=False), False),
#             (Boolean(is_nullable=True), False),
#             (RealNumber(is_nullable=False), True),
#             (RealNumber(is_nullable=True), True),
#             (Integer(is_nullable=False), True),
#             (Integer(is_nullable=True), True),
#             (String(is_nullable=False), False),
#             (String(is_nullable=True), False),
#         ],
#         ids=repr,
#     )
#     def test_should_return_whether_the_column_type_is_numeric(self, column_type: ColumnType, expected: bool) -> None:
#         assert column_type.is_numeric() == expected
