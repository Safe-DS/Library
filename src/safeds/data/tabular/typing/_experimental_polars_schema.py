from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import UnknownColumnNameError

from ._experimental_polars_data_type import _PolarsDataType
from ._experimental_schema import ExperimentalSchema

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.typing import ExperimentalDataType


class _PolarsSchema(ExperimentalSchema):
    """
    The schema of a row or table.

    This implementation is based on Polars' data types.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, schema: dict[str, pl.DataType]):
        self._schema: dict[str, pl.DataType] = schema

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PolarsSchema):
            return NotImplemented
        if self is other:
            return True
        return self._schema == other._schema

    def __hash__(self) -> int:
        return _structural_hash(str(self._schema))

    def __repr__(self) -> str:
        return f"Schema({self!s})"

    def __sizeof__(self) -> int:
        return (
            sum(map(sys.getsizeof, self._schema.keys()))
            + sum(map(sys.getsizeof, self._schema.values()))
            + sys.getsizeof(self._schema)
        )

    def __str__(self) -> str:
        match len(self._schema):
            case 0:
                return "{}"
            case 1:
                return str(self._schema)
            case _:
                lines = (f"    {name!r}: {type_}" for name, type_ in self._schema.items())
                joined = ",\n".join(lines)
                return f"{{\n{joined}\n}}"

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return list(self._schema.keys())

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    def get_column_type(self, name: str) -> ExperimentalDataType:
        if not self.has_column(name):
            raise UnknownColumnNameError([name])
        return _PolarsDataType(self._schema[name])

    def has_column(self, name: str) -> bool:
        return name in self._schema

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, ExperimentalDataType]:
        return {name: _PolarsDataType(type_) for name, type_ in self._schema.items()}

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_markdown_(self) -> str:
        if len(self._schema) == 0:
            return "Empty Schema"

        lines = (f"| {name} | {type_} |" for name, type_ in self._schema.items())
        joined = "\n".join(lines)
        return f"| Column Name | Column Type |\n| --- | --- |\n{joined}"
