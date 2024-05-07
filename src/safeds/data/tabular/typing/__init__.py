"""Types used to define the schema of a tabular dataset."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column_type import Anything, Boolean, ColumnType, Integer, Nothing, RealNumber, String
    from ._experimental_data_type import ExperimentalDataType
    from ._experimental_schema import ExperimentalSchema
    from ._schema import Schema

apipkg.initpkg(
    __name__,
    {
        "Anything": "._column_type:Anything",
        "Boolean": "._column_type:Boolean",
        "ColumnType": "._column_type:ColumnType",
        "ExperimentalDataType": "._experimental_data_type:ExperimentalDataType",
        "ExperimentalSchema": "._experimental_schema:ExperimentalSchema",
        "Integer": "._column_type:Integer",
        "Nothing": "._column_type:Nothing",
        "RealNumber": "._column_type:RealNumber",
        "Schema": "._schema:Schema",
        "String": "._column_type:String",
    },
)

__all__ = [
    "Anything",
    "Boolean",
    "ColumnType",
    "ExperimentalDataType",
    "ExperimentalSchema",
    "Integer",
    "Nothing",
    "RealNumber",
    "Schema",
    "String",
]
