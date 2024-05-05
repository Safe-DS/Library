"""Types used to define the schema of a tabular dataset."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column_type import Anything, Boolean, ColumnType, Integer, Nothing, RealNumber, String
    from ._schema import Schema

apipkg.initpkg(
    __name__,
    {
        "Anything": "._column_type:Anything",
        "Boolean": "._column_type:Boolean",
        "ColumnType": "._column_type:ColumnType",
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
    "Integer",
    "Nothing",
    "RealNumber",
    "Schema",
    "String",
]
