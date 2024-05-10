"""Types used to define the schema of a tabular dataset."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._data_type import DataType
    from ._schema import Schema

apipkg.initpkg(
    __name__,
    {
        "DataType": "._data_type:DataType",
        "Schema": "._schema:Schema",
    },
)

__all__ = [
    "DataType",
    "Schema",
]
