"""Types used to define the schema of a tabular dataset."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column_type import ColumnType
    from ._schema import Schema

apipkg.initpkg(
    __name__,
    {
        "ColumnType": "._column_type:ColumnType",
        "Schema": "._schema:Schema",
    },
)

__all__ = [
    "ColumnType",
    "Schema",
]
