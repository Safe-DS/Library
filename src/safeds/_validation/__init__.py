"""Validation of preconditions."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._check_columns_exist import _check_columns_exist
    from ._normalize_and_check_file_path import _normalize_and_check_file_path

apipkg.initpkg(
    __name__,
    {
        "_check_columns_exist": "._check_columns_exist:_check_columns_exist",
        "_normalize_and_check_file_path": "._normalize_and_check_file_path:_normalize_and_check_file_path",
    },
)

__all__ = [
    "_check_columns_exist",
    "_normalize_and_check_file_path",
]
