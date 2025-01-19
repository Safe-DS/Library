"""Validation of preconditions."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._check_bounds_module import _check_bounds, _ClosedBound, _OpenBound
    from ._check_column_has_no_missing_values import _check_column_has_no_missing_values
    from ._check_column_is_numeric_module import _check_column_is_numeric, _check_columns_are_numeric
    from ._check_columns_dont_exist_module import _check_columns_dont_exist
    from ._check_columns_exist_module import _check_columns_exist
    from ._check_indices_module import _check_indices
    from ._check_row_counts_are_equal_module import _check_row_counts_are_equal
    from ._check_schema_module import _check_schema
    from ._check_time_zone_module import _check_time_zone
    from ._convert_and_check_datetime_format_module import _convert_and_check_datetime_format
    from ._normalize_and_check_file_path_module import _normalize_and_check_file_path

apipkg.initpkg(
    __name__,
    {
        "_check_bounds": "._check_bounds_module:_check_bounds",
        "_ClosedBound": "._check_bounds_module:_ClosedBound",
        "_OpenBound": "._check_bounds_module:_OpenBound",
        "_check_column_has_no_missing_values": "._check_column_has_no_missing_values:_check_column_has_no_missing_values",
        "_check_column_is_numeric": "._check_column_is_numeric_module:_check_column_is_numeric",
        "_check_columns_are_numeric": "._check_column_is_numeric_module:_check_columns_are_numeric",
        "_check_columns_dont_exist": "._check_columns_dont_exist_module:_check_columns_dont_exist",
        "_check_columns_exist": "._check_columns_exist_module:_check_columns_exist",
        "_check_indices": "._check_indices_module:_check_indices",
        "_check_row_counts_are_equal": "._check_row_counts_are_equal_module:_check_row_counts_are_equal",
        "_check_schema": "._check_schema_module:_check_schema",
        "_check_time_zone": "._check_time_zone_module:_check_time_zone",
        "_convert_and_check_datetime_format": "._convert_and_check_datetime_format_module:_convert_and_check_datetime_format",
        "_normalize_and_check_file_path": "._normalize_and_check_file_path_module:_normalize_and_check_file_path",
    },
)

__all__ = [
    "_ClosedBound",
    "_OpenBound",
    "_check_bounds",
    "_check_column_has_no_missing_values",
    "_check_column_is_numeric",
    "_check_columns_are_numeric",
    "_check_columns_dont_exist",
    "_check_columns_exist",
    "_check_indices",
    "_check_row_counts_are_equal",
    "_check_schema",
    "_check_time_zone",
    "_convert_and_check_datetime_format",
    "_normalize_and_check_file_path",
]
