"""Custom exceptions that can be raised by Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column_not_found_error import ColumnNotFoundError
    from ._data import (
        ColumnLengthMismatchError,
        ColumnSizeError,
        DuplicateColumnNameError,
        DuplicateIndexError,
        IllegalFormatError,
        IndexOutOfBoundsError,
        MissingValuesColumnError,
        NonNumericColumnError,
        OutputLengthMismatchError,
        TransformerNotFittedError,
        ValueNotPresentWhenFittedError,
        WrongFileExtensionError,
    )
    from ._generic import (
        OutOfBoundsError,
    )
    from ._ml import (
        DatasetMissesDataError,
        DatasetMissesFeaturesError,
        FeatureDataMismatchError,
        InputSizeError,
        InvalidModelStructureError,
        LearningError,
        ModelNotFittedError,
        PlainTableError,
        PredictionError,
    )

apipkg.initpkg(
    __name__,
    {
        "ColumnNotFoundError": "._column_not_found_error:ColumnNotFoundError",
        # Generic exceptions
        "OutOfBoundsError": "._generic:OutOfBoundsError",
        # Data exceptions
        "ColumnIsTargetError": "._data:ColumnIsTargetError",
        "ColumnIsTimeError": "._data:ColumnIsTimeError",
        "ColumnLengthMismatchError": "._data:ColumnLengthMismatchError",
        "ColumnSizeError": "._data:ColumnSizeError",
        "DuplicateColumnNameError": "._data:DuplicateColumnNameError",
        "DuplicateIndexError": "._data:DuplicateIndexError",
        "IllegalFormatError": "._data:IllegalFormatError",
        "IllegalSchemaModificationError": "._data:IllegalSchemaModificationError",
        "IndexOutOfBoundsError": "._data:IndexOutOfBoundsError",
        "MissingValuesColumnError": "._data:MissingValuesColumnError",
        "NonNumericColumnError": "._data:NonNumericColumnError",
        "OutputLengthMismatchError": "._data:OutputLengthMismatchError",
        "TransformerNotFittedError": "._data:TransformerNotFittedError",
        "ValueNotPresentWhenFittedError": "._data:ValueNotPresentWhenFittedError",
        "WrongFileExtensionError": "._data:WrongFileExtensionError",
        # ML exceptions
        "DatasetMissesDataError": "._ml:DatasetMissesDataError",
        "DatasetMissesFeaturesError": "._ml:DatasetMissesFeaturesError",
        "FeatureDataMismatchError": "._ml:FeatureDataMismatchError",
        "InputSizeError": "._ml:InputSizeError",
        "InvalidModelStructureError": "._ml:InvalidModelStructureError",
        "LearningError": "._ml:LearningError",
        "ModelNotFittedError": "._ml:ModelNotFittedError",
        "PlainTableError": "._ml:PlainTableError",
        "PredictionError": "._ml:PredictionError",
        # Other
        "Bound": "._generic:Bound",
        "ClosedBound": "._generic:ClosedBound",
        "OpenBound": "._generic:OpenBound",
    },
)

__all__ = [
    "ColumnNotFoundError",
    # Generic exceptions
    "OutOfBoundsError",
    # Data exceptions
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "DuplicateIndexError",
    "IllegalFormatError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "OutputLengthMismatchError",
    "TransformerNotFittedError",
    "ColumnNotFoundError",
    "ValueNotPresentWhenFittedError",
    "WrongFileExtensionError",
    # ML exceptions
    "DatasetMissesDataError",
    "DatasetMissesFeaturesError",
    "FeatureDataMismatchError",
    "InputSizeError",
    "InvalidModelStructureError",
    "LearningError",
    "ModelNotFittedError",
    "PlainTableError",
    "PredictionError",
]
