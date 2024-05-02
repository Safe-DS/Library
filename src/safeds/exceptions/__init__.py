"""Custom exceptions that can be raised by Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from safeds.exceptions._data import (
        ColumnIsTargetError,
        ColumnIsTimeError,
        ColumnLengthMismatchError,
        ColumnSizeError,
        DuplicateColumnNameError,
        DuplicateIndexError,
        IllegalFormatError,
        IllegalSchemaModificationError,
        IndexOutOfBoundsError,
        MissingValuesColumnError,
        NonNumericColumnError,
        TransformerNotFittedError,
        UnknownColumnNameError,
        ValueNotPresentWhenFittedError,
        WrongFileExtensionError,
    )
    from safeds.exceptions._generic import (
        Bound,
        ClosedBound,
        OpenBound,
        OutOfBoundsError,
    )
    from safeds.exceptions._ml import (
        DatasetMissesDataError,
        DatasetMissesFeaturesError,
        FeatureDataMismatchError,
        InputSizeError,
        LearningError,
        ModelNotFittedError,
        NonTimeSeriesError,
        PlainTableError,
        PredictionError,
    )

apipkg.initpkg(
    __name__,
    {
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
        "TransformerNotFittedError": "._data:TransformerNotFittedError",
        "UnknownColumnNameError": "._data:UnknownColumnNameError",
        "ValueNotPresentWhenFittedError": "._data:ValueNotPresentWhenFittedError",
        "WrongFileExtensionError": "._data:WrongFileExtensionError",
        # ML exceptions
        "DatasetMissesDataError": "._ml:DatasetMissesDataError",
        "DatasetMissesFeaturesError": "._ml:DatasetMissesFeaturesError",
        "FeatureDataMismatchError": "._ml:FeatureDataMismatchError",
        "InputSizeError": "._ml:InputSizeError",
        "LearningError": "._ml:LearningError",
        "ModelNotFittedError": "._ml:ModelNotFittedError",
        "NonTimeSeriesError": "._ml:NonTimeSeriesError",
        "PlainTableError": "._ml:PlainTableError",
        "PredictionError": "._ml:PredictionError",
        # Other
        "Bound": "._generic:Bound",
        "ClosedBound": "._generic:ClosedBound",
        "OpenBound": "._generic:OpenBound",
    },
)

__all__ = [
    # Generic exceptions
    "OutOfBoundsError",
    # Data exceptions
    "ColumnIsTargetError",
    "ColumnIsTimeError",
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "DuplicateIndexError",
    "IllegalFormatError",
    "IllegalSchemaModificationError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "TransformerNotFittedError",
    "UnknownColumnNameError",
    "ValueNotPresentWhenFittedError",
    "WrongFileExtensionError",
    # ML exceptions
    "DatasetMissesDataError",
    "DatasetMissesFeaturesError",
    "FeatureDataMismatchError",
    "InputSizeError",
    "LearningError",
    "ModelNotFittedError",
    "NonTimeSeriesError",
    "PlainTableError",
    "PredictionError",
    # Other
    "Bound",
    "ClosedBound",
    "OpenBound",
]
