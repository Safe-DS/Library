"""Converters between our data contains and tensors."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._input_converter import InputConversion
    from ._input_converter_image_to_column import InputConversionImageToColumn
    from ._input_converter_image_to_image import InputConversionImageToImage
    from ._input_converter_image_to_table import InputConversionImageToTable
    from ._input_converter_table import InputConversionTable
    from ._input_converter_time_series import InputConversionTimeSeries

apipkg.initpkg(
    __name__,
    {
        "InputConversion": "._input_converter:InputConversion",
        "InputConversionImageToColumn": "._input_converter_image_to_column:InputConversionImageToColumn",
        "InputConversionImageToImage": "._input_converter_image_to_image:InputConversionImageToImage",
        "InputConversionImageToTable": "._input_converter_image_to_table:InputConversionImageToTable",
        "InputConversionTable": "._input_converter_table:InputConversionTable",
        "InputConversionTimeSeries": "._input_converter_time_series:InputConversionTimeSeries",
    },
)

__all__ = [
    "InputConversion",
    "InputConversionImageToColumn",
    "InputConversionImageToImage",
    "InputConversionImageToTable",
    "InputConversionTable",
    "InputConversionTimeSeries",
]
