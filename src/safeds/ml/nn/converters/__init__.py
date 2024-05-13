"""Converters between our data contains and tensors."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._input_converter import InputConversion
    from ._input_converter_image import InputConversionImage
    from ._input_converter_table import InputConversionTable
    from ._input_converter_time_series import InputConversionTimeSeries
    from ._output_converter import OutputConversion
    from ._output_converter_image import (
        OutputConversionImageToColumn,
        OutputConversionImageToImage,
        OutputConversionImageToTable,
    )
    from ._output_converter_table import OutputConversionTable
    from ._output_converter_time_series import OutputConversionTimeSeries

apipkg.initpkg(
    __name__,
    {
        "InputConversion": "._input_converter:InputConversion",
        "InputConversionImage": "._input_converter_image:InputConversionImage",
        "InputConversionTable": "._input_converter_table:InputConversionTable",
        "InputConversionTimeSeries": "._input_converter_time_series:InputConversionTimeSeries",
        "OutputConversion": "._output_converter:OutputConversion",
        "OutputConversionImageToColumn": "._output_converter_image:OutputConversionImageToColumn",
        "OutputConversionImageToImage": "._output_converter_image:OutputConversionImageToImage",
        "OutputConversionImageToTable": "._output_converter_image:OutputConversionImageToTable",
        "OutputConversionTable": "._output_converter_table:OutputConversionTable",
        "OutputConversionTimeSeries": "._output_converter_time_series:OutputConversionTimeSeries",
    },
)

__all__ = [
    "InputConversion",
    "InputConversionImage",
    "InputConversionTable",
    "InputConversionTimeSeries",
    "OutputConversion",
    "OutputConversionImageToColumn",
    "OutputConversionImageToImage",
    "OutputConversionImageToTable",
    "OutputConversionTable",
    "OutputConversionTimeSeries",
]
