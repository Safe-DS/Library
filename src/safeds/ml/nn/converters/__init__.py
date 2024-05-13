"""Converters between our data contains and tensors."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._input_conversion import InputConversion
    from ._input_conversion_image import InputConversionImage
    from ._input_conversion_table import InputConversionTable
    from ._input_conversion_time_series import InputConversionTimeSeries
    from ._output_conversion import OutputConversion
    from ._output_conversion_image import (
        OutputConversionImageToColumn,
        OutputConversionImageToImage,
        OutputConversionImageToTable,
    )
    from ._output_conversion_table import OutputConversionTable
    from ._output_conversion_time_series import OutputConversionTimeSeries

apipkg.initpkg(
    __name__,
    {
        "InputConversion": "._input_conversion:InputConversion",
        "InputConversionImage": "._input_conversion_image:InputConversionImage",
        "InputConversionTable": "._input_conversion_table:InputConversionTable",
        "InputConversionTimeSeries": "._input_conversion_time_series:InputConversionTimeSeries",
        "OutputConversion": "._output_conversion:OutputConversion",
        "OutputConversionImageToColumn": "._output_conversion_image:OutputConversionImageToColumn",
        "OutputConversionImageToImage": "._output_conversion_image:OutputConversionImageToImage",
        "OutputConversionImageToTable": "._output_conversion_image:OutputConversionImageToTable",
        "OutputConversionTable": "._output_conversion_table:OutputConversionTable",
        "OutputConversionTimeSeries": "._output_conversion_time_series:OutputConversionTimeSeries",
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
