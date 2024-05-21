"""Converters between our data contains and tensors."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._input_converter import _Converter
    from ._input_converter_image import _ImageConverter
    from ._input_converter_image_to_column import _ImageToColumnConverter
    from ._input_converter_image_to_image import _ImageToImageConverter
    from ._input_converter_image_to_table import _ImageToTableConverter
    from ._input_converter_table import _TableConverter
    from ._input_converter_time_series import _TimeSeriesConverter

apipkg.initpkg(
    __name__,
    {
        "_Converter": "._input_converter:_Converter",
        "_ImageConverter": "._input_converter_image:_ImageConverter",
        "_ImageToColumnConverter": "._input_converter_image_to_column:_ImageToColumnConverter",
        "_ImageToImageConverter": "._input_converter_image_to_image:_ImageToImageConverter",
        "_ImageToTableConverter": "._input_converter_image_to_table:_ImageToTableConverter",
        "_TableConverter": "._input_converter_table:_TableConverter",
        "_TimeSeriesConverter": "._input_converter_time_series:_TimeSeriesConverter",
    },
)

__all__ = [
    "_Converter",
    "_ImageConverter",
    "_ImageToColumnConverter",
    "_ImageToImageConverter",
    "_ImageToTableConverter",
    "_TableConverter",
    "_TimeSeriesConverter",
]
