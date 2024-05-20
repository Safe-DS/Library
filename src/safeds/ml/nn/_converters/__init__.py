"""Converters between our data contains and tensors."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._converter import _Converter
    from ._image_converter import _ImageConverter
    from ._image_to_column_converter import _ImageToColumnConverter
    from ._image_to_image_converter import _ImageToImageConverter
    from ._image_to_table_converter import _ImageToTableConverter
    from ._table_converter import _TableConverter
    from ._time_series_converter import _TimeSeriesConverter

apipkg.initpkg(
    __name__,
    {
        "_Converter": "._converter:_Converter",
        "_ImageConverter": "._image_converter:_ImageConverter",
        "_ImageToColumnConverter": "._image_to_column_converter:_ImageToColumnConverter",
        "_ImageToImageConverter": "._image_to_image_converter:_ImageToImageConverter",
        "_ImageToTableConverter": "._image_to_table_converter:_ImageToTableConverter",
        "_TableConverter": "._table_converter:_TableConverter",
        "_TimeSeriesConverter": "._time_series_converter:_TimeSeriesConverter",
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
