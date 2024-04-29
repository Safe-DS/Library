import pytest
import torch

from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn import OutputConversionImageToTable, OutputConversionImageToImage
from safeds.ml.nn._output_conversion_image import _OutputConversionImage, OutputConversionImageToColumn


class TestDataConversionToColumn:

    @pytest.mark.parametrize(
        ("output_conversion", "kwargs"),
        [
            (OutputConversionImageToColumn(), {"column_name": "a", "one_hot_encoder": OneHotEncoder()}),
            (OutputConversionImageToTable(), {"column_names": ["a"]}),
            (OutputConversionImageToImage(), {}),
        ]
    )
    def test_should_raise_if_input_data_is_multi_size(self, output_conversion: _OutputConversionImage, kwargs: dict):
        with pytest.raises(ValueError, match=r"The given input ImageList contains images of different sizes."):
            output_conversion._data_conversion(input_data=_MultiSizeImageList(), output_data=torch.empty(1), **kwargs)
