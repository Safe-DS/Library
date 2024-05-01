import pytest
import torch

from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn import OutputConversionImageToTable, OutputConversionImageToImage, OutputConversionImageToColumn
from safeds.ml.nn._output_conversion_image import _OutputConversionImage


class TestDataConversionImage:

    @pytest.mark.parametrize(
        ("output_conversion", "kwargs"),
        [
            (OutputConversionImageToColumn(), {"column_name": "a", "one_hot_encoder": OneHotEncoder()}),
            (OutputConversionImageToTable(), {"column_names": ["a"]}),
            (OutputConversionImageToImage(), {}),
        ]
    )
    def test_should_raise_if_input_data_is_multi_size(self, output_conversion: _OutputConversionImage, kwargs: dict) -> None:
        with pytest.raises(ValueError, match=r"The given input ImageList contains images of different sizes."):
            output_conversion._data_conversion(input_data=_MultiSizeImageList(), output_data=torch.empty(1), **kwargs)


class TestOutputConversionImageToColumn:

    def test_should_raise_if_column_name_not_set(self) -> None:
        with pytest.raises(ValueError, match=r"The column_name is not set. The data can only be converted if the column_name is provided as `str` in the kwargs."):
            OutputConversionImageToColumn()._data_conversion(input_data=_SingleSizeImageList(), output_data=torch.empty(1), one_hot_encoder=OneHotEncoder())

    def test_should_raise_if_one_hot_encoder_not_set(self) -> None:
        with pytest.raises(ValueError, match=r"The one_hot_encoder is not set. The data can only be converted if the one_hot_encoder is provided as `OneHotEncoder` in the kwargs."):
            OutputConversionImageToColumn()._data_conversion(input_data=_SingleSizeImageList(), output_data=torch.empty(1), column_name="column_name")


class TestOutputConversionImageToTable:

    def test_should_raise_if_column_names_not_set(self) -> None:
        with pytest.raises(ValueError, match=r"The column_names are not set. The data can only be converted if the column_names are provided as `list\[str\]` in the kwargs."):
            OutputConversionImageToTable()._data_conversion(input_data=_SingleSizeImageList(), output_data=torch.empty(1))
