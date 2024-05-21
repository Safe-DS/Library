import sys

import pytest
import torch
from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Table
from safeds.ml.nn.converters import (
    _ImageToColumnConverter,
    _ImageToImageConverter,
    _ImageToTableConverter,
)
from safeds.ml.nn.converters._input_converter_image import _ImageConverter


class TestDataConversionImage:
    @pytest.mark.parametrize(
        "input_conversion",
        [
            _ImageToColumnConverter(ImageSize(1, 1, 1)),
        ],
    )
    def test_should_raise_if_input_data_is_multi_size(
        self,
        input_conversion: _ImageConverter,
    ) -> None:
        with pytest.raises(ValueError, match=r"The given input ImageList contains images of different sizes."):
            input_conversion._data_conversion_output(
                input_data=_MultiSizeImageList(),
                output_data=torch.empty(1),
            )

    class TestEq:
        @pytest.mark.parametrize(
            ("output_conversion_image1", "output_conversion_image2"),
            [
                (_ImageToColumnConverter(ImageSize(1, 1, 1)), _ImageToColumnConverter(ImageSize(1, 1, 1))),
                (_ImageToTableConverter(ImageSize(1, 1, 1)), _ImageToTableConverter(ImageSize(1, 1, 1))),
                (_ImageToImageConverter(ImageSize(1, 1, 1)), _ImageToImageConverter(ImageSize(1, 1, 1))),
            ],
        )
        def test_should_be_equal(
            self,
            output_conversion_image1: _ImageConverter,
            output_conversion_image2: _ImageConverter,
        ) -> None:
            assert output_conversion_image1 == output_conversion_image2

        def test_should_be_not_implemented(self) -> None:
            output_conversion_image_to_image = _ImageToImageConverter(ImageSize(1, 1, 1))
            output_conversion_image_to_table = _ImageToTableConverter(ImageSize(1, 1, 1))
            output_conversion_image_to_column = _ImageToColumnConverter(ImageSize(1, 1, 1))
            other = Table()
            assert output_conversion_image_to_image.__eq__(other) is NotImplemented
            assert output_conversion_image_to_image.__eq__(output_conversion_image_to_table) is NotImplemented
            assert output_conversion_image_to_image.__eq__(output_conversion_image_to_column) is NotImplemented
            assert output_conversion_image_to_table.__eq__(other) is NotImplemented
            assert output_conversion_image_to_table.__eq__(output_conversion_image_to_image) is NotImplemented
            assert output_conversion_image_to_table.__eq__(output_conversion_image_to_column) is NotImplemented
            assert output_conversion_image_to_column.__eq__(other) is NotImplemented
            assert output_conversion_image_to_column.__eq__(output_conversion_image_to_table) is NotImplemented
            assert output_conversion_image_to_column.__eq__(output_conversion_image_to_image) is NotImplemented

    class TestHash:
        @pytest.mark.parametrize(
            ("output_conversion_image1", "output_conversion_image2"),
            [
                (_ImageToColumnConverter(ImageSize(1, 1, 1)), _ImageToColumnConverter(ImageSize(1, 1, 1))),
                (_ImageToTableConverter(ImageSize(1, 1, 1)), _ImageToTableConverter(ImageSize(1, 1, 1))),
                (_ImageToImageConverter(ImageSize(1, 1, 1)), _ImageToImageConverter(ImageSize(1, 1, 1))),
            ],
        )
        def test_hash_should_be_equal(
            self,
            output_conversion_image1: _ImageConverter,
            output_conversion_image2: _ImageConverter,
        ) -> None:
            assert hash(output_conversion_image1) == hash(output_conversion_image2)

        def test_hash_should_not_be_equal(self) -> None:
            output_conversion_image_to_image = _ImageToImageConverter(ImageSize(1, 1, 1))
            output_conversion_image_to_table = _ImageToTableConverter(ImageSize(1, 1, 1))
            output_conversion_image_to_column = _ImageToColumnConverter(ImageSize(1, 1, 1))
            assert hash(output_conversion_image_to_image) != hash(output_conversion_image_to_table)
            assert hash(output_conversion_image_to_image) != hash(output_conversion_image_to_column)
            assert hash(output_conversion_image_to_table) != hash(output_conversion_image_to_column)

    class TestSizeOf:
        @pytest.mark.parametrize(
            "output_conversion_image",
            [
                _ImageToColumnConverter(ImageSize(1, 1, 1)),
                _ImageToTableConverter(ImageSize(1, 1, 1)),
                _ImageToImageConverter(ImageSize(1, 1, 1)),
            ],
        )
        def test_should_size_be_greater_than_normal_object(
            self,
            output_conversion_image: _ImageConverter,
        ) -> None:
            assert sys.getsizeof(output_conversion_image) > sys.getsizeof(object())


class TestInputConversionImageToTable:
    def test_should_raise_if_column_names_not_set(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The column_names are not set. The data can only be converted if the column_names are provided as `list\[str\]` in the kwargs.",
        ):
            _ImageToTableConverter(ImageSize(1, 1, 1))._data_conversion_output(
                input_data=_SingleSizeImageList(),
                output_data=torch.empty(1),
            )
