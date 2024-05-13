import sys

import pytest
import torch
from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn.converters import (
    OutputConversionImageToColumn,
    OutputConversionImageToImage,
    OutputConversionImageToTable,
)
from safeds.ml.nn.converters._output_conversion_image import _OutputConversionImage


class TestDataConversionImage:

    @pytest.mark.parametrize(
        ("output_conversion", "kwargs"),
        [
            (OutputConversionImageToColumn(), {"column_name": "a", "one_hot_encoder": OneHotEncoder()}),
            (OutputConversionImageToTable(), {"column_names": ["a"]}),
            (OutputConversionImageToImage(), {}),
        ],
    )
    def test_should_raise_if_input_data_is_multi_size(
        self,
        output_conversion: _OutputConversionImage,
        kwargs: dict,
    ) -> None:
        with pytest.raises(ValueError, match=r"The given input ImageList contains images of different sizes."):
            output_conversion._data_conversion(input_data=_MultiSizeImageList(), output_data=torch.empty(1), **kwargs)

    class TestEq:

        @pytest.mark.parametrize(
            ("output_conversion_image1", "output_conversion_image2"),
            [
                (OutputConversionImageToColumn(), OutputConversionImageToColumn()),
                (OutputConversionImageToTable(), OutputConversionImageToTable()),
                (OutputConversionImageToImage(), OutputConversionImageToImage()),
            ],
        )
        def test_should_be_equal(
            self,
            output_conversion_image1: _OutputConversionImage,
            output_conversion_image2: _OutputConversionImage,
        ) -> None:
            assert output_conversion_image1 == output_conversion_image2

        def test_should_be_not_implemented(self) -> None:
            output_conversion_image_to_image = OutputConversionImageToImage()
            output_conversion_image_to_table = OutputConversionImageToTable()
            output_conversion_image_to_column = OutputConversionImageToColumn()
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
                (OutputConversionImageToColumn(), OutputConversionImageToColumn()),
                (OutputConversionImageToTable(), OutputConversionImageToTable()),
                (OutputConversionImageToImage(), OutputConversionImageToImage()),
            ],
        )
        def test_hash_should_be_equal(
            self,
            output_conversion_image1: _OutputConversionImage,
            output_conversion_image2: _OutputConversionImage,
        ) -> None:
            assert hash(output_conversion_image1) == hash(output_conversion_image2)

        def test_hash_should_not_be_equal(self) -> None:
            output_conversion_image_to_image = OutputConversionImageToImage()
            output_conversion_image_to_table = OutputConversionImageToTable()
            output_conversion_image_to_column = OutputConversionImageToColumn()
            assert hash(output_conversion_image_to_image) != hash(output_conversion_image_to_table)
            assert hash(output_conversion_image_to_image) != hash(output_conversion_image_to_column)
            assert hash(output_conversion_image_to_table) != hash(output_conversion_image_to_column)

    class TestSizeOf:

        @pytest.mark.parametrize(
            "output_conversion_image",
            [
                OutputConversionImageToColumn(),
                OutputConversionImageToTable(),
                OutputConversionImageToImage(),
            ],
        )
        def test_should_size_be_greater_than_normal_object(
            self,
            output_conversion_image: _OutputConversionImage,
        ) -> None:
            assert sys.getsizeof(output_conversion_image) > sys.getsizeof(object())


class TestOutputConversionImageToColumn:

    def test_should_raise_if_column_name_not_set(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The column_name is not set. The data can only be converted if the column_name is provided as `str` in the kwargs.",
        ):
            OutputConversionImageToColumn()._data_conversion(
                input_data=_SingleSizeImageList(),
                output_data=torch.empty(1),
                one_hot_encoder=OneHotEncoder(),
            )

    def test_should_raise_if_one_hot_encoder_not_set(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The one_hot_encoder is not set. The data can only be converted if the one_hot_encoder is provided as `OneHotEncoder` in the kwargs.",
        ):
            OutputConversionImageToColumn()._data_conversion(
                input_data=_SingleSizeImageList(),
                output_data=torch.empty(1),
                column_name="column_name",
            )


class TestOutputConversionImageToTable:

    def test_should_raise_if_column_names_not_set(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The column_names are not set. The data can only be converted if the column_names are provided as `list\[str\]` in the kwargs.",
        ):
            OutputConversionImageToTable()._data_conversion(
                input_data=_SingleSizeImageList(),
                output_data=torch.empty(1),
            )
