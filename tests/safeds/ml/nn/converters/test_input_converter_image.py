import sys

import pytest
from safeds.data.image.containers import ImageList
from safeds.data.image.typing import ImageSize
from safeds.data.labeled.containers import ImageDataset
from safeds.data.tabular.containers import Column, Table
from safeds.ml.nn.converters import InputConversionImageToImage

from tests.helpers import images_all, resolve_resource_path

_test_image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)


class TestIsFitDataValid:
    @pytest.mark.parametrize(
        ("image_dataset_valid", "image_dataset_invalid"),
        [
            (
                ImageDataset(_test_image_list, Column("images", images_all())),
                ImageDataset(_test_image_list, _test_image_list),
            ),
            (
                ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
                ImageDataset(_test_image_list, _test_image_list),
            ),
            (
                ImageDataset(_test_image_list, _test_image_list),
                ImageDataset(_test_image_list, Column("images", images_all())),
            ),
            (
                ImageDataset(_test_image_list, _test_image_list),
                ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
            ),
            (
                ImageDataset(_test_image_list, Column("images", images_all())),
                ImageDataset(_test_image_list.resize(20, 20), Column("images", images_all())),
            ),
            (
                ImageDataset(_test_image_list, Column("images", images_all())),
                ImageDataset(_test_image_list, Column("ims", images_all())),
            ),
            (
                ImageDataset(_test_image_list, Column("images", images_all())),
                ImageDataset(_test_image_list, Column("images", [s + "10" for s in images_all()])),
            ),
            (
                ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
                ImageDataset(
                    _test_image_list.resize(20, 20),
                    Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]}),
                ),
            ),
            (
                ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
                ImageDataset(_test_image_list, Table({"b": [0, 0, 1, 1, 0, 1, 0], "c": [1, 1, 0, 0, 1, 0, 1]})),
            ),
            (
                ImageDataset(_test_image_list, _test_image_list),
                ImageDataset(_test_image_list.resize(20, 20), _test_image_list),
            ),
            (
                ImageDataset(_test_image_list, _test_image_list),
                ImageDataset(_test_image_list, _test_image_list.resize(20, 20)),
            ),
        ],
    )
    def test_should_return_false_if_fit_data_is_invalid(
        self,
        image_dataset_valid: ImageDataset,
        image_dataset_invalid: ImageDataset,
    ) -> None:
        input_conversion = InputConversionImageToImage(image_dataset_valid.input_size)
        assert input_conversion._is_fit_data_valid(image_dataset_valid)
        assert input_conversion._is_fit_data_valid(image_dataset_valid)
        assert not input_conversion._is_fit_data_valid(image_dataset_invalid)


class TestEq:
    @pytest.mark.parametrize(
        ("input_conversion_image1", "input_conversion_image2"),
        [(InputConversionImageToImage(ImageSize(1, 2, 3)), InputConversionImageToImage(ImageSize(1, 2, 3)))],
    )
    def test_should_be_equal(
        self,
        input_conversion_image1: InputConversionImageToImage,
        input_conversion_image2: InputConversionImageToImage,
    ) -> None:
        assert input_conversion_image1 == input_conversion_image2

    @pytest.mark.parametrize("input_conversion_image1", [InputConversionImageToImage(ImageSize(1, 2, 3))])
    @pytest.mark.parametrize(
        "input_conversion_image2",
        [
            InputConversionImageToImage(ImageSize(2, 2, 3)),
            InputConversionImageToImage(ImageSize(1, 1, 3)),
            InputConversionImageToImage(ImageSize(1, 2, 1)),
            InputConversionImageToImage(ImageSize(1, 2, 4)),
        ],
    )
    def test_should_not_be_equal(
        self,
        input_conversion_image1: InputConversionImageToImage,
        input_conversion_image2: InputConversionImageToImage,
    ) -> None:
        assert input_conversion_image1 != input_conversion_image2

    def test_should_be_not_implemented(self) -> None:
        input_conversion_image = InputConversionImageToImage(ImageSize(1, 2, 3))
        other = Table()
        assert input_conversion_image.__eq__(other) is NotImplemented


class TestHash:
    @pytest.mark.parametrize(
        ("input_conversion_image1", "input_conversion_image2"),
        [(InputConversionImageToImage(ImageSize(1, 2, 3)), InputConversionImageToImage(ImageSize(1, 2, 3)))],
    )
    def test_hash_should_be_equal(
        self,
        input_conversion_image1: InputConversionImageToImage,
        input_conversion_image2: InputConversionImageToImage,
    ) -> None:
        assert hash(input_conversion_image1) == hash(input_conversion_image2)

    @pytest.mark.parametrize("input_conversion_image1", [InputConversionImageToImage(ImageSize(1, 2, 3))])
    @pytest.mark.parametrize(
        "input_conversion_image2",
        [
            InputConversionImageToImage(ImageSize(2, 2, 3)),
            InputConversionImageToImage(ImageSize(1, 1, 3)),
            InputConversionImageToImage(ImageSize(1, 2, 1)),
            InputConversionImageToImage(ImageSize(1, 2, 4)),
        ],
    )
    def test_hash_should_not_be_equal(
        self,
        input_conversion_image1: InputConversionImageToImage,
        input_conversion_image2: InputConversionImageToImage,
    ) -> None:
        assert hash(input_conversion_image1) != hash(input_conversion_image2)


class TestSizeOf:
    @pytest.mark.parametrize("input_conversion_image", [InputConversionImageToImage(ImageSize(1, 2, 3))])
    def test_should_size_be_greater_than_normal_object(
        self, input_conversion_image: InputConversionImageToImage,
    ) -> None:
        assert sys.getsizeof(input_conversion_image) > sys.getsizeof(object())
