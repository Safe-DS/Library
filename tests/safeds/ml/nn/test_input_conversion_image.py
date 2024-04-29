import pytest

from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.tabular.containers import Column, Table
from safeds.ml.nn import InputConversionImage
from tests.helpers import resolve_resource_path, images_all

_test_image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)


class TestIsFitDataValid:

    @pytest.mark.parametrize(
        ("image_dataset_valid", "image_dataset_invalid"),
        [
            (ImageDataset(_test_image_list, Column("images", images_all())), ImageDataset(_test_image_list, _test_image_list)),
            (ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})), ImageDataset(_test_image_list, _test_image_list)),
            (ImageDataset(_test_image_list, _test_image_list), ImageDataset(_test_image_list, Column("images", images_all()))),
            (ImageDataset(_test_image_list, _test_image_list), ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]}))),
            (ImageDataset(_test_image_list, Column("images", images_all())), ImageDataset(_test_image_list.resize(20, 20), Column("images", images_all()))),
            (ImageDataset(_test_image_list, Column("images", images_all())), ImageDataset(_test_image_list, Column("ims", images_all()))),
            (ImageDataset(_test_image_list, Column("images", images_all())), ImageDataset(_test_image_list, Column("images", [s + "10" for s in images_all()]))),
            (ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})), ImageDataset(_test_image_list.resize(20, 20), Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]}))),
            (ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})), ImageDataset(_test_image_list, Table({"b": [0, 0, 1, 1, 0, 1, 0], "c": [1, 1, 0, 0, 1, 0, 1]}))),
            (ImageDataset(_test_image_list, _test_image_list), ImageDataset(_test_image_list.resize(20, 20), _test_image_list)),
            (ImageDataset(_test_image_list, _test_image_list), ImageDataset(_test_image_list, _test_image_list.resize(20, 20))),
        ]
    )
    def test_should_return_false_if_fit_data_is_invalid(self, image_dataset_valid: ImageDataset, image_dataset_invalid: ImageDataset):
        input_conversion = InputConversionImage(image_dataset_valid.input_size)
        assert input_conversion._is_fit_data_valid(image_dataset_valid)
        assert input_conversion._is_fit_data_valid(image_dataset_valid)
        assert not input_conversion._is_fit_data_valid(image_dataset_invalid)

