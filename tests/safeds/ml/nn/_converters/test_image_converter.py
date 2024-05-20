# TODO: individual tests for the different implementations of _ImageConverter
# import sys
#
# import pytest
# from safeds.data.image.containers import ImageList
# from safeds.data.image.typing import ImageSize
# from safeds.data.labeled.containers import ImageDataset
# from safeds.data.tabular.containers import Column, Table
# from safeds.ml.nn._converters import _ImageToImageConverter
#
# from tests.helpers import images_all, resolve_resource_path
#
# _test_image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
#
#
# class TestIsFitDataValid:
#     @pytest.mark.parametrize(
#         ("image_dataset_valid", "image_dataset_invalid"),
#         [
#             (
#                 ImageDataset(_test_image_list, Column("images", images_all())),
#                 ImageDataset(_test_image_list, _test_image_list),
#             ),
#             (
#                 ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
#                 ImageDataset(_test_image_list, _test_image_list),
#             ),
#             (
#                 ImageDataset(_test_image_list, _test_image_list),
#                 ImageDataset(_test_image_list, Column("images", images_all())),
#             ),
#             (
#                 ImageDataset(_test_image_list, _test_image_list),
#                 ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
#             ),
#             (
#                 ImageDataset(_test_image_list, Column("images", images_all())),
#                 ImageDataset(_test_image_list.resize(20, 20), Column("images", images_all())),
#             ),
#             (
#                 ImageDataset(_test_image_list, Column("images", images_all())),
#                 ImageDataset(_test_image_list, Column("ims", images_all())),
#             ),
#             (
#                 ImageDataset(_test_image_list, Column("images", images_all())),
#                 ImageDataset(_test_image_list, Column("images", [s + "10" for s in images_all()])),
#             ),
#             (
#                 ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
#                 ImageDataset(
#                     _test_image_list.resize(20, 20),
#                     Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]}),
#                 ),
#             ),
#             (
#                 ImageDataset(_test_image_list, Table({"a": [0, 0, 1, 1, 0, 1, 0], "b": [1, 1, 0, 0, 1, 0, 1]})),
#                 ImageDataset(_test_image_list, Table({"b": [0, 0, 1, 1, 0, 1, 0], "c": [1, 1, 0, 0, 1, 0, 1]})),
#             ),
#             (
#                 ImageDataset(_test_image_list, _test_image_list),
#                 ImageDataset(_test_image_list.resize(20, 20), _test_image_list),
#             ),
#             (
#                 ImageDataset(_test_image_list, _test_image_list),
#                 ImageDataset(_test_image_list, _test_image_list.resize(20, 20)),
#             ),
#         ],
#     )
#     def test_should_return_false_if_fit_data_is_invalid(
#         self,
#         image_dataset_valid: ImageDataset,
#         image_dataset_invalid: ImageDataset,
#     ) -> None:
#         input_conversion = _ImageToImageConverter(image_dataset_valid.input_size)
#         assert input_conversion._is_fit_data_valid(image_dataset_valid)
#         assert input_conversion._is_fit_data_valid(image_dataset_valid)
#         assert not input_conversion._is_fit_data_valid(image_dataset_invalid)
#
#
# class TestEq:
#     @pytest.mark.parametrize(
#         ("input_conversion_image1", "input_conversion_image2"),
#         [(_ImageToImageConverter(ImageSize(1, 2, 3)), _ImageToImageConverter(ImageSize(1, 2, 3)))],
#     )
#     def test_should_be_equal(
#         self,
#         input_conversion_image1: _ImageToImageConverter,
#         input_conversion_image2: _ImageToImageConverter,
#     ) -> None:
#         assert input_conversion_image1 == input_conversion_image2
#
#     @pytest.mark.parametrize("input_conversion_image1", [_ImageToImageConverter(ImageSize(1, 2, 3))])
#     @pytest.mark.parametrize(
#         "input_conversion_image2",
#         [
#             _ImageToImageConverter(ImageSize(2, 2, 3)),
#             _ImageToImageConverter(ImageSize(1, 1, 3)),
#             _ImageToImageConverter(ImageSize(1, 2, 1)),
#             _ImageToImageConverter(ImageSize(1, 2, 4)),
#         ],
#     )
#     def test_should_not_be_equal(
#         self,
#         input_conversion_image1: _ImageToImageConverter,
#         input_conversion_image2: _ImageToImageConverter,
#     ) -> None:
#         assert input_conversion_image1 != input_conversion_image2
#
#     def test_should_be_not_implemented(self) -> None:
#         input_conversion_image = _ImageToImageConverter(ImageSize(1, 2, 3))
#         other = Table()
#         assert input_conversion_image.__eq__(other) is NotImplemented
#
#
# class TestHash:
#     @pytest.mark.parametrize(
#         ("input_conversion_image1", "input_conversion_image2"),
#         [(_ImageToImageConverter(ImageSize(1, 2, 3)), _ImageToImageConverter(ImageSize(1, 2, 3)))],
#     )
#     def test_hash_should_be_equal(
#         self,
#         input_conversion_image1: _ImageToImageConverter,
#         input_conversion_image2: _ImageToImageConverter,
#     ) -> None:
#         assert hash(input_conversion_image1) == hash(input_conversion_image2)
#
#     @pytest.mark.parametrize("input_conversion_image1", [_ImageToImageConverter(ImageSize(1, 2, 3))])
#     @pytest.mark.parametrize(
#         "input_conversion_image2",
#         [
#             _ImageToImageConverter(ImageSize(2, 2, 3)),
#             _ImageToImageConverter(ImageSize(1, 1, 3)),
#             _ImageToImageConverter(ImageSize(1, 2, 1)),
#             _ImageToImageConverter(ImageSize(1, 2, 4)),
#         ],
#     )
#     def test_hash_should_not_be_equal(
#         self,
#         input_conversion_image1: _ImageToImageConverter,
#         input_conversion_image2: _ImageToImageConverter,
#     ) -> None:
#         assert hash(input_conversion_image1) != hash(input_conversion_image2)
#
#
# class TestSizeOf:
#     @pytest.mark.parametrize("input_conversion_image", [_ImageToImageConverter(ImageSize(1, 2, 3))])
#     def test_should_size_be_greater_than_normal_object(
#         self,
#         input_conversion_image: _ImageToImageConverter,
#     ) -> None:
#         assert sys.getsizeof(input_conversion_image) > sys.getsizeof(object())

# import sys
#
# import pytest
# import torch
# from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
# from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
# from safeds.data.image.typing import ImageSize
# from safeds.data.tabular.containers import Table
# from safeds.ml.nn._converters import (
#     _ImageConverter,
#     _ImageToColumnConverter,
#     _ImageToImageConverter,
#     _ImageToTableConverter,
# )
#
#
# class TestDataConversionImage:
#     @pytest.mark.parametrize(
#         "input_conversion",
#         [
#             _ImageToColumnConverter(ImageSize(1, 1, 1)),
#         ],
#     )
#     def test_should_raise_if_input_data_is_multi_size(
#         self,
#         input_conversion: _ImageConverter,
#     ) -> None:
#         with pytest.raises(ValueError, match=r"The given input ImageList contains images of different sizes."):
#             input_conversion._data_conversion_output(
#                 input_data=_MultiSizeImageList(),
#                 output_data=torch.empty(1),
#             )
#
#     class TestEq:
#         @pytest.mark.parametrize(
#             ("output_conversion_image1", "output_conversion_image2"),
#             [
#                 (_ImageToColumnConverter(ImageSize(1, 1, 1)), _ImageToColumnConverter(ImageSize(1, 1, 1))),
#                 (_ImageToTableConverter(ImageSize(1, 1, 1)), _ImageToTableConverter(ImageSize(1, 1, 1))),
#                 (_ImageToImageConverter(ImageSize(1, 1, 1)), _ImageToImageConverter(ImageSize(1, 1, 1))),
#             ],
#         )
#         def test_should_be_equal(
#             self,
#             output_conversion_image1: _ImageConverter,
#             output_conversion_image2: _ImageConverter,
#         ) -> None:
#             assert output_conversion_image1 == output_conversion_image2
#
#         def test_should_be_not_implemented(self) -> None:
#             output_conversion_image_to_image = _ImageToImageConverter(ImageSize(1, 1, 1))
#             output_conversion_image_to_table = _ImageToTableConverter(ImageSize(1, 1, 1))
#             output_conversion_image_to_column = _ImageToColumnConverter(ImageSize(1, 1, 1))
#             other = Table()
#             assert output_conversion_image_to_image.__eq__(other) is NotImplemented
#             assert output_conversion_image_to_image.__eq__(output_conversion_image_to_table) is NotImplemented
#             assert output_conversion_image_to_image.__eq__(output_conversion_image_to_column) is NotImplemented
#             assert output_conversion_image_to_table.__eq__(other) is NotImplemented
#             assert output_conversion_image_to_table.__eq__(output_conversion_image_to_image) is NotImplemented
#             assert output_conversion_image_to_table.__eq__(output_conversion_image_to_column) is NotImplemented
#             assert output_conversion_image_to_column.__eq__(other) is NotImplemented
#             assert output_conversion_image_to_column.__eq__(output_conversion_image_to_table) is NotImplemented
#             assert output_conversion_image_to_column.__eq__(output_conversion_image_to_image) is NotImplemented
#
#     class TestHash:
#         @pytest.mark.parametrize(
#             ("output_conversion_image1", "output_conversion_image2"),
#             [
#                 (_ImageToColumnConverter(ImageSize(1, 1, 1)), _ImageToColumnConverter(ImageSize(1, 1, 1))),
#                 (_ImageToTableConverter(ImageSize(1, 1, 1)), _ImageToTableConverter(ImageSize(1, 1, 1))),
#                 (_ImageToImageConverter(ImageSize(1, 1, 1)), _ImageToImageConverter(ImageSize(1, 1, 1))),
#             ],
#         )
#         def test_hash_should_be_equal(
#             self,
#             output_conversion_image1: _ImageConverter,
#             output_conversion_image2: _ImageConverter,
#         ) -> None:
#             assert hash(output_conversion_image1) == hash(output_conversion_image2)
#
#         def test_hash_should_not_be_equal(self) -> None:
#             output_conversion_image_to_image = _ImageToImageConverter(ImageSize(1, 1, 1))
#             output_conversion_image_to_table = _ImageToTableConverter(ImageSize(1, 1, 1))
#             output_conversion_image_to_column = _ImageToColumnConverter(ImageSize(1, 1, 1))
#             assert hash(output_conversion_image_to_image) != hash(output_conversion_image_to_table)
#             assert hash(output_conversion_image_to_image) != hash(output_conversion_image_to_column)
#             assert hash(output_conversion_image_to_table) != hash(output_conversion_image_to_column)
#
#     class TestSizeOf:
#         @pytest.mark.parametrize(
#             "output_conversion_image",
#             [
#                 _ImageToColumnConverter(ImageSize(1, 1, 1)),
#                 _ImageToTableConverter(ImageSize(1, 1, 1)),
#                 _ImageToImageConverter(ImageSize(1, 1, 1)),
#             ],
#         )
#         def test_should_size_be_greater_than_normal_object(
#             self,
#             output_conversion_image: _ImageConverter,
#         ) -> None:
#             assert sys.getsizeof(output_conversion_image) > sys.getsizeof(object())
#
#
# class TestInputConversionImageToTable:
#     def test_should_raise_if_column_names_not_set(self) -> None:
#         with pytest.raises(
#             ValueError,
#             match=r"The column_names are not set. The data can only be converted if the column_names are provided as `list\[str\]` in the kwargs.",
#         ):
#             _ImageToTableConverter(ImageSize(1, 1, 1))._data_conversion_output(
#                 input_data=_SingleSizeImageList(),
#                 output_data=torch.empty(1),
#             )
