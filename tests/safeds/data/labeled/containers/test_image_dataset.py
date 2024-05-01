import math
from typing import TypeVar

import pytest
import torch
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._empty_image_list import _EmptyImageList
from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _ColumnAsTensor, _TableAsTensor
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import (
    IndexOutOfBoundsError,
    NonNumericColumnError,
    OutOfBoundsError,
    OutputLengthMismatchError,
    TransformerNotFittedError,
)
from torch import Tensor

from tests.helpers import images_all, plane_png_path, resolve_resource_path, white_square_png_path

T = TypeVar("T", Column, Table, ImageList)


class TestImageDatasetInit:

    @pytest.mark.parametrize(
        ("input_data", "output_data", "error", "error_msg"),
        [
            (
                _MultiSizeImageList(),
                Table(),
                ValueError,
                r"The given input ImageList contains images of different sizes.",
            ),
            (_EmptyImageList(), Table(), ValueError, r"The given input ImageList contains no images."),
            (
                ImageList.from_files(resolve_resource_path([plane_png_path, plane_png_path])),
                ImageList.from_files(resolve_resource_path([plane_png_path, white_square_png_path])),
                ValueError,
                r"The given output ImageList contains images of different sizes.",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                _EmptyImageList(),
                OutputLengthMismatchError,
                r"The length of the output container differs",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                Table(),
                OutputLengthMismatchError,
                r"The length of the output container differs",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                Column("column", [1, 2]),
                OutputLengthMismatchError,
                r"The length of the output container differs",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                ImageList.from_files(resolve_resource_path([plane_png_path, plane_png_path])),
                OutputLengthMismatchError,
                r"The length of the output container differs",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                Table({"a": ["1"]}),
                NonNumericColumnError,
                r"Tried to do a numerical operation on one or multiple non-numerical columns: \nColumns \['a'\] are not numerical.",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                Table({"a": [2]}),
                ValueError,
                r"Columns \['a'\] have values outside of the interval \[0, 1\].",
            ),
            (
                ImageList.from_files(resolve_resource_path(plane_png_path)),
                Table({"a": [-1]}),
                ValueError,
                r"Columns \['a'\] have values outside of the interval \[0, 1\].",
            ),
        ],
    )
    def test_should_raise_with_invalid_data(
        self,
        input_data: ImageList,
        output_data: T,
        error: type[Exception],
        error_msg: str,
    ) -> None:
        with pytest.raises(error, match=error_msg):
            ImageDataset(input_data, output_data)


class TestLength:

    def test_should_return_length(self) -> None:
        image_dataset = ImageDataset(ImageList.from_files(resolve_resource_path(plane_png_path)), Column("images", [1]))
        assert len(image_dataset) == 1


class TestShuffle:

    def test_should_be_different_order(self) -> None:
        torch.manual_seed(1234)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        image_dataset_shuffled = image_dataset.shuffle()
        batch = image_dataset._get_batch(0, len(image_dataset))
        batch_shuffled = image_dataset_shuffled._get_batch(0, len(image_dataset))
        assert not torch.all(torch.eq(batch[0], batch_shuffled[0]))
        assert not torch.all(torch.eq(batch[1], batch_shuffled[1]))


class TestBatch:

    @pytest.mark.parametrize(
        ("batch_number", "batch_size"),
        [
            (-1, len(images_all())),
            (1, len(images_all())),
            (2, math.ceil(len(images_all()) / 2)),
            (3, math.ceil(len(images_all()) / 3)),
            (4, math.ceil(len(images_all()) / 4)),
        ],
    )
    def test_should_raise_index_out_of_bounds_error(self, batch_number: int, batch_size: int) -> None:
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        with pytest.raises(IndexOutOfBoundsError):
            image_dataset._get_batch(batch_number, batch_size)

    def test_should_raise_out_of_bounds_error(self) -> None:
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        with pytest.raises(OutOfBoundsError):
            image_dataset._get_batch(0, -1)


class TestTableAsTensor:

    def test_should_raise_if_not_one_hot_encoded(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The given table is not correctly one hot encoded as it contains rows that have a sum not equal to 1.",
        ):
            _TableAsTensor(Table({"a": [0.2, 0.2, 0.2, 0.3, 0.2]}))

    @pytest.mark.parametrize(
        ("tensor", "error_msg"),
        [
            (torch.randn(10), r"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got 1."),
            (torch.randn(10, 10, 10), r"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got 3."),
            (torch.randn(10, 10), r"Tensor and column_names have different amounts of classes \(10!=2\)."),
        ],
    )
    def test_should_raise_from_tensor(self, tensor: Tensor, error_msg: str) -> None:
        with pytest.raises(ValueError, match=error_msg):
            _TableAsTensor._from_tensor(tensor, ["a", "b"])


class TestColumnAsTensor:

    @pytest.mark.parametrize(
        ("tensor", "one_hot_encoder", "error", "error_msg"),
        [
            (
                torch.randn(10),
                OneHotEncoder(),
                ValueError,
                r"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got 1.",
            ),
            (
                torch.randn(10, 10, 10),
                OneHotEncoder(),
                ValueError,
                r"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got 3.",
            ),
            (torch.randn(10, 10), OneHotEncoder(), TransformerNotFittedError, r""),
            (
                torch.randn(10, 10),
                OneHotEncoder().fit(Table({"b": ["a", "b", "c"]}), None),
                ValueError,
                r"Tensor and one_hot_encoder have different amounts of classes \(10!=3\).",
            ),
        ],
    )
    def test_should_raise_from_tensor(
        self,
        tensor: Tensor,
        one_hot_encoder: OneHotEncoder,
        error: type[Exception],
        error_msg: str,
    ) -> None:
        with pytest.raises(error, match=error_msg):
            _ColumnAsTensor._from_tensor(tensor, "a", one_hot_encoder)
