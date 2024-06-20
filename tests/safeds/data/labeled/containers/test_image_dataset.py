import math
import sys
import warnings
from typing import TypeVar

import pytest
import torch
from safeds._config import _get_device
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
from torch.types import Device

from tests.helpers import (
    configure_test_with_device,
    get_devices,
    get_devices_ids,
    images_all,
    plane_png_path,
    resolve_resource_path,
    white_square_png_path,
)

T = TypeVar("T", Column, Table, ImageList)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
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
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        with pytest.raises(error, match=error_msg):
            ImageDataset(input_data, output_data)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestLength:
    def test_should_return_length(self, device: Device) -> None:
        configure_test_with_device(device)
        image_dataset = ImageDataset(ImageList.from_files(resolve_resource_path(plane_png_path)), Column("images", [1]))
        assert len(image_dataset) == 1
        assert image_dataset._input._tensor.device == _get_device()
        assert image_dataset._output._tensor.device == _get_device()


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestEq:
    @pytest.mark.parametrize(
        "image_dataset_output",
        [
            Column("images", [1]),
            Table({"images": [1]}),
            plane_png_path,
        ],
    )
    def test_should_be_equal(self, image_dataset_output: str | Column | Table, device: Device) -> None:
        configure_test_with_device(device)
        image_dataset1 = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset_output))
                if isinstance(image_dataset_output, str)
                else image_dataset_output
            ),
        )  # type: ignore[type-var]
        image_dataset2 = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset_output))
                if isinstance(image_dataset_output, str)
                else image_dataset_output
            ),
        )  # type: ignore[type-var]
        assert image_dataset1 is not image_dataset2
        assert image_dataset1 == image_dataset2
        assert image_dataset1._input._tensor.device == _get_device()
        assert image_dataset1._output._tensor.device == _get_device()
        assert image_dataset2._input._tensor.device == _get_device()
        assert image_dataset2._output._tensor.device == _get_device()

    @pytest.mark.parametrize(
        "image_dataset1_output",
        [
            Column("images", [1]),
            Table({"images": [1]}),
            plane_png_path,
        ],
    )
    @pytest.mark.parametrize(
        ("image_dataset2_input", "image_dataset2_output"),
        [
            (plane_png_path, Column("ims", [1])),
            (plane_png_path, Table({"ims": [1]})),
            (plane_png_path, Column("images", [0])),
            (plane_png_path, Table({"images": [0], "others": [1]})),
            (plane_png_path, white_square_png_path),
            (white_square_png_path, Column("images", [1])),
            (white_square_png_path, Table({"images": [1]})),
            (white_square_png_path, plane_png_path),
        ],
    )
    def test_should_not_be_equal(
        self,
        image_dataset1_output: str | Column | Table,
        image_dataset2_input: str,
        image_dataset2_output: str | Column | Table,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image_dataset1 = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset1_output))
                if isinstance(image_dataset1_output, str)
                else image_dataset1_output
            ),
        )  # type: ignore[type-var]
        image_dataset2 = ImageDataset(
            ImageList.from_files(resolve_resource_path(image_dataset2_input)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset2_output))
                if isinstance(image_dataset2_output, str)
                else image_dataset2_output
            ),
        )  # type: ignore[type-var]
        assert image_dataset1 != image_dataset2
        assert image_dataset1._input._tensor.device == _get_device()
        assert image_dataset1._output._tensor.device == _get_device()
        assert image_dataset2._input._tensor.device == _get_device()
        assert image_dataset2._output._tensor.device == _get_device()

    def test_should_be_not_implemented(self, device: Device) -> None:
        configure_test_with_device(device)
        image_dataset = ImageDataset(ImageList.from_files(resolve_resource_path(plane_png_path)), Column("images", [1]))
        other = Table()
        assert image_dataset.__eq__(other) is NotImplemented


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestHash:
    @pytest.mark.parametrize(
        "image_dataset_output",
        [
            Column("images", [1]),
            Table({"images": [1]}),
            plane_png_path,
        ],
    )
    def test_hash_should_be_equal(self, image_dataset_output: str | Column | Table, device: Device) -> None:
        configure_test_with_device(device)
        image_dataset1 = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset_output))
                if isinstance(image_dataset_output, str)
                else image_dataset_output
            ),
        )  # type: ignore[type-var]
        image_dataset2 = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset_output))
                if isinstance(image_dataset_output, str)
                else image_dataset_output
            ),
        )  # type: ignore[type-var]
        assert image_dataset1 is not image_dataset2
        assert hash(image_dataset1) == hash(image_dataset2)
        assert image_dataset1._input._tensor.device == _get_device()
        assert image_dataset1._output._tensor.device == _get_device()
        assert image_dataset2._input._tensor.device == _get_device()
        assert image_dataset2._output._tensor.device == _get_device()

    @pytest.mark.parametrize(
        "image_dataset1_output",
        [
            Column("images", [1]),
            Table({"images": [1]}),
            plane_png_path,
        ],
    )
    @pytest.mark.parametrize(
        ("image_dataset2_input", "image_dataset2_output"),
        [
            (plane_png_path, Column("ims", [1])),
            (plane_png_path, Table({"ims": [1]})),
            (plane_png_path, Column("images", [0])),
            (plane_png_path, Table({"images": [0], "others": [1]})),
            (plane_png_path, white_square_png_path),
            (white_square_png_path, Column("images", [1])),
            (white_square_png_path, Table({"images": [1]})),
            (white_square_png_path, plane_png_path),
        ],
    )
    def test_hash_should_not_be_equal(
        self,
        image_dataset1_output: str | Column | Table,
        image_dataset2_input: str,
        image_dataset2_output: str | Column | Table,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image_dataset1 = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset1_output))
                if isinstance(image_dataset1_output, str)
                else image_dataset1_output
            ),
        )  # type: ignore[type-var]
        image_dataset2 = ImageDataset(
            ImageList.from_files(resolve_resource_path(image_dataset2_input)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset2_output))
                if isinstance(image_dataset2_output, str)
                else image_dataset2_output
            ),
        )  # type: ignore[type-var]
        assert hash(image_dataset1) != hash(image_dataset2)
        assert image_dataset1._input._tensor.device == _get_device()
        assert image_dataset1._output._tensor.device == _get_device()
        assert image_dataset2._input._tensor.device == _get_device()
        assert image_dataset2._output._tensor.device == _get_device()


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestSizeOf:
    @pytest.mark.parametrize(
        "image_dataset_output",
        [
            Column("images", [1]),
            Table({"images": [1]}),
            plane_png_path,
        ],
    )
    def test_should_size_be_greater_than_normal_object(
        self,
        image_dataset_output: str | Column | Table,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image_dataset = ImageDataset(
            ImageList.from_files(resolve_resource_path(plane_png_path)),
            (
                ImageList.from_files(resolve_resource_path(image_dataset_output))
                if isinstance(image_dataset_output, str)
                else image_dataset_output
            ),
        )  # type: ignore[type-var]
        assert sys.getsizeof(image_dataset) > sys.getsizeof(object())
        assert image_dataset._input._tensor.device == _get_device()
        assert image_dataset._output._tensor.device == _get_device()


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestShuffle:
    def test_should_be_different_order(self, device: Device) -> None:
        configure_test_with_device(device)
        torch.manual_seed(1234)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        image_dataset_shuffled = image_dataset.shuffle()
        batch = image_dataset._get_batch(0, len(image_dataset))
        batch_shuffled = image_dataset_shuffled._get_batch(0, len(image_dataset))
        assert not torch.all(torch.eq(batch[0], batch_shuffled[0]))
        assert not torch.all(torch.eq(batch[1], batch_shuffled[1]))
        assert image_dataset._input._tensor.device == _get_device()
        assert image_dataset._output._tensor.device == _get_device()


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
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
    def test_should_raise_index_out_of_bounds_error(self, batch_number: int, batch_size: int, device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        with pytest.raises(IndexOutOfBoundsError):
            image_dataset._get_batch(batch_number, batch_size)

    def test_should_raise_out_of_bounds_error(self, device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        with pytest.raises(OutOfBoundsError):
            image_dataset._get_batch(0, -1)

    def test_get_batch_device(self, device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        batch = image_dataset._get_batch(0)
        assert batch[0].device == _get_device()
        assert batch[1].device == _get_device()


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
@pytest.mark.parametrize(
    "shuffle",
    [
        True,
        False
    ]
)
class TestSplit:

    @pytest.mark.parametrize(
        "output",
        [
            Column("images", images_all()[:4] + images_all()[5:]),
            Table({"0": [1, 0, 0, 0, 0, 0], "1": [0, 1, 0, 0, 0, 0], "2": [0, 0, 1, 0, 0, 0], "3": [0, 0, 0, 1, 0, 0], "4": [0, 0, 0, 0, 1, 0], "5": [0, 0, 0, 0, 0, 1]}),
            _EmptyImageList(),
        ],
        ids=["Column", "Table", "ImageList"]
    )
    def test_should_split(self, device: Device, shuffle: bool, output: Column | Table | ImageList) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).remove_duplicate_images().resize(10, 10)
        if isinstance(output, _EmptyImageList):
            output = image_list
        image_dataset = ImageDataset(image_list, output)  # type: ignore[type-var]
        image_dataset1, image_dataset2 = image_dataset.split(0.4, shuffle=shuffle)
        offset = len(image_dataset1)
        assert len(image_dataset1) == round(0.4 * len(image_dataset))
        assert len(image_dataset2) == len(image_dataset) - offset
        assert len(image_dataset1.get_input()) == round(0.4 * len(image_dataset))
        assert len(image_dataset2.get_input()) == len(image_dataset) - offset
        if isinstance(image_dataset1.get_output(), Table):
            assert image_dataset1.get_output().row_count == round(0.4 * len(image_dataset))
        else:
            assert len(image_dataset1.get_output()) == round(0.4 * len(image_dataset))
        if isinstance(image_dataset2.get_output(), Table):
            assert image_dataset2.get_output().row_count == len(image_dataset) - offset
        else:
            assert len(image_dataset2.get_output()) == len(image_dataset) - offset

        assert image_dataset != image_dataset1
        assert image_dataset != image_dataset2
        assert image_dataset1 != image_dataset2

        for i, image in enumerate(image_dataset1.get_input().to_images()):
            index = image_list.index(image)[0]
            if not shuffle:
                assert index == i
            out = image_dataset1.get_output()
            if isinstance(out, ImageList):
                assert image_list.index(out.get_image(i))[0] == index
            elif isinstance(out, Column):
                assert output.to_list().index(out.to_list()[i]) == index
            elif isinstance(out, Table):
                assert output.get_column(str(index)).to_list()[index] == 1

        for i, image in enumerate(image_dataset2.get_input().to_images()):
            index = image_list.index(image)[0]
            if not shuffle:
                assert index == i + offset
            out = image_dataset2.get_output()
            if isinstance(out, ImageList):
                assert image_list.index(out.get_image(i))[0] == index
            elif isinstance(out, Column):
                assert output.to_list().index(out.to_list()[i]) == index
            elif isinstance(out, Table):
                assert output.get_column(str(index)).to_list()[index] == 1

        image_dataset._batch_size = len(image_dataset)
        image_dataset1._batch_size = 1
        image_dataset2._batch_size = 1
        image_dataset_batch = next(iter(image_dataset))

        for i, b in enumerate(iter(image_dataset1)):
            assert b[0] in image_dataset_batch[0]
            index = (b[0] == image_dataset_batch[0]).all(dim=[1, 2, 3]).nonzero()[0][0]
            if not shuffle:
                assert index == i
            assert torch.all(torch.eq(b[0], image_dataset_batch[0][index]))
            assert torch.all(torch.eq(b[1], image_dataset_batch[1][index]))

        for i, b in enumerate(iter(image_dataset2)):
            assert b[0] in image_dataset_batch[0]
            index = (b[0] == image_dataset_batch[0]).all(dim=[1, 2, 3]).nonzero()[0][0]
            if not shuffle:
                assert index == i + offset
            assert torch.all(torch.eq(b[0], image_dataset_batch[0][index]))
            assert torch.all(torch.eq(b[1], image_dataset_batch[1][index]))

    @pytest.mark.parametrize(
        "percentage",
        [-1, -0.1, 1.1, 2]
    )
    def test_should_raise(self, device: Device, shuffle: bool, percentage: float) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(images_all())).resize(10, 10)
        image_dataset = ImageDataset(image_list, Column("images", images_all()))
        with pytest.raises(OutOfBoundsError):
            image_dataset.split(percentage, shuffle=shuffle)



@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestTableAsTensor:
    def test_should_raise_if_not_one_hot_encoded(self, device: Device) -> None:
        configure_test_with_device(device)
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
    def test_should_raise_from_tensor(self, tensor: Tensor, error_msg: str, device: Device) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(_get_device())
        with pytest.raises(ValueError, match=error_msg):
            _TableAsTensor._from_tensor(tensor, ["a", "b"])

    def test_eq_should_be_not_implemented(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _TableAsTensor(Table()).__eq__(Table()) is NotImplemented


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
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
                OneHotEncoder().fit(Table({"b": ["a", "b", "c"]})),
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
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(_get_device())
        with pytest.raises(error, match=error_msg):
            _ColumnAsTensor._from_tensor(tensor, "a", one_hot_encoder)

    def test_eq_should_be_not_implemented(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _ColumnAsTensor(Column("column", [1])).__eq__(Table()) is NotImplemented

    def test_should_not_warn(self, device: Device) -> None:
        configure_test_with_device(device)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            _ColumnAsTensor(Column("column", [1, 2, 3]))
