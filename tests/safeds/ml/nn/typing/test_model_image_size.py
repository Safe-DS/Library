import sys
from typing import Any

import pytest
from safeds.data.image.containers import Image
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn.typing import ConstantImageSize, ModelImageSize, VariableImageSize
from torch.types import Device

from tests.helpers import (
    configure_test_with_device,
    get_devices,
    get_devices_ids,
    images_all,
    images_all_ids,
    plane_png_path,
    resolve_resource_path,
)


class TestFromImage:
    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    @pytest.mark.parametrize("resource_path", images_all(), ids=images_all_ids())
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_should_create(self, resource_path: str, device: Device, image_size_class: type[ModelImageSize]) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        expected_image_size = image_size_class(image.width, image.height, image.channel)
        assert image_size_class.from_image(image) == expected_image_size


class TestFromImageSize:
    def test_should_create(self) -> None:
        constant_image_size = ConstantImageSize(1, 2, 3)
        variable_image_size = VariableImageSize(1, 2, 3)
        assert isinstance(VariableImageSize.from_image_size(constant_image_size), VariableImageSize)
        assert isinstance(ConstantImageSize.from_image_size(variable_image_size), ConstantImageSize)


class TestEq:
    @pytest.mark.parametrize(
        ("image_size", "width", "height", "channel"),
        [
            (ConstantImageSize(1, 2, 3), 1, 2, 3),
            (VariableImageSize(1, 2, 3), 1, 2, 3),
        ],
    )
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_should_be_equal(
        self,
        image_size: ModelImageSize,
        width: int,
        height: int,
        channel: int,
        image_size_class: type[ModelImageSize],
    ) -> None:
        assert image_size == image_size_class(width, height, channel)

    @pytest.mark.parametrize(
        ("image_size", "width", "height", "channel"),
        [
            (ConstantImageSize(1, 2, 3), 3, 2, 1),
            (VariableImageSize(1, 2, 3), 3, 2, 1),
        ],
    )
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_should_not_be_equal(
        self,
        image_size: ModelImageSize,
        width: int,
        height: int,
        channel: int,
        image_size_class: type[ModelImageSize],
    ) -> None:
        assert image_size != image_size_class(width, height, channel)

    @pytest.mark.parametrize(
        ("image_size", "other"),
        [
            (ConstantImageSize(1, 2, 3), None),
            (ConstantImageSize(1, 2, 3), Image.from_file(resolve_resource_path(plane_png_path))),
            (VariableImageSize(1, 2, 3), None),
            (VariableImageSize(1, 2, 3), Image.from_file(resolve_resource_path(plane_png_path))),
        ],
        ids=["ConstantImageSize-None", "ConstantImageSize-Image", "VariableImageSize-None", "VariableImageSize-Image"],
    )
    def test_should_be_not_implemented(self, image_size: ModelImageSize, other: Any) -> None:
        assert image_size.__eq__(other) is NotImplemented


class TestHash:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_hash_should_be_equal(self, resource_path: str, image_size_class: type[ModelImageSize]) -> None:
        image = Image.from_file(resolve_resource_path(resource_path))
        image2 = Image.from_file(resolve_resource_path(resource_path))
        assert hash(image_size_class.from_image(image)) == hash(image_size_class.from_image(image2))

    @pytest.mark.parametrize(
        "image_size_class1",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    @pytest.mark.parametrize(
        "image_size_class2",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_hash_should_not_be_equal(
        self,
        image_size_class1: type[ModelImageSize],
        image_size_class2: type[ModelImageSize],
    ) -> None:
        assert hash(image_size_class1(1, 2, 3)) != hash(image_size_class2(3, 2, 1))

    def test_hash_should_not_be_equal_different_model_image_sizes(self) -> None:
        assert hash(ConstantImageSize(1, 2, 3)) != hash(VariableImageSize(1, 2, 3))


class TestSizeOf:
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_should_size_be_greater_than_normal_object(self, image_size_class: type[ModelImageSize]) -> None:
        assert sys.getsizeof(image_size_class(1, 2, 3)) >= sys.getsizeof(0) * 3


class TestStr:
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_string(self, image_size_class: type[ModelImageSize]) -> None:
        image_size = image_size_class(1, 2, 3)
        assert (
            str(image_size)
            == f"{image_size_class.__name__} | {image_size.width}x{image_size.height}x{image_size.channel} (WxHxC)"
        )


class TestProperties:
    @pytest.mark.parametrize("width", list(range(1, 5)))
    @pytest.mark.parametrize("height", list(range(1, 5)))
    @pytest.mark.parametrize("channel", [1, 3, 4])
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_width_height_channel(
        self,
        width: int,
        height: int,
        channel: int,
        image_size_class: type[ModelImageSize],
    ) -> None:
        image_size = image_size_class(width, height, channel)
        assert image_size.width == width
        assert image_size.height == height
        assert image_size.channel == channel

    @pytest.mark.parametrize("channel", [2, 5, 6])
    @pytest.mark.parametrize(
        "image_size_class",
        [
            ConstantImageSize,
            VariableImageSize,
        ],
        ids=["ConstantImageSize", "VariableImageSize"],
    )
    def test_should_ignore_invalid_channel(self, channel: int, image_size_class: type[ModelImageSize]) -> None:
        assert image_size_class(1, 1, channel, _ignore_invalid_channel=True).channel == channel


@pytest.mark.parametrize(
    "image_size_class",
    [
        ConstantImageSize,
        VariableImageSize,
    ],
    ids=["ConstantImageSize", "VariableImageSize"],
)
class TestErrors:
    @pytest.mark.parametrize("width", [-1, 0])
    def test_should_raise_invalid_width(self, width: int, image_size_class: type[ModelImageSize]) -> None:
        with pytest.raises(OutOfBoundsError):
            image_size_class(width, 1, 1)

    @pytest.mark.parametrize("height", [-1, 0])
    def test_should_raise_invalid_height(self, height: int, image_size_class: type[ModelImageSize]) -> None:
        with pytest.raises(OutOfBoundsError):
            image_size_class(1, height, 1)

    @pytest.mark.parametrize("channel", [-1, 0, 2, 5])
    def test_should_raise_invalid_channel(self, channel: int, image_size_class: type[ModelImageSize]) -> None:
        with pytest.raises(ValueError, match=rf"Channel {channel} is not a valid channel option. Use either 1, 3 or 4"):
            image_size_class(1, 1, channel)

    @pytest.mark.parametrize("channel", [-1, 0])
    def test_should_raise_negative_channel_ignore_invalid_channel(
        self,
        channel: int,
        image_size_class: type[ModelImageSize],
    ) -> None:
        with pytest.raises(OutOfBoundsError):
            image_size_class(1, 1, channel, _ignore_invalid_channel=True)
