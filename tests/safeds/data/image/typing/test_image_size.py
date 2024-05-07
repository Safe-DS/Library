import sys
from typing import Any

import pytest
from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageSize
from safeds.exceptions import OutOfBoundsError
from torch.types import Device

from tests.helpers import (
    get_devices,
    get_devices_ids,
    images_all,
    images_all_ids,
    plane_png_path,
    resolve_resource_path,
    configure_test_with_device,
)


class TestFromImage:

    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    @pytest.mark.parametrize("resource_path", images_all(), ids=images_all_ids())
    def test_should_create(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        expected_image_size = ImageSize(image.width, image.height, image.channel)
        assert ImageSize.from_image(image) == expected_image_size


class TestEq:

    @pytest.mark.parametrize(("image_size", "width", "height", "channel"), [(ImageSize(1, 2, 3), 1, 2, 3)])
    def test_should_be_equal(self, image_size: ImageSize, width: int, height: int, channel: int) -> None:
        assert image_size == ImageSize(width, height, channel)

    @pytest.mark.parametrize(("image_size", "width", "height", "channel"), [(ImageSize(1, 2, 3), 3, 2, 1)])
    def test_should_not_be_equal(self, image_size: ImageSize, width: int, height: int, channel: int) -> None:
        assert image_size != ImageSize(width, height, channel)

    @pytest.mark.parametrize(
        ("image_size", "other"),
        [
            (ImageSize(1, 2, 3), None),
            (ImageSize(1, 2, 3), Image.from_file(resolve_resource_path(plane_png_path))),
        ],
        ids=["None", "Image"],
    )
    def test_should_be_not_implemented(self, image_size: ImageSize, other: Any) -> None:
        assert image_size.__eq__(other) is NotImplemented


class TestHash:

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_hash_should_be_equal(self, resource_path: str) -> None:
        image = Image.from_file(resolve_resource_path(resource_path))
        image2 = Image.from_file(resolve_resource_path(resource_path))
        assert hash(ImageSize.from_image(image)) == hash(ImageSize.from_image(image2))

    def test_hash_should_not_be_equal(self) -> None:
        assert hash(ImageSize(1, 2, 3)) != hash(ImageSize(3, 2, 1))


class TestSizeOf:

    @pytest.mark.parametrize("image_size", [ImageSize(1, 2, 3)])
    def test_should_size_be_greater_than_normal_object(self, image_size: ImageSize) -> None:
        assert sys.getsizeof(image_size) >= sys.getsizeof(0) * 3


class TestStr:

    @pytest.mark.parametrize("image_size", [ImageSize(1, 2, 3)])
    def test_should_size_be_greater_than_normal_object(self, image_size: ImageSize) -> None:
        assert str(image_size) == f"{image_size.width}x{image_size.height}x{image_size.channel} (WxHxC)"


class TestProperties:

    @pytest.mark.parametrize("width", list(range(1, 5)))
    @pytest.mark.parametrize("height", list(range(1, 5)))
    @pytest.mark.parametrize("channel", [1, 3, 4])
    def test_width_height_channel(self, width: int, height: int, channel: int) -> None:
        image_size = ImageSize(width, height, channel)
        assert image_size.width == width
        assert image_size.height == height
        assert image_size.channel == channel

    @pytest.mark.parametrize("channel", [2, 5, 6])
    def test_should_ignore_invalid_channel(self, channel: int) -> None:
        assert ImageSize(1, 1, channel, _ignore_invalid_channel=True).channel == channel


class TestErrors:

    @pytest.mark.parametrize("width", [-1, 0])
    def test_should_raise_invalid_width(self, width: int) -> None:
        with pytest.raises(OutOfBoundsError, match=rf"{width} is not inside \[1, \u221e\)."):
            ImageSize(width, 1, 1)

    @pytest.mark.parametrize("height", [-1, 0])
    def test_should_raise_invalid_height(self, height: int) -> None:
        with pytest.raises(OutOfBoundsError, match=rf"{height} is not inside \[1, \u221e\)."):
            ImageSize(1, height, 1)

    @pytest.mark.parametrize("channel", [-1, 0, 2, 5])
    def test_should_raise_invalid_channel(self, channel: int) -> None:
        with pytest.raises(ValueError, match=rf"Channel {channel} is not a valid channel option. Use either 1, 3 or 4"):
            ImageSize(1, 1, channel)

    @pytest.mark.parametrize("channel", [-1, 0])
    def test_should_raise_negative_channel_ignore_invalid_channel(self, channel: int) -> None:
        with pytest.raises(OutOfBoundsError, match=rf"channel \(={channel}\) is not inside \[1, \u221e\)."):
            ImageSize(1, 1, channel, _ignore_invalid_channel=True)
