import sys
import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import PIL.Image
import pytest
import torch

from safeds._config import _get_device
from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Table
from safeds.exceptions import IllegalFormatError, OutOfBoundsError
from syrupy import SnapshotAssertion
from torch.types import Device

from tests.helpers import (
    configure_test_with_device,
    get_devices,
    get_devices_ids,
    grayscale_jpg_id,
    grayscale_jpg_path,
    grayscale_png_id,
    grayscale_png_path,
    images_all,
    images_all_ids,
    images_asymmetric,
    images_asymmetric_ids,
    os_mac,
    plane_jpg_id,
    plane_jpg_path,
    plane_png_id,
    plane_png_path,
    resolve_resource_path,
    rgba_png_id,
    rgba_png_path,
    skip_if_os,
    white_square_jpg_id,
    white_square_jpg_path,
    white_square_png_id,
    white_square_png_path, device_cpu, device_cuda,
)


def _assert_width_height_channel(image1: Image, image2: Image) -> None:
    assert image1.width == image2.width
    assert image1.height == image2.height
    assert image1.channel == image2.channel


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestFromFile:
    @pytest.mark.parametrize(
        "resource_path",
        [*images_all(), *[Path(image_path) for image_path in images_all()]],
        ids=[
            *["file-" + image_id for image_id in images_all_ids()],
            *["path-" + image_id for image_id in images_all_ids()],
        ],
    )
    def test_should_load_from_file(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        assert image != Image(torch.empty(1, 1, 1))

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/missing_file.jpg",
            Path("image/missing_file.jpg"),
            "image/missing_file.png",
            Path("image/missing_file.png"),
        ],
        ids=["missing_file_jpg", "missing_file_jpg_Path", "missing_file_png", "missing_file_png_Path"],
    )
    def test_should_raise_if_file_not_found(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(FileNotFoundError):
            Image.from_file(resolve_resource_path(resource_path))


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestFromBytes:
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, white_square_jpg_path, white_square_png_path, grayscale_jpg_path, grayscale_png_path],
        ids=[plane_jpg_id, white_square_jpg_id, white_square_png_id, grayscale_jpg_id, grayscale_png_id],
    )
    def test_should_write_and_load_bytes_jpeg(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_copy = Image.from_bytes(typing.cast(bytes, image._repr_jpeg_()))
        _assert_width_height_channel(image, image_copy)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_write_and_load_bytes_png(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_copy = Image.from_bytes(image._repr_png_())
        assert image == image_copy


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestToNumpyArray:

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_numpy_array(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image_safeds = Image.from_file(resolve_resource_path(resource_path))
        image_np = np.array(PIL.Image.open(resolve_resource_path(resource_path)))
        assert np.all(np.array(image_safeds).squeeze() == image_np)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestReprJpeg:
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, white_square_jpg_path, white_square_png_path, grayscale_jpg_path, grayscale_png_path],
        ids=[plane_jpg_id, white_square_jpg_id, white_square_png_id, grayscale_jpg_id, grayscale_png_id],
    )
    def test_should_return_bytes(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        assert isinstance(image._repr_jpeg_(), bytes)

    @pytest.mark.parametrize(
        "resource_path",
        [
            plane_png_path,
            rgba_png_path,
        ],
        ids=[plane_png_id, rgba_png_id],
    )
    def test_should_return_none_if_image_has_alpha_channel(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        assert image._repr_jpeg_() is None


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestReprPng:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_bytes(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        assert isinstance(image._repr_png_(), bytes)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestToJpegFile:
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, white_square_jpg_path, white_square_png_path, grayscale_jpg_path, grayscale_png_path],
        ids=[plane_jpg_id, white_square_jpg_id, white_square_png_id, grayscale_jpg_id, grayscale_png_id],
    )
    def test_should_save_file(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with NamedTemporaryFile(suffix=".jpg") as tmp_jpeg_file:
            tmp_jpeg_file.close()
            with Path(tmp_jpeg_file.name).open("w", encoding="utf-8") as tmp_file:
                image.to_jpeg_file(tmp_file.name)
            with Path(tmp_jpeg_file.name).open("r", encoding="utf-8") as tmp_file:
                image_r = Image.from_file(tmp_file.name)
        _assert_width_height_channel(image, image_r)

    @pytest.mark.parametrize(
        "resource_path",
        [
            plane_png_path,
            rgba_png_path,
        ],
        ids=[plane_png_id, rgba_png_id],
    )
    def test_should_raise_if_image_has_alpha_channel(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with NamedTemporaryFile(suffix=".jpg") as tmp_jpeg_file:
            tmp_jpeg_file.close()
            with Path(tmp_jpeg_file.name).open("w", encoding="utf-8") as tmp_file, pytest.raises(
                IllegalFormatError,
                match=r"This format is illegal. Use one of the following formats: png",
            ):
                image.to_jpeg_file(tmp_file.name)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestToPngFile:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_save_file(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with NamedTemporaryFile(suffix=".png") as tmp_png_file:
            tmp_png_file.close()
            with Path(tmp_png_file.name).open("w", encoding="utf-8") as tmp_file:
                image.to_png_file(tmp_file.name)
            with Path(tmp_png_file.name).open("r", encoding="utf-8") as tmp_file:
                image_r = Image.from_file(tmp_file.name)
        assert image == image_r


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestProperties:
    @pytest.mark.parametrize(
        ("resource_path", "width", "height", "channel"),
        [
            (
                white_square_jpg_path,
                1,
                1,
                3,
            ),
            (
                white_square_png_path,
                1,
                1,
                3,
            ),
            (
                plane_jpg_path,
                568,
                320,
                3,
            ),
            (
                plane_png_path,
                568,
                320,
                4,
            ),
            (
                rgba_png_path,
                7,
                5,
                4,
            ),
            (
                grayscale_jpg_path,
                16,
                16,
                1,
            ),
            (
                grayscale_png_path,
                16,
                16,
                1,
            ),
        ],
        ids=[
            "[3,1,1]" + white_square_jpg_id,
            "[3,1,1]" + white_square_png_id,
            "[4,568,320]" + plane_jpg_id,
            "[3,568,320]" + plane_png_id,
            "[4,568,320]" + rgba_png_id,
            "[1,16,16]" + grayscale_jpg_id,
            "[1,16,16]" + grayscale_png_id,
        ],
    )
    def test_should_return_image_properties(
        self,
        resource_path: str,
        width: int,
        height: int,
        channel: int,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        assert image.width == width
        assert image.height == height
        assert image.channel == channel
        assert image.size == ImageSize(width, height, channel)


class TestEQ:
    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_be_equal(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image2 = Image.from_file(resolve_resource_path(resource_path))
        assert image == image2

    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    def test_should_not_be_equal(self, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(plane_png_path))
        image2 = Image.from_file(resolve_resource_path(white_square_png_path))
        assert image != image2

    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_be_not_implemented(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        other = Table()
        assert (image.__eq__(other)) is NotImplemented


class TestHash:
    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_hash_be_equal(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image2 = Image.from_file(resolve_resource_path(resource_path))
        assert hash(image) == hash(image2)

    @pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
    def test_should_hash_not_be_equal(self, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(plane_png_path))
        image2 = Image.from_file(resolve_resource_path(white_square_png_path))
        assert hash(image) != hash(image2)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestChangeChannel:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize("channel", [1, 3, 4], ids=["to-gray-1-channel", "to-rgb-3-channel", "to-rgba-4-channel"])
    def test_should_change_channel(
        self,
        resource_path: str,
        channel: int,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        new_image = image.change_channel(channel)
        assert new_image.channel == channel
        assert new_image == snapshot_png_image

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize("channel", [2], ids=["invalid-channel"])
    def test_should_raise(self, resource_path: str, channel: int, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(ValueError, match=rf"Channel {channel} is not a valid channel option. Use either 1, 3 or 4"):
            image.change_channel(channel)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestResize:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize(
        ("new_width", "new_height"),
        [
            (
                2,
                3,
            ),
            (
                50,
                75,
            ),
            (
                700,
                400,
            ),
        ],
        ids=["(2, 3)", "(50, 75)", "(700, 400)"],
    )
    def test_should_return_resized_image(
        self,
        resource_path: str,
        new_width: int,
        new_height: int,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        new_image = image.resize(new_width, new_height)
        assert new_image.width == new_width
        assert new_image.height == new_height
        assert image.channel == new_image.channel
        assert image != new_image
        assert new_image == snapshot_png_image

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize(
        ("new_width", "new_height"),
        [(-10, 10), (10, -10), (-10, -10)],
        ids=["invalid width", "invalid height", "invalid width and height"],
    )
    def test_should_raise(self, resource_path: str, new_width: int, new_height: int, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(
            OutOfBoundsError,
            match=rf"At least one of the new sizes new_width and new_height \(={min(new_width, new_height)}\) is not inside \[1, \u221e\).",
        ):
            image.resize(new_width, new_height)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestConvertToGrayscale:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_convert_to_grayscale(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        grayscale_image = image.convert_to_grayscale()
        assert grayscale_image == snapshot_png_image
        _assert_width_height_channel(image, grayscale_image)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestCrop:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_cropped_image(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_cropped = image.crop(0, 0, 100, 100)
        assert image_cropped == snapshot_png_image
        assert image_cropped.channel == image.channel

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize(
        ("new_width", "new_height"),
        [(-10, 1), (1, -10), (-10, -1)],
        ids=["invalid width", "invalid height", "invalid width and height"],
    )
    def test_should_raise_invalid_size(
        self,
        resource_path: str,
        new_width: int,
        new_height: int,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(
            OutOfBoundsError,
            match=rf"At least one of width and height \(={min(new_width, new_height)}\) is not inside \[1, \u221e\).",
        ):
            image.crop(0, 0, new_width, new_height)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize(
        ("new_x", "new_y"),
        [(-10, 1), (1, -10), (-10, -1)],
        ids=["invalid x", "invalid y", "invalid x and y"],
    )
    def test_should_raise_invalid_coordinates(self, resource_path: str, new_x: int, new_y: int, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(
            OutOfBoundsError,
            match=rf"At least one of the coordinates x and y \(={min(new_x, new_y)}\) is not inside \[0, \u221e\).",
        ):
            image.crop(new_x, new_y, 100, 100)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    @pytest.mark.parametrize(
        ("new_x", "new_y"),
        [(10000, 1), (1, 10000), (10000, 10000)],
        ids=["outside x", "outside y", "outside x and y"],
    )
    def test_should_warn_if_coordinates_outsize_image(
        self,
        resource_path: str,
        new_x: int,
        new_y: int,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_blank_tensor = torch.zeros((image.channel, 1, 1), device=device)
        with pytest.warns(
            UserWarning,
            match=r"The specified bounding rectangle does not contain any content of the image. Therefore the image will be blank.",
        ):
            cropped_image = image.crop(new_x, new_y, 1, 1)
            assert torch.all(torch.eq(cropped_image._image_tensor, image_blank_tensor))


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestFlipVertically:
    @pytest.mark.parametrize(
        "resource_path",
        images_asymmetric(),
        ids=images_asymmetric_ids(),
    )
    def test_should_flip_vertically(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_flip_v = image.flip_vertically()
        assert image != image_flip_v
        assert image_flip_v == snapshot_png_image
        _assert_width_height_channel(image, image_flip_v)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_be_original(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_flip_v_v = image.flip_vertically().flip_vertically()
        assert image == image_flip_v_v


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestFlipHorizontally:
    @pytest.mark.parametrize(
        "resource_path",
        images_asymmetric(),
        ids=images_asymmetric_ids(),
    )
    def test_should_flip_horizontally(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_flip_h = image.flip_horizontally()
        assert image != image_flip_h
        assert image_flip_h == snapshot_png_image
        _assert_width_height_channel(image, image_flip_h)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_be_original(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_flip_h_h = image.flip_horizontally().flip_horizontally()
        assert image == image_flip_h_h


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestBrightness:
    @pytest.mark.parametrize("factor", [0.5, 10], ids=["small factor", "large factor"])
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, plane_png_path, grayscale_png_path, grayscale_jpg_path],
        ids=[plane_jpg_id, plane_png_id, grayscale_png_id, grayscale_jpg_id],
    )
    def test_should_adjust_brightness(
        self,
        factor: float,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_adjusted_brightness = image.adjust_brightness(factor)
        assert image != image_adjusted_brightness
        assert image_adjusted_brightness == snapshot_png_image
        _assert_width_height_channel(image, image_adjusted_brightness)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_not_brighten(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(
            UserWarning,
            match="Brightness adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path))
            image2 = image.adjust_brightness(1)
            assert image == image2

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_raise(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
            image.adjust_brightness(-1)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestAddNoise:
    @pytest.mark.parametrize(
        "standard_deviation",
        [
            0.0,
            0.7,
            2.5,
        ],
        ids=["minimum noise", "some noise", "very noisy"],
    )
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_add_noise(
        self,
        resource_path: str,
        standard_deviation: float,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        skip_if_os([os_mac])
        torch.manual_seed(0)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_noise = image.add_noise(standard_deviation)
        assert image_noise == snapshot_png_image
        _assert_width_height_channel(image, image_noise)

    @pytest.mark.parametrize(
        "standard_deviation",
        [-1],
        ids=["sigma below zero"],
    )
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_raise_standard_deviation(
        self,
        resource_path: str,
        standard_deviation: float,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(
            OutOfBoundsError,
            match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
        ):
            image.add_noise(standard_deviation)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestAdjustContrast:
    @pytest.mark.parametrize("factor", [0.75, 5], ids=["small factor", "large factor"])
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, plane_png_path, grayscale_jpg_path, grayscale_png_path],
        ids=[plane_jpg_id, plane_png_id, grayscale_jpg_id, grayscale_png_id],
    )
    def test_should_adjust_contrast(
        self,
        factor: float,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_adjusted_contrast = image.adjust_contrast(factor)
        assert image != image_adjusted_contrast
        assert image_adjusted_contrast == snapshot_png_image
        _assert_width_height_channel(image, image_adjusted_contrast)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_not_adjust_contrast(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(
            UserWarning,
            match="Contrast adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path))
            image_adjusted_contrast = image.adjust_contrast(1)
            assert image == image_adjusted_contrast

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_raise_negative_contrast(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1.0\) is not inside \[0, \u221e\)."):
            Image.from_file(resolve_resource_path(resource_path)).adjust_contrast(-1.0)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestAdjustColor:
    @pytest.mark.parametrize("factor", [2, 0.5, 0], ids=["add color", "remove color", "gray"])
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, plane_png_path, rgba_png_path, white_square_jpg_path, white_square_png_path],
        ids=[plane_jpg_id, plane_png_id, rgba_png_id, white_square_jpg_id, white_square_png_id],
    )
    def test_should_adjust_colors(
        self,
        factor: float,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_adjusted_color_balance = image.adjust_color_balance(factor)
        assert image != image_adjusted_color_balance
        assert image_adjusted_color_balance == snapshot_png_image

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_not_adjust_colors_factor_1(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(
            UserWarning,
            match="Color adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path))
            image_adjusted_color_balance = image.adjust_color_balance(1)
            assert image == image_adjusted_color_balance

    @pytest.mark.parametrize(
        "resource_path",
        [grayscale_png_path, grayscale_jpg_path],
        ids=[grayscale_png_id, grayscale_jpg_id],
    )
    def test_should_not_adjust_colors_channel_1(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(
            UserWarning,
            match="Color adjustment will not have an affect on grayscale images with only one channel",
        ):
            image = Image.from_file(resolve_resource_path(resource_path))
            image_adjusted_color_balance = image.adjust_color_balance(0.5)
            assert image == image_adjusted_color_balance

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_raise_negative_color_adjust(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1.0\) is not inside \[0, \u221e\)."):
            Image.from_file(resolve_resource_path(resource_path)).adjust_color_balance(-1.0)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestBlur:
    @pytest.mark.parametrize(
        "resource_path",
        images_asymmetric(),
        ids=images_asymmetric_ids(),
    )
    def test_should_return_blurred_image(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        skip_if_os([os_mac])
        image = Image.from_file(resolve_resource_path(resource_path))
        image_blurred = image.blur(2)
        assert image_blurred == snapshot_png_image
        _assert_width_height_channel(image, image_blurred)

    @pytest.mark.parametrize(
        "resource_path",
        images_asymmetric(),
        ids=images_asymmetric_ids(),
    )
    def test_should_not_blur_radius_0(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(
            UserWarning,
            match="Blur radius is 0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path))
            image_blurred = image.blur(0)
            assert image == image_blurred

    @pytest.mark.parametrize(
        "resource_path",
        images_asymmetric(),
        ids=images_asymmetric_ids(),
    )
    def test_should_raise_blur_radius_out_of_bounds(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        with pytest.raises(
            OutOfBoundsError,
            match=rf"radius \(=-1\) is not inside \[0, {min(image.width, image.height) - 1}\].",
        ):
            image.blur(-1)
        with pytest.raises(
            OutOfBoundsError,
            match=rf"radius \(={min(image.width, image.height)}\) is not inside \[0, {min(image.width, image.height) - 1}\].",
        ):
            image.blur(min(image.width, image.height))


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestSharpen:
    @pytest.mark.parametrize("factor", [0, 0.5, 10], ids=["zero factor", "small factor", "large factor"])
    @pytest.mark.parametrize(
        "resource_path",
        [plane_jpg_path, plane_png_path, grayscale_jpg_path, grayscale_png_path],
        ids=[plane_jpg_id, plane_png_id, grayscale_jpg_id, grayscale_png_id],
    )
    def test_should_sharpen(
        self,
        factor: float,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_sharpened = image.sharpen(factor)
        assert image != image_sharpened
        assert image_sharpened == snapshot_png_image
        _assert_width_height_channel(image, image_sharpened)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_raise_negative_sharpen(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1.0\) is not inside \[0, \u221e\)."):
            Image.from_file(resolve_resource_path(resource_path)).sharpen(-1.0)

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_not_sharpen(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(UserWarning, match="Sharpen factor is 1.0, this will not make changes to the image."):
            image = Image.from_file(resolve_resource_path(resource_path))
            image_sharpened = image.sharpen(1)
            assert image == image_sharpened


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestInvertColors:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_invert_colors(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_inverted_colors = image.invert_colors()
        assert image_inverted_colors == snapshot_png_image
        _assert_width_height_channel(image, image_inverted_colors)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestRotate:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_clockwise_rotated_image(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_right_rotated = image.rotate_right()
        assert image_right_rotated == snapshot_png_image
        assert image.channel == image_right_rotated.channel

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_counter_clockwise_rotated_image(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_left_rotated = image.rotate_left()
        assert image_left_rotated == snapshot_png_image
        assert image.channel == image_left_rotated.channel

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_flipped_image(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_left_rotated = image.rotate_left().rotate_left()
        image_right_rotated = image.rotate_right().rotate_right()
        image_flipped_h_v = image.flip_horizontally().flip_vertically()
        image_flipped_v_h = image.flip_horizontally().flip_vertically()
        assert image_left_rotated == image_right_rotated
        assert image_left_rotated == image_flipped_h_v
        assert image_left_rotated == image_flipped_v_h

    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_be_original(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_left_right_rotated = image.rotate_left().rotate_right()
        image_right_left_rotated = image.rotate_right().rotate_left()
        image_left_l_l_l_l = image.rotate_left().rotate_left().rotate_left().rotate_left()
        image_left_r_r_r_r = image.rotate_right().rotate_right().rotate_right().rotate_right()
        assert image == image_left_right_rotated
        assert image == image_right_left_rotated
        assert image == image_left_l_l_l_l
        assert image == image_left_r_r_r_r


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestFindEdges:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_return_edges_of_image(
        self,
        resource_path: str,
        snapshot_png_image: SnapshotAssertion,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        image_edges = image.find_edges()
        assert image_edges == snapshot_png_image
        _assert_width_height_channel(image, image_edges)


class TestFilterEdgesKernel:

    def test_should_kernel_change_device(self) -> None:
        assert Image._filter_edges_kernel().device == _get_device()
        configure_test_with_device(device_cpu)
        assert Image._filter_edges_kernel().device == _get_device()
        configure_test_with_device(device_cuda)
        assert Image._filter_edges_kernel().device == _get_device()
        configure_test_with_device(device_cpu)
        assert Image._filter_edges_kernel().device == _get_device()



@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestSizeof:
    @pytest.mark.parametrize(
        "resource_path",
        images_all(),
        ids=images_all_ids(),
    )
    def test_should_size_be_greater_than_normal_object(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        image = Image.from_file(resolve_resource_path(resource_path))
        assert sys.getsizeof(image) >= image.width * image.height * image.channel
