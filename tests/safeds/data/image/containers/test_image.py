import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import torch
from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.exceptions import IllegalFormatError, OutOfBoundsError
from syrupy import SnapshotAssertion
from torch.types import Device

from tests.helpers import resolve_resource_path

_device_cuda = torch.device("cuda")
_device_cpu = torch.device("cpu")

_plane_jpg_path = "image/plane.jpg"
_plane_png_path = "image/plane.png"
_rgba_png_path = "image/rgba.png"
_white_square_jpg_path = "image/white_square.jpg"
_white_square_png_path = "image/white_square.png"
_grayscale_jpg_path = "image/grayscale.jpg"
_grayscale_png_path = "image/grayscale.png"

_plane_jpg_id = "opaque-3-channel-jpg-plane"
_plane_png_id = "opaque-4-channel-png-plane"
_rgba_png_id = "transparent-4-channel-png-rgba"
_white_square_jpg_id = "opaque-3-channel-jpg-white_square"
_white_square_png_id = "opaque-3-channel-png-white_square"
_grayscale_jpg_id = "opaque-1-channel-jpg-grayscale"
_grayscale_png_id = "opaque-1-channel-png-grayscale"


def _test_devices() -> list[torch.device]:
    return [_device_cpu, _device_cuda]


def _test_devices_ids() -> list[str]:
    return ["cpu", "cuda"]


def _test_images_all() -> list[str]:
    return [
        _plane_jpg_path,
        _plane_png_path,
        _rgba_png_path,
        _white_square_jpg_path,
        _white_square_png_path,
        _grayscale_jpg_path,
        _grayscale_png_path,
    ]


def _test_images_all_ids() -> list[str]:
    return [
        _plane_jpg_id,
        _plane_png_id,
        _rgba_png_id,
        _white_square_jpg_id,
        _white_square_png_id,
        _grayscale_jpg_id,
        _grayscale_png_id,
    ]


def _test_images_asymmetric() -> list[str]:
    return [
        _plane_jpg_path,
        _plane_png_path,
        _rgba_png_path,
        _grayscale_jpg_path,
        _grayscale_png_path,
    ]


def _test_images_asymmetric_ids() -> list[str]:
    return [
        _plane_jpg_id,
        _plane_png_id,
        _rgba_png_id,
        _grayscale_jpg_id,
        _grayscale_png_id,
    ]


def _skip_if_device_not_available(device: Device) -> None:
    if device == _device_cuda and not torch.cuda.is_available():
        pytest.skip("This test requires cuda")


def _assert_width_height_channel(image1: Image, image2: Image) -> None:
    assert image1.width == image2.width
    assert image1.height == image2.height
    assert image1.channel == image2.channel


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestFromFile:
    @pytest.mark.parametrize(
        "resource_path",
        [*_test_images_all(), *[Path(image_path) for image_path in _test_images_all()]],
        ids=[
            *["file-" + image_id for image_id in _test_images_all_ids()],
            *["path-" + image_id for image_id in _test_images_all_ids()],
        ],
    )
    def test_should_load_from_file(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
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
        _skip_if_device_not_available(device)
        with pytest.raises(FileNotFoundError):
            Image.from_file(resolve_resource_path(resource_path), device)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestFromBytes:
    @pytest.mark.parametrize(
        "resource_path",
        [_plane_jpg_path, _white_square_jpg_path, _white_square_png_path, _grayscale_jpg_path, _grayscale_png_path],
        ids=[_plane_jpg_id, _white_square_jpg_id, _white_square_png_id, _grayscale_jpg_id, _grayscale_png_id],
    )
    def test_should_write_and_load_bytes_jpeg(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_copy = Image.from_bytes(typing.cast(bytes, image._repr_jpeg_()), device)
        _assert_width_height_channel(image, image_copy)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_write_and_load_bytes_png(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_copy = Image.from_bytes(image._repr_png_(), device)
        assert image == image_copy


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestReprJpeg:
    @pytest.mark.parametrize(
        "resource_path",
        [_plane_jpg_path, _white_square_jpg_path, _white_square_png_path, _grayscale_jpg_path, _grayscale_png_path],
        ids=[_plane_jpg_id, _white_square_jpg_id, _white_square_png_id, _grayscale_jpg_id, _grayscale_png_id],
    )
    def test_should_return_bytes(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert isinstance(image._repr_jpeg_(), bytes)

    @pytest.mark.parametrize(
        "resource_path",
        [
            _plane_png_path,
            _rgba_png_path,
        ],
        ids=[_plane_png_id, _rgba_png_id],
    )
    def test_should_return_none_if_image_has_alpha_channel(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert image._repr_jpeg_() is None


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestReprPng:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_return_bytes(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert isinstance(image._repr_png_(), bytes)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestToJpegFile:
    @pytest.mark.parametrize(
        "resource_path",
        [_plane_jpg_path, _white_square_jpg_path, _white_square_png_path, _grayscale_jpg_path, _grayscale_png_path],
        ids=[_plane_jpg_id, _white_square_jpg_id, _white_square_png_id, _grayscale_jpg_id, _grayscale_png_id],
    )
    def test_should_save_file(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
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
            _plane_png_path,
            _rgba_png_path,
        ],
        ids=[_plane_png_id, _rgba_png_id],
    )
    def test_should_raise_if_image_has_alpha_channel(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        with NamedTemporaryFile(suffix=".jpg") as tmp_jpeg_file:
            tmp_jpeg_file.close()
            with Path(tmp_jpeg_file.name).open("w", encoding="utf-8") as tmp_file, pytest.raises(
                IllegalFormatError,
                match=r"This format is illegal. Use one of the following formats: png",
            ):
                image.to_jpeg_file(tmp_file.name)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestToPngFile:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_save_file(self, resource_path: str | Path, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        with NamedTemporaryFile(suffix=".png") as tmp_png_file:
            tmp_png_file.close()
            with Path(tmp_png_file.name).open("w", encoding="utf-8") as tmp_file:
                image.to_png_file(tmp_file.name)
            with Path(tmp_png_file.name).open("r", encoding="utf-8") as tmp_file:
                image_r = Image.from_file(tmp_file.name)
        assert image == image_r


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestProperties:
    @pytest.mark.parametrize(
        ("resource_path", "width", "height", "channel"),
        [
            (
                _white_square_jpg_path,
                1,
                1,
                3,
            ),
            (
                _plane_png_path,
                568,
                320,
                4,
            ),
            (
                _rgba_png_path,
                7,
                5,
                4,
            ),
        ],
        ids=["[3,1,1]" + _white_square_jpg_id, "[4,568,320]" + _plane_png_id, "[4,568,320]" + _rgba_png_id],
    )
    def test_should_return_image_properties(
        self,
        resource_path: str,
        width: int,
        height: int,
        channel: int,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert image.width == width
        assert image.height == height
        assert image.channel == channel


class TestEQ:
    @pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_be_equal(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image2 = Image.from_file(resolve_resource_path(resource_path), device)
        assert image == image2

    @pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
    def test_should_not_be_equal(self, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(_plane_png_path), device)
        image2 = Image.from_file(resolve_resource_path(_white_square_png_path), device)
        assert image != image2

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_be_equal_different_devices(self, resource_path: str) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path(resource_path), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path(resource_path), torch.device("cuda"))
        assert image == image2
        assert image2 == image

    def test_should_not_be_equal_different_devices(self) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path(_plane_png_path), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path(_white_square_png_path), torch.device("cuda"))
        assert image != image2
        assert image2 != image

    @pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_raise(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        other = Table()
        assert (image.__eq__(other)) is NotImplemented


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestResize:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    @pytest.mark.parametrize(
        ("new_width", "new_height"),
        [
            (
                2,
                3,
            ),
            (
                700,
                400,
            ),
        ],
        ids=["(2, 3)", "(700, 400)"],
    )
    def test_should_return_resized_image(
        self,
        resource_path: str,
        new_width: int,
        new_height: int,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        new_image = image.resize(new_width, new_height)
        assert new_image.width == new_width
        assert new_image.height == new_height
        assert image.channel == new_image.channel
        assert image != new_image
        assert new_image == snapshot_png


class TestDevices:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_change_device(self, resource_path: str) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path(resource_path), torch.device("cpu"))
        new_device = torch.device("cuda", 0)
        assert image._set_device(new_device).device == new_device


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestConvertToGrayscale:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_convert_to_grayscale(self, resource_path: str, snapshot_png: SnapshotAssertion, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        grayscale_image = image.convert_to_grayscale()
        assert grayscale_image == snapshot_png
        _assert_width_height_channel(image, grayscale_image)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestCrop:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_return_cropped_image(
        self,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_cropped = image.crop(0, 0, 100, 100)
        assert image_cropped == snapshot_png
        assert image_cropped.channel == image.channel


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestFlipVertically:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_asymmetric(),
        ids=_test_images_asymmetric_ids(),
    )
    def test_should_flip_vertically(self, resource_path: str, snapshot_png: SnapshotAssertion, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_v = image.flip_vertically()
        assert image != image_flip_v
        assert image_flip_v == snapshot_png
        _assert_width_height_channel(image, image_flip_v)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_be_original(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_v_v = image.flip_vertically().flip_vertically()
        assert image == image_flip_v_v


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestFlipHorizontally:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_asymmetric(),
        ids=_test_images_asymmetric_ids(),
    )
    def test_should_flip_horizontally(
        self,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_h = image.flip_horizontally()
        assert image != image_flip_h
        assert image_flip_h == snapshot_png
        _assert_width_height_channel(image, image_flip_h)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_be_original(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_h_h = image.flip_horizontally().flip_horizontally()
        assert image == image_flip_h_h


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestBrightness:
    @pytest.mark.parametrize("factor", [0.5, 10], ids=["small factor", "large factor"])
    @pytest.mark.parametrize(
        "resource_path",
        [_plane_jpg_path, _plane_png_path],
        ids=[_plane_jpg_id, _plane_png_id],
    )
    def test_should_adjust_brightness(
        self,
        factor: float,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_adjusted_brightness = image.adjust_brightness(factor)
        assert image != image_adjusted_brightness
        assert image_adjusted_brightness == snapshot_png
        _assert_width_height_channel(image, image_adjusted_brightness)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_not_brighten(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(
            UserWarning,
            match="Brightness adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path), device)
            image2 = image.adjust_brightness(1)
            assert image == image2

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_raise(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
            image.adjust_brightness(-1)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
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
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_add_noise(
        self,
        resource_path: str,
        standard_deviation: float,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        torch.manual_seed(0)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_noise = image.add_noise(standard_deviation)
        assert image_noise == snapshot_png
        _assert_width_height_channel(image, image_noise)

    @pytest.mark.parametrize(
        "standard_deviation",
        [-1],
        ids=["sigma below zero"],
    )
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_raise_standard_deviation(
        self,
        resource_path: str,
        standard_deviation: float,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        with pytest.raises(
            OutOfBoundsError,
            match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
        ):
            image.add_noise(standard_deviation)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestAdjustContrast:
    @pytest.mark.parametrize("factor", [0.75, 5], ids=["small factor", "large factor"])
    @pytest.mark.parametrize(
        "resource_path",
        [_plane_jpg_path, _plane_png_path],
        ids=[_plane_jpg_id, _plane_png_id],
    )
    def test_should_adjust_contrast(
        self,
        factor: float,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_adjusted_contrast = image.adjust_contrast(factor)
        assert image != image_adjusted_contrast
        assert image_adjusted_contrast == snapshot_png
        _assert_width_height_channel(image, image_adjusted_contrast)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_not_adjust_contrast(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(
            UserWarning,
            match="Contrast adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path), device)
            image_adjusted_contrast = image.adjust_contrast(1)
            assert image == image_adjusted_contrast

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_raise_negative_contrast(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1.0\) is not inside \[0, \u221e\)."):
            Image.from_file(resolve_resource_path(resource_path), device).adjust_contrast(-1.0)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestBlur:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_asymmetric(),
        ids=_test_images_asymmetric_ids(),
    )
    def test_should_return_blurred_image(
        self,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device=device)
        image_blurred = image.blur(2)
        assert image_blurred == snapshot_png
        _assert_width_height_channel(image, image_blurred)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestSharpen:
    @pytest.mark.parametrize("factor", [0, 0.5, 10], ids=["zero factor", "small factor", "large factor"])
    @pytest.mark.parametrize(
        "resource_path",
        [_plane_jpg_path, _plane_png_path],
        ids=[_plane_jpg_id, _plane_png_id],
    )
    def test_should_sharpen(
        self,
        factor: float,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_sharpened = image.sharpen(factor)
        assert image != image_sharpened
        assert image_sharpened == snapshot_png
        _assert_width_height_channel(image, image_sharpened)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_raise_negative_sharpen(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1.0\) is not inside \[0, \u221e\)."):
            Image.from_file(resolve_resource_path(resource_path), device).sharpen(-1.0)

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_not_sharpen(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(UserWarning, match="Sharpen factor is 1.0, this will not make changes to the image."):
            image = Image.from_file(resolve_resource_path(resource_path), device)
            image_sharpened = image.sharpen(1)
            assert image == image_sharpened


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestInvertColors:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_invert_colors(self, resource_path: str, snapshot_png: SnapshotAssertion, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_inverted_colors = image.invert_colors()
        assert image_inverted_colors == snapshot_png
        _assert_width_height_channel(image, image_inverted_colors)


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestRotate:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_return_clockwise_rotated_image(
        self,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_right_rotated = image.rotate_right()
        assert image_right_rotated == snapshot_png
        assert image.channel == image_right_rotated.channel

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_return_counter_clockwise_rotated_image(
        self,
        resource_path: str,
        snapshot_png: SnapshotAssertion,
        device: Device,
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_left_rotated = image.rotate_left()
        assert image_left_rotated == snapshot_png
        assert image.channel == image_left_rotated.channel

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_return_flipped_image(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_left_rotated = image.rotate_left().rotate_left()
        image_right_rotated = image.rotate_right().rotate_right()
        image_flipped_h_v = image.flip_horizontally().flip_vertically()
        image_flipped_v_h = image.flip_horizontally().flip_vertically()
        assert image_left_rotated == image_right_rotated
        assert image_left_rotated == image_flipped_h_v
        assert image_left_rotated == image_flipped_v_h

    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_be_original(self, resource_path: str, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_left_right_rotated = image.rotate_left().rotate_right()
        image_right_left_rotated = image.rotate_right().rotate_left()
        image_left_l_l_l_l = image.rotate_left().rotate_left().rotate_left().rotate_left()
        image_left_r_r_r_r = image.rotate_right().rotate_right().rotate_right().rotate_right()
        assert image == image_left_right_rotated
        assert image == image_right_left_rotated
        assert image == image_left_l_l_l_l
        assert image == image_left_r_r_r_r


@pytest.mark.parametrize("device", _test_devices(), ids=_test_devices_ids())
class TestFindEdges:
    @pytest.mark.parametrize(
        "resource_path",
        _test_images_all(),
        ids=_test_images_all_ids(),
    )
    def test_should_return_edges_of_image(self, resource_path: str, snapshot_png: SnapshotAssertion, device: Device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device=device)
        image_edges = image.find_edges()
        assert image_edges == snapshot_png
        _assert_width_height_channel(image, image_edges)
