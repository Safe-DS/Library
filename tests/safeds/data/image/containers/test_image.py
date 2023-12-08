from pathlib import Path

import pytest
import torch
from syrupy import SnapshotAssertion

from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError, IllegalFormatError
from tests.helpers import resolve_resource_path

_device_cuda = torch.device("cuda")
_device_cpu = torch.device("cpu")


def _test_devices() -> list[torch.device]:
    return [_device_cpu, _device_cuda]


def _test_devices_ids() -> list[str]:
    return ["cpu", "cuda"]


def _skip_if_device_not_available(device) -> None:
    if device == _device_cuda and not torch.cuda.is_available():
        pytest.skip("This test requires cuda")


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestFromFile:

    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.jpg", Path("image/white_square.jpg"), "image/white_square.png",
         Path("image/white_square.png")],
        ids=["jpg", "jpg_Path", "png", "png_Path"],
    )
    def test_should_load_from_file(self, resource_path: str | Path, device) -> None:
        _skip_if_device_not_available(device)
        Image.from_file(resolve_resource_path(resource_path), device)

    @pytest.mark.parametrize(
        "resource_path",
        ["image/missing_file.jpg", Path("image/missing_file.jpg"), "image/missing_file.png",
         Path("image/missing_file.png")],
        ids=["missing_file_jpg", "missing_file_jpg_Path", "missing_file_png", "missing_file_png_Path"],
    )
    def test_should_raise_if_file_not_found(self, resource_path: str | Path, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(FileNotFoundError):
            Image.from_file(resolve_resource_path(resource_path), device)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestReprJpeg:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.jpg",
            "image/white_square.jpg",
            "image/white_square.png"
        ],
        ids=["plane-jpg", "white_square-jpg", "white_square-png"]
    )
    def test_should_return_bytes(self, resource_path: str | Path, device) -> None:
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert isinstance(image._repr_jpeg_(), bytes)

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["plane-png", "rgba-png"]
    )
    def test_should_raise_if_image_has_alpha_channel(self, resource_path: str | Path, device) -> None:
        image = Image.from_file(resolve_resource_path(resource_path), device)
        with pytest.raises(IllegalFormatError, match=r"This format is illegal. The image has an alpha channel which "
                                                     r"cannot be displayed in jpeg format. Use one of the following "
                                                     r"formats: png"):
            image._repr_jpeg_()


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestReprPng:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.jpg",
            "image/plane.png",
            "image/rgba.png",
            "image/white_square.jpg",
            "image/white_square.png"
        ],
        ids=["plane-jpg", "plane-png", "rgba-png", "white_square-jpg", "white_square-png"]
    )
    def test_should_return_bytes(self, resource_path: str | Path, device) -> None:
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert isinstance(image._repr_png_(), bytes)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestProperties:

    @pytest.mark.parametrize(
        ("resource_path", "width", "height", "channel"),
        [
            (
                "image/white_square.jpg",
                1,
                1,
                3,
            ),
            (
                "image/plane.png",
                568,
                320,
                4,
            ),
            (
                "image/rgba.png",
                7,
                5,
                4,
            ),
        ],
        ids=["[3,1,1].jpg", "[4,568,320].png", "[4,568,320].png"],
    )
    def test_should_return_image_properties(self, resource_path: str, width: int, height: int, channel: int,
                                            device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        assert image.width == width
        assert image.height == height
        assert image.channel == channel


class TestEQ:

    @pytest.mark.parametrize(
        "device", _test_devices(), ids=_test_devices_ids()
    )
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.jpg",
            "image/plane.png",
            "image/rgba.png",
            "image/white_square.jpg",
            "image/white_square.png",
        ],
        ids=["opaque-4-channel-jpg", "opaque-4-channel-png", "transparent", "opaque-3-channel-jpg",
             "opaque-3-channel-png"],
    )
    def test_should_be_equal(self, resource_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image2 = Image.from_file(resolve_resource_path(resource_path), device)
        assert image == image2

    @pytest.mark.parametrize(
        "device", _test_devices(), ids=_test_devices_ids()
    )
    def test_should_not_be_equal(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/plane.png"), device)
        image2 = Image.from_file(resolve_resource_path("image/white_square.png"), device)
        assert image != image2

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.jpg",
            "image/plane.png",
            "image/rgba.png",
            "image/white_square.jpg",
            "image/white_square.png",
        ],
        ids=["opaque-4-channel-jpg", "opaque-4-channel-png", "transparent", "opaque-3-channel-jpg",
             "opaque-3-channel-png"],
    )
    def test_should_be_equal_different_devices(self, resource_path: str) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path(resource_path), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path(resource_path), torch.device("cuda"))
        assert image == image2
        assert image2 == image

    def test_should_not_be_equal_different_devices(self) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path("image/plane.png"), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path("image/white_square.png"), torch.device("cuda"))
        assert image != image2
        assert image2 != image

    @pytest.mark.parametrize(
        "device", _test_devices(), ids=_test_devices_ids()
    )
    def test_should_raise(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/plane.png"), device)
        other = Table()
        assert (image.__eq__(other)) is NotImplemented


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestResize:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["opaque", "transparent"],
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
        device
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        new_image = image.resize(new_width, new_height)
        assert new_image.width == new_width
        assert new_image.height == new_height
        assert image.width != new_width
        assert image.height != new_height
        assert image != new_image
        assert new_image == snapshot_png


class TestDevices:

    def test_should_change_device(self):
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path("image/plane.png"), torch.device("cpu"))
        new_device = torch.device("cuda", 0)
        assert image.set_device(new_device).device == new_device


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestConvertToGrayscale:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["grayscale", "grayscale-transparent"],
    )
    def test_convert_to_grayscale(self, resource_path: str, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        grayscale_image = image.convert_to_grayscale()
        assert grayscale_image == snapshot_png


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestCrop:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["crop", "crop-transparent"],
    )
    def test_should_return_cropped_image(self, resource_path: str, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image = image.crop(0, 0, 100, 100)  # TODO what is expected behaviour if image is smaller
        assert image == snapshot_png


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestFlipVertically:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png"
        ],
        ids=["opaque", "transparent"]
    )
    def test_should_flip_vertically(self, resource_path: str, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_v = image.flip_vertically()
        assert image != image_flip_v
        assert image_flip_v == snapshot_png

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png"
        ],
        ids=["opaque", "transparent"]
    )
    def test_should_be_original(self, resource_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_v_v = image.flip_vertically().flip_vertically()
        assert image == image_flip_v_v


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestFlipHorizontally:

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png"
        ],
        ids=["opaque", "transparent"]
    )
    def test_should_flip_horizontally(self, resource_path: str, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_h = image.flip_horizontally()
        assert image != image_flip_h
        assert image_flip_h == snapshot_png

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png"
        ],
        ids=["opaque", "transparent"]
    )
    def test_should_be_original(self, resource_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_flip_h_h = image.flip_horizontally().flip_horizontally()
        assert image == image_flip_h_h


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestBrightness:
    @pytest.mark.parametrize("factor", [0.5, 10], ids=["small factor", "large factor"])
    def test_should_adjust_brightness(self, factor: float, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/plane.png"), device)
        image_adjusted_brightness = image.adjust_brightness(factor)
        assert image != image_adjusted_brightness
        assert image_adjusted_brightness == snapshot_png

    def test_should_not_brighten(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(
            UserWarning,
            match="Brightness adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path("image/plane.png"), device)
            image2 = image.adjust_brightness(1)
            assert image == image2

    def test_should_raise(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/plane.png"), device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
            image.adjust_brightness(-1)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestAddNoise:
    @pytest.mark.parametrize(
        ("resource_path", "standard_deviation"),
        [
            ("image/plane.png", 0.0),
            ("image/plane.png", 0.7),
            ("image/plane.png", 2.5),
        ],
        ids=["minimum noise", "some noise", "very noisy"],
    )
    def test_should_add_noise(self, resource_path: str, standard_deviation: float, snapshot_png: SnapshotAssertion,
                              device) -> None:
        _skip_if_device_not_available(device)
        torch.manual_seed(0)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_noise = image.add_noise(standard_deviation)
        assert image_noise == snapshot_png

    @pytest.mark.parametrize(
        ("resource_path", "standard_deviation"),
        [("image/plane.png", -1)],
        ids=["sigma below zero"],
    )
    def test_should_raise_standard_deviation(self, resource_path: str, standard_deviation: float, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(
            OutOfBoundsError,
            match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
        ):
            image = Image.from_file(resolve_resource_path(resource_path), device)
            image.add_noise(standard_deviation)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestAdjustContrast:
    @pytest.mark.parametrize("factor", [0.75, 5], ids=["small factor", "large factor"])
    def test_should_adjust_contrast(self, factor: float, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/plane.png"), device)
        image_adjusted_contrast = image.adjust_contrast(factor)
        assert image != image_adjusted_contrast
        assert image_adjusted_contrast == snapshot_png

    def test_should_not_adjust_contrast(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(
            UserWarning,
            match="Contrast adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path("image/plane.png"), device)
            image_adjusted_contrast = image.adjust_contrast(1)
            assert image == image_adjusted_contrast


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestBlur:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png"
        ],
        ids=["blur"],
    )
    def test_should_return_blurred_image(self, resource_path: str, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device=device)
        image_blurred = image.blur(2)
        assert image_blurred == snapshot_png


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestSharpen:
    @pytest.mark.parametrize("factor", [0, 0.5, 10], ids=["zero factor", "small factor", "large factor"])
    def test_should_sharpen(self, factor: float, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/plane.png"), device)
        image_sharpened = image.sharpen(factor)
        assert image != image_sharpened
        assert image_sharpened == snapshot_png

    def test_should_raise_negative_sharpen(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(OutOfBoundsError):
            Image.from_file(resolve_resource_path("image/plane.png"), device).sharpen(-1.0)

    def test_should_not_sharpen(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(UserWarning, match="Sharpen factor is 1.0, this will not make changes to the image."):
            image = Image.from_file(resolve_resource_path("image/plane.png"), device)
            image_sharpened = image.sharpen(1)
            assert image == image_sharpened


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestInvertColors:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["invert-colors", "invert-colors-transparent"],
    )
    def test_should_invert_colors(self, resource_path: str, snapshot_png: SnapshotAssertion, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_inverted_colors = image.invert_colors()
        assert image_inverted_colors == snapshot_png


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestRotate:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["rotate-clockwise", "rotate-clockwise-transparent"],
    )
    def test_should_return_clockwise_rotated_image(self, resource_path: str, snapshot_png: SnapshotAssertion,
                                                   device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_right_rotated = image.rotate_right()
        assert image_right_rotated == snapshot_png

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["rotate-counter-clockwise", "rotate-counter-clockwise-transparent"],
    )
    def test_should_return_counter_clockwise_rotated_image(self, resource_path: str, snapshot_png: SnapshotAssertion,
                                                           device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(resource_path), device)
        image_left_rotated = image.rotate_left()
        assert image_left_rotated == snapshot_png

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["rotate-to-flipped", "rotate-to-flipped-transparent"],
    )
    def test_should_return_flipped_image(self, resource_path: str, device) -> None:
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
        [
            "image/plane.png",
            "image/rgba.png",
        ],
        ids=["rotate-to-flipped", "rotate-to-flipped-transparent"],
    )
    def test_should_be_original(self, resource_path: str, device) -> None:
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
