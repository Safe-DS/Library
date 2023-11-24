from pathlib import Path

import pytest
import torch

from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from tests.helpers import resolve_resource_path

_device_cuda = torch.device("cuda")
_device_cpu = torch.device("cpu")


def _test_devices() -> list[torch.device]:
    return [_device_cpu, _device_cuda]


def _test_devices_ids() -> list[str]:
    return ["cpu", "cuda"]


def _skip_if_device_not_available(device) -> None:
    if device == _device_cuda and torch.cuda.is_available():
        pytest.skip("This test requires cuda")


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestFromFile:

    @pytest.mark.parametrize(
        "path",
        ["image/white_square.jpg", Path("image/white_square.jpg"), "image/white_square.png",
         Path("image/white_square.png")],
        ids=["jpg", "jpg_Path", "png", "png_Path"],
    )
    def test_should_load_from_file(self, path: str | Path, device) -> None:
        _skip_if_device_not_available(device)
        Image.from_file(resolve_resource_path(path), device)

    @pytest.mark.parametrize(
        "path",
        ["image/missing_file.jpg", Path("image/missing_file.jpg"), "image/missing_file.png",
         Path("image/missing_file.png")],
        ids=["missing_file_jpg", "missing_file_jpg_Path", "missing_file_png", "missing_file_png_Path"],
    )
    def test_should_raise_if_file_not_found(self, path: str | Path, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(FileNotFoundError):
            Image.from_file(resolve_resource_path(path), device)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestProperties:

    @pytest.mark.parametrize(
        ("image_path", "width", "height"),
        [
            (
                "image/white_square.jpg",
                1,
                1,
            ),
            (
                "image/snapshot_boxplot.png",
                640,
                480,
            ),
        ],
        ids=["[1,1].jpg", "[640,480].png"],
    )
    def test_should_return_image_properties(self, image_path: str, width: int, height: int, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        assert image.width == width
        assert image.height == height


class TestEQ:

    @pytest.mark.parametrize(
        "device", _test_devices(), ids=_test_devices_ids()
    )
    def test_should_be_equal(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        image2 = Image.from_file(resolve_resource_path("image/copy.png"), device)
        assert image == image2

    @pytest.mark.parametrize(
        "device", _test_devices(), ids=_test_devices_ids()
    )
    def test_should_not_be_equal(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        image2 = Image.from_file(resolve_resource_path("image/white_square.png"), device)
        assert image != image2

    def test_should_be_equal_different_devices(self) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path("image/original.png"), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path("image/copy.png"), torch.device("cuda"))
        assert image == image2
        assert image2 == image

    def test_should_not_be_equal_different_devices(self) -> None:
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path("image/original.png"), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path("image/white_square.png"), torch.device("cuda"))
        assert image != image2
        assert image2 != image

    @pytest.mark.parametrize(
        "device", _test_devices(), ids=_test_devices_ids()
    )
    def test_should_raise(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        other = Table()
        assert (image.__eq__(other)) is NotImplemented


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestResize:
    @pytest.mark.parametrize(
        ("image_path", "new_width", "new_height"),
        [
            (
                "image/white_square.png",
                2,
                3,
            ),
        ],
        ids=["(2, 3)"],
    )
    def test_should_return_resized_image(
        self,
        image_path: Image,
        new_width: int,
        new_height: int,
        device
    ) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/white_square.png"), device)
        new_image = image.resize(new_width, new_height)
        assert new_image.width == new_width
        assert new_image.height == new_height
        assert image.width != new_width
        assert image.height != new_height
        assert image != new_image


class TestDevices:

    def test_should_change_device(self):
        _skip_if_device_not_available(_device_cuda)
        image = Image.from_file(resolve_resource_path("image/original.png"), torch.device("cpu"))
        new_device = torch.device("cuda", 0)
        assert image.set_device(new_device).device == new_device


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestConvertToGrayscale:
    @pytest.mark.parametrize(
        ("image_path", "expected_path"),
        [
            (
                "image/snapshot_heatmap.png",
                "image/snapshot_heatmap_grayscale.png"
            ),
        ],
        ids=["grayscale"],
    )
    def test_convert_to_grayscale(self, image_path: str, expected_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        expected = Image.from_file(resolve_resource_path(expected_path), device)
        grayscale_image = image.convert_to_grayscale()
        assert grayscale_image == expected


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestCrop:
    @pytest.mark.parametrize(
        ("image_path", "expected_path"),
        [
            (
                "image/white.png",
                "image/whiteCropped.png",
            ),
        ],
        ids=["crop"],
    )
    def test_should_return_cropped_image(self, image_path: str, expected_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        expected = Image.from_file(resolve_resource_path(expected_path), device)
        image = image.crop(0, 0, 100, 100)
        assert image == expected


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestFlipVertically:
    def test_should_flip_vertically(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        image2 = image.flip_vertically()
        image3 = Image.from_file(resolve_resource_path("image/flip_vertically.png"), device)
        assert image != image2
        assert image2 == image3

    def test_should_be_original(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        image2 = image.flip_vertically().flip_vertically()
        assert image == image2


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestFlipHorizontally:
    def test_should_flip_horizontally(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        image2 = image.flip_horizontally()
        image3 = Image.from_file(resolve_resource_path("image/flip_horizontally.png"), device)
        assert image != image2
        assert image2 == image3

    def test_should_be_original(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/original.png"), device)
        image2 = image.flip_horizontally().flip_horizontally()
        assert image == image2


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestBrightness:
    @pytest.mark.parametrize("factor", [0.5, 10], ids=["small factor", "large factor"])
    def test_should_adjust_brightness(self, factor: float, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/brightness/to_brighten.png"), device)
        image2 = image.adjust_brightness(factor)
        image3 = Image.from_file(resolve_resource_path("image/brightness/brightened_by_" + str(factor) + ".png"),
                                 device)
        assert image != image2
        assert image2 == image3

    def test_should_not_brighten(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(
            UserWarning,
            match="Brightness adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path("image/brightness/to_brighten.png"), device)
            image2 = image.adjust_brightness(1)
            assert image == image2

    def test_should_raise(self, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/brightness/to_brighten.png"), device)
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
            image.adjust_brightness(-1)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestAddGaussianNoise:
    @pytest.mark.parametrize(
        ("image_path", "standard_deviation"),
        [
            ("image/boy.png", 0.0),
            ("image/boy.png", 0.7),
            ("image/boy.png", 2.5),
        ],
        ids=["minimum noise", "some noise", "very noisy"],
    )
    def test_should_add_noise(self, image_path: str, standard_deviation: float, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        expected = Image.from_file(
            resolve_resource_path("image/noise/noise_" + str(standard_deviation) + ".png"), device
        )
        image = image.add_gaussian_noise(standard_deviation)

        # plt.imshow(image._image_tensor.squeeze().permute(1, 2, 0).cpu())
        # plt.show()
        #
        # plt.imshow(expected._image_tensor.squeeze().permute(1, 2, 0).cpu())
        # plt.show()

        assert image == expected

    @pytest.mark.parametrize(
        ("image_path", "standard_deviation"),
        [("image/boy.png", -1)],
        ids=["sigma below zero"],
    )
    def test_should_raise_standard_deviation(self, image_path: str, standard_deviation: float, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(
            OutOfBoundsError,
            match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
        ):
            image = Image.from_file(resolve_resource_path(image_path), device)
            image.add_gaussian_noise(standard_deviation)


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestAdjustContrast:
    @pytest.mark.parametrize("factor", [0.75, 5], ids=["small factor", "large factor"])
    def test_should_adjust_contrast(self, factor: float, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/contrast/to_adjust_contrast.png"), device)
        image2 = image.adjust_contrast(factor)
        image3 = Image.from_file(
            resolve_resource_path("image/contrast/contrast_adjusted_by_" + str(factor) + ".png"), device
        )

        # plt.imshow(image2._image_tensor.squeeze().permute(1, 2, 0).cpu())
        # plt.show()
        #
        # plt.imshow(image3._image_tensor.squeeze().permute(1, 2, 0).cpu())
        # plt.show()

        assert image != image2
        assert image2 == image3

    def test_should_not_adjust_contrast(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(
            UserWarning,
            match="Contrast adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_file(resolve_resource_path("image/contrast/to_adjust_contrast.png"), device)
            image2 = image.adjust_contrast(1)
            assert image == image2


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestBlur:
    @pytest.mark.parametrize(
        ("image_path", "expected_path"),
        [
            (
                "image/boy.png",
                "image/blurredBoy.png",
            ),
        ],
        ids=["blur"],
    )
    def test_should_return_blurred_image(self, image_path: str, expected_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device=device)
        expected = Image.from_file(resolve_resource_path(expected_path), device=device)
        image = image.blur(2)
        # plt.imshow(image._image_tensor.squeeze().permute(1, 2, 0).cpu())
        # plt.show()
        #
        # plt.imshow(expected._image_tensor.squeeze().permute(1, 2, 0).cpu())
        # plt.show()
        assert image == expected


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestSharpen:
    @pytest.mark.parametrize("factor", [0, 0.5, 10], ids=["zero factor", "small factor", "large factor"])
    def test_should_sharpen(self, factor: float, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path("image/sharpen/to_sharpen.png"), device)
        image2 = image.sharpen(factor)
        # image2.to_png_file(resolve_resource_path("image/sharpen/sharpened_by_" + str(factor) + ".png"))
        assert image != image2
        assert image2 == Image.from_file(
            resolve_resource_path("image/sharpen/sharpened_by_" + str(factor) + ".png"), device
        )

    def test_should_raise_negative_sharpen(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.raises(OutOfBoundsError):
            Image.from_file(resolve_resource_path("image/sharpen/to_sharpen.png"), device).sharpen(-1.0)

    def test_should_not_sharpen(self, device) -> None:
        _skip_if_device_not_available(device)
        with pytest.warns(UserWarning, match="Sharpen factor is 1.0, this will not make changes to the image."):
            image = Image.from_file(resolve_resource_path("image/sharpen/to_sharpen.png"), device)
            image2 = image.sharpen(1)
            assert image == image2


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestInvertColors:
    @pytest.mark.parametrize(
        ("image_path", "expected_path"),
        [
            (
                "image/original.png",
                "image/inverted_colors_original.png",
            ),
        ],
        ids=["invert-colors"],
    )
    def test_should_invert_colors(self, image_path: str, expected_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        expected = Image.from_file(resolve_resource_path(expected_path), device)
        image = image.invert_colors()
        assert image == expected


@pytest.mark.parametrize(
    "device", _test_devices(), ids=_test_devices_ids()
)
class TestRotate:
    @pytest.mark.parametrize(
        ("image_path", "expected_path"),
        [
            (
                "image/snapshot_boxplot.png",
                "image/snapshot_boxplot_right_rotation.png",
            ),
        ],
        ids=["rotate-clockwise"],
    )
    def test_should_return_clockwise_rotated_image(self, image_path: str, expected_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        expected = Image.from_file(resolve_resource_path(expected_path), device)
        image = image.rotate_right()
        assert image == expected

    @pytest.mark.parametrize(
        ("image_path", "expected_path"),
        [
            (
                "image/snapshot_boxplot.png",
                "image/snapshot_boxplot_left_rotation.png",
            ),
        ],
        ids=["rotate-counter-clockwise"],
    )
    def test_should_return_counter_clockwise_rotated_image(self, image_path: str, expected_path: str, device) -> None:
        _skip_if_device_not_available(device)
        image = Image.from_file(resolve_resource_path(image_path), device)
        expected = Image.from_file(resolve_resource_path(expected_path), device)
        image = image.rotate_left()
        assert image == expected
