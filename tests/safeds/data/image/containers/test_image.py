from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageFormat
from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


class TestFromJpegFile:
    @pytest.mark.parametrize(
        "path",
        ["image/white_square.jpg", Path("image/white_square.jpg")],
    )
    def test_should_load_jpeg_file(self, path: str | Path) -> None:
        Image.from_jpeg_file(resolve_resource_path(path))

    @pytest.mark.parametrize(
        "path",
        ["image/missing_file.jpg", Path("image/missing_file.jpg")],
    )
    def test_should_raise_if_file_not_found(self, path: str | Path) -> None:
        with pytest.raises(FileNotFoundError):
            Image.from_jpeg_file(resolve_resource_path(path))


class TestFromPngFile:
    @pytest.mark.parametrize(
        "path",
        ["image/white_square.png", Path("image/white_square.png")],
    )
    def test_should_load_png_file(self, path: str | Path) -> None:
        Image.from_png_file(resolve_resource_path(path))

    @pytest.mark.parametrize(
        "path",
        ["image/missing_file.png", Path("image/missing_file.png")],
    )
    def test_should_raise_if_file_not_found(self, path: str | Path) -> None:
        with pytest.raises(FileNotFoundError):
            Image.from_png_file(resolve_resource_path(path))


class TestFormat:
    @pytest.mark.parametrize(
        ("image", "format_"),
        [
            (Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg")), ImageFormat.JPEG),
            (Image.from_png_file(resolve_resource_path("image/white_square.png")), ImageFormat.PNG),
        ],
    )
    def test_should_return_correct_format(self, image: Image, format_: ImageFormat) -> None:
        assert image.format == format_


class TestProperties:
    @pytest.mark.parametrize(
        ("image", "width", "height"),
        [
            (
                Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg")),
                1,
                1,
            ),
            (
                Image.from_png_file(resolve_resource_path("image/snapshot_boxplot.png")),
                640,
                480,
            ),
        ],
        ids=["[1,1].jpg", "[640,480].png"],
    )
    def test_should_return_image_properties(self, image: Image, width: int, height: int) -> None:
        assert image.width == width
        assert image.height == height


class TestToJpegFile:
    @pytest.mark.parametrize(
        "path",
        ["image/white_square.jpg"],
    )
    def test_should_save_jpeg_file_by_str(self, path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_jpeg_file(tmp_write_file.name)
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_jpeg_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()

    @pytest.mark.parametrize(
        "path",
        ["image/white_square.jpg"],
    )
    def test_should_save_jpeg_file_by_path(self, path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_jpeg_file(Path(tmp_write_file.name))
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_jpeg_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()


class TestToPngFile:
    @pytest.mark.parametrize(
        "path",
        ["image/white_square.png"],
    )
    def test_should_save_png_file_by_str(self, path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_png_file(tmp_write_file.name)
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_png_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()

    @pytest.mark.parametrize(
        "path",
        ["image/white_square.png"],
    )
    def test_should_save_png_file_by_path(self, path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_png_file(Path(tmp_write_file.name))
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_png_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()


class TestReprJpeg:
    @pytest.mark.parametrize(
        "image",
        [Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg"))],
    )
    def test_should_return_bytes_if_image_is_jpeg(self, image: Image) -> None:
        assert isinstance(image._repr_jpeg_(), bytes)

    @pytest.mark.parametrize(
        "image",
        [Image.from_png_file(resolve_resource_path("image/white_square.png"))],
    )
    def test_should_return_none_if_image_is_not_jpeg(self, image: Image) -> None:
        assert image._repr_jpeg_() is None


class TestReprPng:
    @pytest.mark.parametrize(
        "image",
        [Image.from_png_file(resolve_resource_path("image/white_square.png"))],
    )
    def test_should_return_bytes_if_image_is_png(self, image: Image) -> None:
        assert isinstance(image._repr_png_(), bytes)

    @pytest.mark.parametrize(
        "image",
        [Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg"))],
    )
    def test_should_return_none_if_image_is_not_png(self, image: Image) -> None:
        assert image._repr_png_() is None


class TestResize:
    @pytest.mark.parametrize(
        ("image", "new_width", "new_height", "new_size"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/white_square.png")),
                2,
                3,
                (2, 3),
            ),
        ],
        ids=["(2, 3)"],
    )
    def test_should_return_resized_image(
        self,
        image: Image,
        new_width: int,
        new_height: int,
        new_size: tuple[int, int],
    ) -> None:
        assert image.resize(new_width, new_height)._image.size == new_size


class TestConvertToGrayscale:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/snapshot_heatmap.png")),
                Image.from_png_file(resolve_resource_path("image/snapshot_heatmap_grayscale.png")),
            ),
        ],
        ids=["grayscale"],
    )
    def test_convert_to_grayscale(self, image: Image, expected: Image) -> None:
        grayscale_image = image.convert_to_grayscale()
        assert grayscale_image == expected


class TestEQ:
    def test_should_be_equal(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        image2 = Image.from_png_file(resolve_resource_path("image/copy.png"))
        assert image == image2

    def test_should_not_be_equal(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        image2 = Image.from_png_file(resolve_resource_path("image/white_square.png"))
        assert image != image2

    def test_should_raise(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        other = Table()
        assert (image.__eq__(other)) is NotImplemented


class TestFlipVertically:
    def test_should_flip_vertically(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        image2 = image.flip_vertically()
        image3 = Image.from_png_file(resolve_resource_path("image/flip_vertically.png"))
        assert image != image2
        assert image2 == image3

    def test_should_be_original(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        image2 = image.flip_vertically().flip_vertically()
        assert image == image2


class TestFlipHorizontally:
    def test_should_flip_horizontally(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        image2 = image.flip_horizontally()
        image3 = Image.from_png_file(resolve_resource_path("image/flip_horizontally.png"))
        assert image != image2
        assert image2 == image3

    def test_should_be_original(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/original.png"))
        image2 = image.flip_horizontally().flip_horizontally()
        assert image == image2


class TestAdjustContrast:
    @pytest.mark.parametrize("factor", [0.75, 5])
    def test_should_adjust_contrast(self, factor: float) -> None:
        image = Image.from_png_file(resolve_resource_path("image/contrast/to_adjust_contrast.png"))
        image2 = image.adjust_contrast(factor)
        image3 = Image.from_png_file(
            resolve_resource_path("image/contrast/contrast_adjusted_by_" + str(factor) + ".png"),
        )
        assert image != image2
        assert image2 == image3

    def test_should_not_adjust_contrast(self) -> None:
        with pytest.warns(
            UserWarning,
            match="Contrast adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_png_file(resolve_resource_path("image/contrast/to_adjust_contrast.png"))
            image2 = image.adjust_contrast(1)
            assert image == image2

    def test_should_raise(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/brightness/to_brighten.png"))
        with pytest.raises(ValueError, match="Contrast factor has to be 0 or bigger"):
            image.adjust_contrast(-1)


class TestBrightness:
    @pytest.mark.parametrize("factor", [0.5, 10])
    def test_should_adjust_brightness(self, factor: float) -> None:
        image = Image.from_png_file(resolve_resource_path("image/brightness/to_brighten.png"))
        image2 = image.adjust_brightness(factor)
        image3 = Image.from_png_file(resolve_resource_path("image/brightness/brightened_by_" + str(factor) + ".png"))
        assert image != image2
        assert image2 == image3

    def test_should_not_brighten(self) -> None:
        with pytest.warns(
            UserWarning,
            match="Brightness adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image = Image.from_png_file(resolve_resource_path("image/brightness/to_brighten.png"))
            image2 = image.adjust_brightness(1)
            assert image == image2

    def test_should_raise(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/brightness/to_brighten.png"))
        with pytest.raises(ValueError, match="Brightness factor has to be 0 or bigger"):
            image.adjust_brightness(-1)


class TestInvertColors:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/original.png")),
                Image.from_png_file(resolve_resource_path("image/inverted_colors_original.png")),
            ),
        ],
        ids=["invert-colors"],
    )
    def test_should_invert_colors(self, image: Image, expected: Image) -> None:
        image = image.invert_colors()
        assert image == expected


class TestGaussianNoise:
    @pytest.mark.parametrize(
        ("image", "standard_deviation", "opacity"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                0.7,
                1.0
            ),
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                2.5,
                0.6
            )
        ],
        ids=["one", "two"]
    )
    def test_should_add_noise(self, image: Image, standard_deviation: float, opacity: float) -> None:
        expected = Image.from_png_file(
            resolve_resource_path("image/noise/noise_" + str(standard_deviation) + "_" + str(opacity) + ".png"))
        assert image.add_gaussian_noise(standard_deviation, opacity) == expected


    @pytest.mark.parametrize(
        ("image", "standard_deviation", "opacity"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                -1,
                1.0
            )
        ]
    )
    def test_should_raise_standard_deviation(self, image: Image, standard_deviation: float, opacity: float) -> None:
        with pytest.raises(ValueError, match="Standard deviation has to be 0 or bigger."):
            image.add_gaussian_noise(standard_deviation, opacity)

    @pytest.mark.parametrize(
        ("image", "standard_deviation", "opacity"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                2,
                -1
            ),
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                2,
                2
            )
        ],
        ids=["below zero", "above zero"]
    )
    def test_should_raise_opacity(self, image: Image, standard_deviation: float, opacity: float) -> None:
        with pytest.raises(ValueError, match="Opacity has to be between 0 and 1."):
            image.add_gaussian_noise(standard_deviation, opacity)

    @pytest.mark.parametrize(
        ("image", "standard_deviation", "opacity"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                1,
                0
            )
        ],
    )
    def test_should_warn(self, image: Image, standard_deviation: float, opacity: float) -> None:
        with pytest.warns(UserWarning, match="Opacity is 0, this will not make changes to the image."):
            image.add_gaussian_noise(standard_deviation, opacity)


class TestBlend:
    @pytest.mark.parametrize(
        ("image", "other", "alpha", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/blend/original_scaled.png")),
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                0.7,
                Image.from_png_file(resolve_resource_path("image/blend/blended.png"))
            )
        ],
        ids=["normal"]
    )
    def test_should_blend_properly(self, image: Image, other: Image, alpha: float, expected: Image) -> None:
        blend = image.blend(other, alpha)
        assert blend == expected

    @pytest.mark.parametrize(
        ("image", "other"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/original.png")),
                Image.from_png_file(resolve_resource_path("image/boy.png"))
            )
        ],
        ids=["normal"]
    )
    def test_should_raise_different_size(self, image: Image, other: Image) -> None:
        with pytest.raises(AttributeError, match="Cannot blend two images of different size."):
            image.blend(other)

    @pytest.mark.parametrize(
        ("image", "other", "alpha"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/blend/original_scaled.png")),
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                5.0
            )
        ],
        ids=["normal"]
    )
    def test_should_raise_alpha(self, image: Image, other: Image, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be between 0 and 1."):
            image.blend(other, alpha)


class TestBlur:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                Image.from_png_file(resolve_resource_path("image/blurredBoy.png")),
            ),
        ],
        ids=["blur"],
    )
    def test_should_return_blurred_image(self, image: Image, expected: Image) -> None:
        image = image.blur(2)
        assert image == expected


class TestCrop:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/white.png")),
                Image.from_png_file(resolve_resource_path("image/whiteCropped.png")),
            ),
        ],
        ids=["crop"],
    )
    def test_should_return_cropped_image(self, image: Image, expected: Image) -> None:
        image = image.crop(0, 0, 100, 100)
        assert image == expected


class TestSharpen:
    @pytest.mark.parametrize("factor", [-1, 0.5, 10])
    def test_should_sharpen(self, factor: float) -> None:
        image = Image.from_png_file(resolve_resource_path("image/sharpen/to_sharpen.png"))
        image2 = image.sharpen(factor)
        image2.to_png_file(resolve_resource_path("image/sharpen/sharpened_by_" + str(factor) + ".png"))
        assert image != image2
        assert image2 == Image.from_png_file(
            resolve_resource_path("image/sharpen/sharpened_by_" + str(factor) + ".png"),
        )

    def test_should_not_sharpen(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/sharpen/to_sharpen.png"))
        image2 = image.sharpen(1)
        assert image == image2


class TestRotate:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/snapshot_boxplot.png")),
                Image.from_png_file(resolve_resource_path("image/snapshot_boxplot_right_rotation.png")),
            ),
        ],
        ids=["rotate-clockwise"],
    )
    def test_should_return_clockwise_rotated_image(self, image: Image, expected: Image) -> None:
        image = image.rotate_right()
        assert image == expected

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/snapshot_boxplot.png")),
                Image.from_png_file(resolve_resource_path("image/snapshot_boxplot_left_rotation.png")),
            ),
        ],
        ids=["rotate-counter-clockwise"],
    )
    def test_should_return_counter_clockwise_rotated_image(self, image: Image, expected: Image) -> None:
        image = image.rotate_left()
        assert image == expected


class TestFindEdges:
    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (
                Image.from_png_file(resolve_resource_path("image/boy.png")),
                Image.from_png_file(resolve_resource_path("image/edgyBoy.png")),
            ),
        ],
        ids=["find_edges"],
    )
    def test_should_return_edges_of_image(self, image: Image, expected: Image) -> None:
        image = image.find_edges()
        assert image == expected
