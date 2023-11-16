from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageFormat
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError

from tests.helpers import resolve_resource_path


class TestFromJpegFile:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.jpg",
            Path("image/plane.jpg"),
        ],
        ids=["jpg", "jpg_Path"],
    )
    def test_should_load_jpeg_file(self, resource_path: str | Path) -> None:
        Image.from_jpeg_file(resolve_resource_path(resource_path))

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/missing_file.jpg",
            Path("image/missing_file.jpg"),
        ],
        ids=["missing_file_jpg", "missing_file_jpg_Path"],
    )
    def test_should_raise_if_file_not_found(self, resource_path: str | Path) -> None:
        with pytest.raises(FileNotFoundError):
            Image.from_jpeg_file(resolve_resource_path(resource_path))


class TestFromPngFile:
    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/plane.png",
            Path("image/plane.png"),
        ],
        ids=["png", "png_Path"],
    )
    def test_should_load_png_file(self, resource_path: str | Path) -> None:
        Image.from_png_file(resolve_resource_path(resource_path))

    @pytest.mark.parametrize(
        "resource_path",
        [
            "image/missing_file.png",
            Path("image/missing_file.png"),
        ],
        ids=["missing_file_png", "missing_file_png_Path"],
    )
    def test_should_raise_if_file_not_found(self, resource_path: str | Path) -> None:
        with pytest.raises(FileNotFoundError):
            Image.from_png_file(resolve_resource_path(resource_path))


class TestFormat:
    @pytest.mark.parametrize(
        ("image", "format_"),
        [
            (Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg")), ImageFormat.JPEG),
            (Image.from_png_file(resolve_resource_path("image/white_square.png")), ImageFormat.PNG),
        ],
        ids=["jpg", "png"],
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
                Image.from_png_file(resolve_resource_path("image/plane.png")),
                568,
                320,
            ),
        ],
        ids=["[1,1]", "[568,320]"],
    )
    def test_should_return_image_properties(self, image: Image, width: int, height: int) -> None:
        assert image.width == width
        assert image.height == height


class TestToJpegFile:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.jpg"],
        ids=["jpg"],
    )
    def test_should_save_jpeg_file_by_str(self, resource_path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(resource_path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_jpeg_file(tmp_write_file.name)
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_jpeg_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()

    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.jpg"],
        ids=["jpg"],
    )
    def test_should_save_jpeg_file_by_path(self, resource_path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(resource_path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_jpeg_file(Path(tmp_write_file.name))
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_jpeg_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()


class TestToPngFile:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.png"],
        ids=["png"],
    )
    def test_should_save_png_file_by_str(self, resource_path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_png_file(tmp_write_file.name)
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_png_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()

    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.png"],
        ids=["png"],
    )
    def test_should_save_png_file_by_path(self, resource_path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_png_file(Path(tmp_write_file.name))
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_png_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()


class TestReprJpeg:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.jpg"],
        ids=["jpg"],
    )
    def test_should_return_bytes_if_image_is_jpeg(self, resource_path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(resource_path))
        assert isinstance(image._repr_jpeg_(), bytes)

    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.png"],
        ids=["png"],
    )
    def test_should_return_none_if_image_is_not_jpeg(self, resource_path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image._repr_jpeg_() is None


class TestReprPng:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.png"],
        ids=["png"],
    )
    def test_should_return_bytes_if_image_is_png(self, resource_path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert isinstance(image._repr_png_(), bytes)

    @pytest.mark.parametrize(
        "resource_path",
        ["image/white_square.jpg"],
        ids=["jpg"],
    )
    def test_should_return_none_if_image_is_not_png(self, resource_path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(resource_path))
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
        "resource_path",
        ["image/plane.png"],
        ids=["plane"],
    )
    def test_convert_to_grayscale(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.convert_to_grayscale() == snapshot_png


class TestEquals:
    def test_should_be_equal(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/white_square.png"))
        image2 = Image.from_png_file(resolve_resource_path("image/white_square.png"))
        assert image == image2

    def test_should_not_be_equal(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/white_square.png"))
        image2 = Image.from_png_file(resolve_resource_path("image/plane.png"))
        assert image != image2

    def test_should_raise(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/white_square.png"))
        other = Table()
        assert (image.__eq__(other)) is NotImplemented


class TestFlipVertically:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["plane"],
    )
    def test_should_flip_vertically(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.flip_vertically() == snapshot_png

    def test_should_be_original_if_flipped_twice(self) -> None:
        original = Image.from_png_file(resolve_resource_path("image/plane.png"))
        flipped_twice = original.flip_vertically().flip_vertically()
        assert original == flipped_twice


class TestFlipHorizontally:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["plane"],
    )
    def test_should_flip_horizontally(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.flip_horizontally() == snapshot_png

    def test_should_be_original_if_flipped_twice(self) -> None:
        original = Image.from_png_file(resolve_resource_path("image/plane.png"))
        flipped_twice = original.flip_horizontally().flip_horizontally()
        assert original == flipped_twice


class TestAdjustContrast:
    @pytest.mark.parametrize(
        ("resource_path", "factor"),
        [
            ("image/plane.png", 0.75),
            ("image/plane.png", 5)
        ],
        ids=["small factor", "large factor"]
    )
    def test_should_adjust_contrast(self, resource_path: str, factor: float, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.adjust_contrast(factor) == snapshot_png

    def test_should_not_adjust_contrast(self) -> None:
        original = Image.from_png_file(resolve_resource_path("image/plane.png"))
        with pytest.warns(
            UserWarning,
            match="Contrast adjustment factor is 1.0, this will not make changes to the image.",
        ):
            adjusted = original.adjust_contrast(1)
        assert original == adjusted

    def test_should_raise(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/plane.png"))
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
            image.adjust_contrast(-1)


class TestAdjustBrightness:
    @pytest.mark.parametrize(
        ("resource_path", "factor"),
        [
            ("image/plane.png", 0.5),
            ("image/plane.png", 10)
        ],
        ids=["small factor", "large factor"]
    )
    def test_should_adjust_brightness(self, resource_path: str, factor: float, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.adjust_brightness(factor) == snapshot_png

    def test_should_not_brighten(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/plane.png"))
        with pytest.warns(
            UserWarning,
            match="Brightness adjustment factor is 1.0, this will not make changes to the image.",
        ):
            image2 = image.adjust_brightness(1)
        assert image == image2

    def test_should_raise(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/plane.png"))
        with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
            image.adjust_brightness(-1)


class TestInvertColors:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["invert-colors"],
    )
    def test_should_invert_colors(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.invert_colors() == snapshot_png


class TestColorAdjust:
    @pytest.mark.parametrize(
        ("resource_path", "factor"),
        [
            ("image/plane.png", 2),
            ("image/plane.png", 0.5),
            ("image/plane.png", 0),
        ],
        ids=["add color", "remove color", "remove all color"],
    )
    def test_should_adjust_colors(self, resource_path: str, factor: float, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.adjust_color_balance(factor) == snapshot_png

    @pytest.mark.parametrize(
        ("resource_path", "factor"),
        [
            ("image/plane.png", -1),
        ],
        ids=["negative"],
    )
    def test_should_throw(self, resource_path: str, factor: float) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        with pytest.raises(OutOfBoundsError, match=rf"factor \(={factor}\) is not inside \[0, \u221e\)."):
            image.adjust_color_balance(factor)

    @pytest.mark.parametrize(
        ("resource_path", "factor"),
        [
            ("image/plane.png", 1),
        ],
        ids=["no change"],
    )
    def test_should_warn(self, resource_path: str, factor: float) -> None:
        original = Image.from_png_file(resolve_resource_path(resource_path))
        with pytest.warns(
            UserWarning,
            match="Color adjustment factor is 1.0, this will not make changes to the image.",
        ):
            adjusted = original.adjust_color_balance(factor)
        assert adjusted == original


class TestAddGaussianNoise:
    @pytest.mark.parametrize(
        ("resource_path", "standard_deviation"),
        [
            ("image/plane.png", 0.0),
            ("image/plane.png", 0.7),
            ("image/plane.png", 2.5),
        ],
        ids=["minimum noise", "some noise", "very noisy"],
    )
    def test_should_add_noise(self, resource_path: str, standard_deviation: float, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.add_gaussian_noise(standard_deviation) == snapshot_png

    @pytest.mark.parametrize(
        ("resource_path", "standard_deviation"),
        [("image/plane.png", -1)],
        ids=["sigma below zero"],
    )
    def test_should_raise_standard_deviation(self, resource_path: str, standard_deviation: float) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))

        with pytest.raises(
            OutOfBoundsError,
            match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
        ):
            image.add_gaussian_noise(standard_deviation)


class TestBlur:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["blur"],
    )
    def test_should_return_blurred_image(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.blur(2) == snapshot_png


class TestCrop:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["crop"],
    )
    def test_should_return_cropped_image(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.crop(0, 0, 100, 100) == snapshot_png


class TestSharpen:
    @pytest.mark.parametrize(
        ("resource_path", "factor"),
        [
            ("image/plane.png", -1),
            ("image/plane.png", 0.5),
            ("image/plane.png", 10),
        ],
        ids=["negative factor", "small factor", "large factor"],
    )
    def test_should_sharpen(self, resource_path: str, factor: float, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.sharpen(factor) == snapshot_png

    def test_should_not_sharpen_if_factor_is_1(self) -> None:
        image = Image.from_png_file(resolve_resource_path("image/plane.png"))
        image2 = image.sharpen(1)
        assert image == image2


class TestRotate:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["rotate-clockwise"],
    )
    def test_should_return_clockwise_rotated_image(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.rotate_right() == snapshot_png

    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["rotate-counter-clockwise"],
    )
    def test_should_return_counter_clockwise_rotated_image(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.rotate_left() == snapshot_png


class TestFindEdges:
    @pytest.mark.parametrize(
        "resource_path",
        ["image/plane.png"],
        ids=["find_edges"],
    )
    def test_should_return_edges_of_image(self, resource_path: str, snapshot_png) -> None:
        image = Image.from_png_file(resolve_resource_path(resource_path))
        assert image.find_edges() == snapshot_png
