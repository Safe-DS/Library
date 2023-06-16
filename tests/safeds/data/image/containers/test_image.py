from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageFormat

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
                Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg")),
                2,
                3,
                (2, 3),
            ),
            (
                Image.from_png_file(resolve_resource_path("image/white_square.png")),
                2,
                3,
                (2, 3),
            ),
        ],
        ids=[".jpg", ".png"],
    )
    def test_should_return_resized_image(
        self,
        image: Image,
        new_width: int,
        new_height: int,
        new_size: tuple[int, int],
    ) -> None:
        assert image.resize(new_width, new_height)._image.size == new_size
