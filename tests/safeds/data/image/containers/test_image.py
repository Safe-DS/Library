from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from helpers import resolve_resource_path
from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageFormat


class TestFromJpegFile:
    @pytest.mark.parametrize(
        "path",
        [
            "image/white_square.jpg"
        ],
    )
    def test_should_load_jpeg_file(self, path: str) -> None:
        Image.from_jpeg_file(resolve_resource_path(path))

    @pytest.mark.parametrize(
        "path",
        [
            "image/missing_file.jpg"
        ],
    )
    def test_should_raise_if_file_not_found(self, path: str) -> None:
        with pytest.raises(FileNotFoundError):
            Image.from_jpeg_file(resolve_resource_path(path))


class TestFromPngFile:
    @pytest.mark.parametrize(
        "path",
        [
            "image/white_square.png"
        ],
    )
    def test_should_load_png_file(self, path: str) -> None:
        Image.from_png_file(resolve_resource_path(path))

    @pytest.mark.parametrize(
        "path",
        [
            "image/missing_file.png"
        ],
    )
    def test_should_raise_if_file_not_found(self, path: str) -> None:
        with pytest.raises(FileNotFoundError):
            Image.from_png_file(resolve_resource_path(path))


class TestFormat:
    @pytest.mark.parametrize(
        ("image", "format_"),
        [
            (Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg")), ImageFormat.JPEG),
            (Image.from_png_file(resolve_resource_path("image/white_square.png")), ImageFormat.PNG)
        ],
    )
    def test_should_return_correct_format(self, image: Image, format_: ImageFormat) -> None:
        assert image.format == format_


class TestToJpegFile:
    @pytest.mark.parametrize(
        "path",
        [
            "image/white_square.jpg"
        ],
    )
    def test_should_save_jpeg_file(self, path: str) -> None:
        image = Image.from_jpeg_file(resolve_resource_path(path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_jpeg_file(tmp_write_file.name)
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_jpeg_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()


class TestToPngFile:
    @pytest.mark.parametrize(
        "path",
        [
            "image/white_square.png"
        ],
    )
    def test_should_save_png_file(self, path: str) -> None:
        image = Image.from_png_file(resolve_resource_path(path))

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
        with Path(tmp_file.name).open("wb") as tmp_write_file:
            image.to_png_file(tmp_write_file.name)
        with Path(tmp_file.name).open("rb") as tmp_read_file:
            image_read_back = Image.from_png_file(tmp_read_file.name)

        assert image._image.tobytes() == image_read_back._image.tobytes()


class TestReprJpeg:
    @pytest.mark.parametrize(
        "image",
        [
            Image.from_png_file(resolve_resource_path("image/white_square.png"))
        ],
    )
    def test_should_return_none_if_image_is_not_jpeg(self, image: Image) -> None:
        assert image._repr_jpeg_() is None


class TestReprPng:
    @pytest.mark.parametrize(
        "image",
        [
            Image.from_jpeg_file(resolve_resource_path("image/white_square.jpg"))
        ],
    )
    def test_should_return_none_if_image_is_not_png(self, image: Image) -> None:
        assert image._repr_png_() is None
