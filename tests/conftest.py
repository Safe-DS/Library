import io
from typing import Any

import matplotlib as mpl
import pytest
from PIL.Image import open as open_image
from safeds.data.image.containers import Image, ImageList
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData, SerializedData

# Fix for failures when running pytest in a terminal (https://github.com/Safe-DS/Library/issues/482)
mpl.use("agg")


class JPEGImageExtension(SingleFileSnapshotExtension):
    _file_extension = "jpg"

    def serialize(self, data: Image, **_kwargs: Any) -> SerializedData:
        return data._repr_jpeg_()


@pytest.fixture()
def snapshot_jpeg_image(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(JPEGImageExtension)


class PNGImageSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "png"

    def serialize(self, data: Image, **_kwargs: Any) -> SerializedData:
        return data._repr_png_()

    def matches(
        self,
        *,
        serialized_data: SerializableData,
        snapshot_data: SerializableData,
    ) -> bool:

        # We decode the byte arrays, since torchvision seems to use different compression methods on different operating
        # systems, thus leading to different byte arrays for the same image.
        actual = open_image(io.BytesIO(serialized_data))
        expected = open_image(io.BytesIO(snapshot_data))

        return actual == expected


@pytest.fixture()
def snapshot_png_image(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(PNGImageSnapshotExtension)


class PNGImageListSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "png"

    def serialize(self, data: ImageList, **_kwargs: Any) -> SerializedData:
        return data._repr_png_()

    def matches(
        self,
        *,
        serialized_data: SerializableData,
        snapshot_data: SerializableData,
    ) -> bool:
        # We decode the byte arrays, since torchvision seems to use different compression methods on different operating
        # systems, thus leading to different byte arrays for the same image.
        actual = open_image(io.BytesIO(serialized_data))
        expected = open_image(io.BytesIO(snapshot_data))

        return actual == expected


@pytest.fixture()
def snapshot_png_image_list(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(PNGImageListSnapshotExtension)
