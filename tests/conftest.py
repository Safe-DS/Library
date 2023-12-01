from typing import Any

import matplotlib as mpl
import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializedData

from safeds.data.image.containers import ImagePil, Image

# Fix for failures when running pytest in a terminal (https://github.com/Safe-DS/Library/issues/482)
mpl.use("agg")


class JPEGImageExtension(SingleFileSnapshotExtension):
    _file_extension = "jpg"

    def serialize(self, data: ImagePil | Image, **_kwargs: Any) -> SerializedData:
        return data._repr_jpeg_()


@pytest.fixture()
def snapshot_jpeg(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(JPEGImageExtension)


class PNGImageSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "png"

    def serialize(self, data: ImagePil | Image, **_kwargs: Any) -> SerializedData:
        return data._repr_png_()


@pytest.fixture()
def snapshot_png(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(PNGImageSnapshotExtension)
