import pytest

from safeds.data.image.typing import ImageFormat


class TestValue:

    @pytest.mark.parametrize(
        "image_format, expected_value",
        [
            (ImageFormat.JPEG, "jpeg"),
            (ImageFormat.PNG, "png"),
        ],
    )
    def test_should_return_correct_value(self, image_format: ImageFormat, expected_value: str) -> None:
        assert image_format.value == expected_value
