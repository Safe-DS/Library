import warnings

from safeds.exceptions import ClosedBound, OutOfBoundsError


def _check_remove_images_with_size_errors(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise OutOfBoundsError(min(width, height), name="At least one of width and height", lower_bound=ClosedBound(1))


def _check_resize_errors(new_width: int, new_height: int) -> None:
    if new_width <= 0 or new_height <= 0:
        raise OutOfBoundsError(
            min(new_width, new_height),
            name="At least one of the new sizes new_width and new_height",
            lower_bound=ClosedBound(1),
        )


def _check_crop_errors_and_warnings(
    x: int,
    y: int,
    width: int,
    height: int,
    min_width: int,
    min_height: int,
    plural: bool,
) -> None:
    if x < 0 or y < 0:
        raise OutOfBoundsError(min(x, y), name="At least one of the coordinates x and y", lower_bound=ClosedBound(0))
    if width <= 0 or height <= 0:
        raise OutOfBoundsError(min(width, height), name="At least one of width and height", lower_bound=ClosedBound(1))
    if x >= min_width or y >= min_height:
        warnings.warn(
            f"The specified bounding rectangle does not contain any content of {'at least one' if plural else 'the'} image. Therefore {'these images' if plural else 'the image'} will be blank.",
            UserWarning,
            stacklevel=2,
        )


def _check_adjust_brightness_errors_and_warnings(factor: float, plural: bool) -> None:
    if factor < 0:
        raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
    elif factor == 1:
        warnings.warn(
            f"Brightness adjustment factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )


def _check_add_noise_errors(standard_deviation: float) -> None:
    if standard_deviation < 0:
        raise OutOfBoundsError(standard_deviation, name="standard_deviation", lower_bound=ClosedBound(0))


def _check_adjust_contrast_errors_and_warnings(factor: float, plural: bool) -> None:
    if factor < 0:
        raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
    elif factor == 1:
        warnings.warn(
            f"Contrast adjustment factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )


def _check_adjust_color_balance_errors_and_warnings(factor: float, channel: int, plural: bool) -> None:
    if factor < 0:
        raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
    elif factor == 1:
        warnings.warn(
            f"Color adjustment factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )
    elif channel == 1:
        warnings.warn(
            "Color adjustment will not have an affect on grayscale images with only one channel.",
            UserWarning,
            stacklevel=2,
        )


def _check_blur_errors_and_warnings(radius: int, max_radius: int, plural: bool) -> None:
    if radius < 0 or radius >= max_radius:
        raise OutOfBoundsError(
            radius,
            name="radius",
            lower_bound=ClosedBound(0),
            upper_bound=ClosedBound(max_radius - 1),
        )
    elif radius == 0:
        warnings.warn(
            f"Blur radius is 0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )


def _check_sharpen_errors_and_warnings(factor: float, plural: bool) -> None:
    if factor < 0:
        raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
    elif factor == 1:
        warnings.warn(
            f"Sharpen factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )
