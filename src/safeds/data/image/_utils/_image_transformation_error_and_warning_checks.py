import warnings

from safeds._validation import _check_bounds, _ClosedBound


def _check_remove_images_with_size_errors(width: int, height: int) -> None:
    _check_bounds("width", width, lower_bound=_ClosedBound(1))
    _check_bounds("height", height, lower_bound=_ClosedBound(1))


def _check_resize_errors(new_width: int, new_height: int) -> None:
    _check_bounds("new_width", new_width, lower_bound=_ClosedBound(1))
    _check_bounds("new_height", new_height, lower_bound=_ClosedBound(1))


def _check_crop_warnings(
    x: int,
    y: int,
    min_width: int,
    min_height: int,
    plural: bool,
) -> None:
    if x >= min_width or y >= min_height:
        warnings.warn(
            f"The specified bounding rectangle does not contain any content of {'at least one' if plural else 'the'} image. Therefore {'these images' if plural else 'the image'} will be blank.",
            UserWarning,
            stacklevel=2,
        )


def _check_crop_errors(
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    _check_bounds("x", x, lower_bound=_ClosedBound(0))
    _check_bounds("y", y, lower_bound=_ClosedBound(0))
    _check_bounds("width", width, lower_bound=_ClosedBound(1))
    _check_bounds("height", height, lower_bound=_ClosedBound(1))


def _check_adjust_brightness_errors_and_warnings(factor: float, plural: bool) -> None:
    _check_bounds("factor", factor, lower_bound=_ClosedBound(0))
    if factor == 1:
        warnings.warn(
            f"Brightness adjustment factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )


def _check_add_noise_errors(standard_deviation: float) -> None:
    _check_bounds("standard_deviation", standard_deviation, lower_bound=_ClosedBound(0))


def _check_adjust_contrast_errors_and_warnings(factor: float, plural: bool) -> None:
    _check_bounds("factor", factor, lower_bound=_ClosedBound(0))
    if factor == 1:
        warnings.warn(
            f"Contrast adjustment factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )


def _check_adjust_color_balance_errors_and_warnings(factor: float, channel: int, plural: bool) -> None:
    _check_bounds("factor", factor, lower_bound=_ClosedBound(0))
    if factor == 1:
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
    _check_bounds("radius", radius, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(max_radius - 1))
    if radius == 0:
        warnings.warn(
            f"Blur radius is 0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )


def _check_sharpen_errors_and_warnings(factor: float, plural: bool) -> None:
    _check_bounds("factor", factor, lower_bound=_ClosedBound(0))
    if factor == 1:
        warnings.warn(
            f"Sharpen factor is 1.0, this will not make changes to the {'images' if plural else 'image'}.",
            UserWarning,
            stacklevel=2,
        )
