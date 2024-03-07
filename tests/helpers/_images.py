plane_jpg_path = "image/plane.jpg"
plane_png_path = "image/plane.png"
rgba_png_path = "image/rgba.png"
white_square_jpg_path = "image/white_square.jpg"
white_square_png_path = "image/white_square.png"
grayscale_jpg_path = "image/grayscale.jpg"
grayscale_png_path = "image/grayscale.png"

plane_jpg_id = "opaque-3-channel-jpg-plane"
plane_png_id = "opaque-4-channel-png-plane"
rgba_png_id = "transparent-4-channel-png-rgba"
white_square_jpg_id = "opaque-3-channel-jpg-white_square"
white_square_png_id = "opaque-3-channel-png-white_square"
grayscale_jpg_id = "opaque-1-channel-jpg-grayscale"
grayscale_png_id = "opaque-1-channel-png-grayscale"


def images_all() -> list[str]:
    return [
        plane_jpg_path,
        plane_png_path,
        rgba_png_path,
        white_square_jpg_path,
        white_square_png_path,
        grayscale_jpg_path,
        grayscale_png_path,
    ]


def images_all_ids() -> list[str]:
    return [
        plane_jpg_id,
        plane_png_id,
        rgba_png_id,
        white_square_jpg_id,
        white_square_png_id,
        grayscale_jpg_id,
        grayscale_png_id,
    ]


def images_asymmetric() -> list[str]:
    return [
        plane_jpg_path,
        plane_png_path,
        rgba_png_path,
        grayscale_jpg_path,
        grayscale_png_path,
    ]


def images_asymmetric_ids() -> list[str]:
    return [
        plane_jpg_id,
        plane_png_id,
        rgba_png_id,
        grayscale_jpg_id,
        grayscale_png_id,
    ]


def images_all_channel() -> list[str]:
    return [
        plane_jpg_path,
        rgba_png_path,
        grayscale_png_path,
    ]


def images_all_channel_ids() -> list[str]:
    return [
        plane_jpg_id,
        rgba_png_id,
        grayscale_png_id,
    ]
