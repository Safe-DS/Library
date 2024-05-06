from pathlib import Path

from safeds.exceptions import WrongFileExtensionError


def _check_and_normalize_file_path(
    path: str | Path,
    canonical_file_extension: str,
    valid_file_extensions: list[str],  # Should be ordered to ensure consistent error messages.
    *,
    check_if_file_exists: bool = False,
) -> Path:  # pragma: no cover
    """
    Check if the provided path is a valid file path and normalize it.

    Parameters
    ----------
    path:
        Path to check and normalize.
    canonical_file_extension:
        If the path has no extension, this extension will be added. Should include the leading dot.
    valid_file_extensions:
        If the path has an extension, it must be in this set. Should include the leading dots.
    check_if_file_exists:
        Whether to also check if the path points to an existing file.

    Returns
    -------
    normalized_path:
        The normalized path.

    Raises
    ------
    ValueError
        If the path has an extension that is not in the `valid_file_extensions` list.
    FileNotFoundError
        If `check_if_file_exists` is True and the file does not exist.
    """
    path = Path(path)

    # Normalize and check file extension
    if not path.suffix:
        path = path.with_suffix(canonical_file_extension)
    elif path.suffix not in valid_file_extensions:
        raise WrongFileExtensionError(path, valid_file_extensions)

    # Check if file exists
    if check_if_file_exists and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path
