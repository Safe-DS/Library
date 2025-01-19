"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from pathlib import Path

from safeds.exceptions import FileExtensionError


def _normalize_and_check_file_path(
    path: str | Path,
    canonical_file_extension: str,
    valid_file_extensions: list[str],
    *,
    check_if_file_exists: bool = False,
) -> Path:
    """
    Normalize a path and check its validity.

    Parameters
    ----------
    path:
        Path to normalize and check.
    canonical_file_extension:
        If the path has no extension, this extension will be added. It should include the leading dot.
    valid_file_extensions:
        If the path has an extension, it must be in this set. It should include the leading dots.
    check_if_file_exists:
        Whether to also check if the path points to an existing file.

    Returns
    -------
    normalized_path:
        The normalized path.

    Raises
    ------
    FileExtensionError
        If the path has an extension that is not in the `valid_file_extensions` list.
    FileNotFoundError
        If `check_if_file_exists` is True and the file does not exist.
    """
    path = Path(path)

    # Normalize and check file extension
    if not path.suffix:
        path = path.with_suffix(canonical_file_extension)
    elif path.suffix not in valid_file_extensions:
        message = _build_file_extension_error_message(path.suffix, valid_file_extensions)
        raise FileExtensionError(message) from None

    # Check if file exists
    if check_if_file_exists and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def _build_file_extension_error_message(actual_file_extension: str, valid_file_extensions: list[str]) -> str:
    return f"Expected path with extension in {valid_file_extensions} but got {actual_file_extension}."
